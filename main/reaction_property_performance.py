"""
# =============================================================================
# reaction_property_performance.V.2.7.0.py
# =============================================================================
#
# Purpose
# -------
# XTQSPRKinetic reaction-property performance evaluation utility.
#
# This module evaluates an XTQSPRKinetic-generated pipeline (.pkl) on a reaction
# network dataset and exports predictions and performance summaries. It is a
# post-processing/comparison utility that reuses the stored pipeline configuration,
# routing/family assignments, target definitions, feature definitions, and model
# artifacts for reproducible evaluation without retraining.
#
# The v2.7.0 evaluator supports the generalized target_cols workflow introduced
# in XTQSPRKinetic v2.7.0. The legacy Arrhenius set ['lnA', 'n', 'Ea'] remains
# the default case study and fallback for older pipeline.pkl files.
#
# Key Principles (Reproducibility Contract)
# -----------------------------------------
# - No retraining, no re-clustering, no feature selection:
#   pipeline.pkl is the single source of truth.
# - The evaluated reaction network is mapped ONLY into the existing learned topic
#   space and predicted using the existing family/global models from the pickle.
# - Target names are recovered from pipeline.pkl['target_cols'] or
#   config_snapshot['target_cols']; if unavailable, ['lnA', 'n', 'Ea'] is used.
# - Features are computed ONLY as needed to reconstruct the pipeline's required
#   input columns:
#     * Mordred descriptors (per pipeline config_snapshot; sum/delta/concat modes)
#     * DRFP fingerprints (n_bits / radius taken from config_snapshot)
#     * Reaction flags and hydrogen-radical descriptors as stored in the pipeline
# - Scaling strictly follows pipeline settings:
#     * If config_snapshot['use_family_scaler'] is False, the global scaler is
#       used for all models.
#     * Otherwise, the family scaler is used when available, with fallback to
#       the global scaler.
#
# Inputs
# ------
# 1) pipeline.pkl
#    Produced by XTQSPRKinetic and contains:
#      - Models (global + per-family)
#      - Scalers (global + optional per-family)
#      - Final feature list and optional per-family feature subsets
#      - Topic routing method and assigner objects
#      - target_cols metadata for v2.7.0+ pipelines
#      - config_snapshot with feature / routing / scaling settings
#
# 2) Reaction network CSV (required)
#    Must contain at minimum:
#      - reaction_smiles : string formatted as "reactants>>products"
#
#    Optional columns for computing metrics:
#      - all target columns used by the trained pipeline
#        e.g., lnA, n, Ea for the default Arrhenius case study
#
# Outputs
# -------
# Written to the selected output directory:
# - <input_basename>_property_performance.xlsx
#     Sheets may include:
#       * Predictions: <target>_pred and optional per-row errors
#       * Global_Metrics: R2, AdjR2, MAE, RMSE, nRMSE, AARD, RAE by target
#       * Family_Metrics: same metrics by predicted family
# - topic_proba_pred.csv
# - family_assignments_pred.csv
# - eval_summary.txt
# - property_eval.log
#
# Typical Usage
# -------------
# python reaction_property_performance.py \
#     --input_csv RS_1.csv \
#     --pipeline_pkl pipeline.pkl \
#     --output_dir eval_output
#
# Compatibility
# -------------
# - Backward-compatible with v2.6.x Arrhenius pipelines.
# - For older pipeline.pkl files lacking target_cols, the evaluator falls back to
#   ['lnA', 'n', 'Ea'] unless target_cols is supplied in the evaluator config.
#
# =============================================================================
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Hamidreza Najafi / XThermo Lab
# Maintainer: Hamidreza Najafi (XThermo Lab)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import importlib.util
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# -------- Optional deps (handled gracefully) --------
try:
    from drfp import DrfpEncoder
except Exception:
    DrfpEncoder = None

try:
    import skfuzzy as fuzz  # for FCM membership prediction
except Exception:
    fuzz = None

try:
    from hdbscan import prediction as hdbscan_prediction
except Exception:
    hdbscan_prediction = None

try:
    from mordred import Calculator, descriptors
    from rdkit import Chem
except Exception:
    Calculator = None
    descriptors = None
    Chem = None
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
    from XTQSPRKinetic.src.mordred_xtqspr_patch import apply_xtqspr_mordred_patch
except Exception:
    apply_xtqspr_mordred_patch = None

try:
    from XTQSPRKinetic.src.utils import (
        _adj_r2, _nrmse, _aard, _rae,
        _yaml_safe_load_with_duplicate_warnings,
        log_yaml_duplicate_key_summary,
    )
except Exception:
    _yaml_safe_load_with_duplicate_warnings = None
    log_yaml_duplicate_key_summary = None
    _adj_r2 = None
    _nrmse = None
    _aard = None
    _rae  = None

from XTQSPRKinetic.src.xtqspr_models import MultiOutputXGBRegressor

try:
    from XTQSPRKinetic.src.featurization import (
        build_hydrogen_radical_feature_df as _xt_build_hydrogen_radical_feature_df,
        _repair_hydrogen_radical_zero_drfp as _xt_repair_hydrogen_radical_zero_drfp,
        calculate_descriptors_mordred as _xt_calculate_descriptors_mordred,
    )
except Exception:
    _xt_build_hydrogen_radical_feature_df = None
    _xt_repair_hydrogen_radical_zero_drfp = None
    _xt_calculate_descriptors_mordred = None


# =========================== CLI / CONFIG / LOGGING ===========================

DEFAULT_TARGET_COLS = ["lnA", "n", "Ea"]

DEFAULT_CONFIG = {
    "input_csv": "RS_1.csv",                 # must contain reaction_smiles; targets optional for scoring
    "pipeline_pkl": "pipeline.pkl",
    "output_dir": "eval_output",
    # Optional target override for old pickles without target_cols metadata.
    # Prefer the target list saved in pipeline.pkl whenever present.
    "target_cols": "",                       # comma-separated string or YAML/JSON list; empty => infer from pipeline
    # Routing for inference:
    "routing_mode": "auto",                  # auto | soft | hard | hybrid
    "hybrid_metric": "entropy",              # entropy | vote_fraction
    "hybrid_threshold": 0.35,                # used if routing_mode is hybrid (or auto->hybrid)
    # External hybrid routing policy:
    # Training-time Bootstrap hybrid routing uses bootstrap stability diagnostics that
    # are not available for new evaluation reactions. In auto mode, keep this False
    # to fall back to soft routing when the stored pipeline used Bootstrap-hybrid.
    # Set True only if you explicitly want an approximate hybrid rule based on
    # current routing probabilities (entropy / max probability).
    "allow_hybrid_approximation": False,
    # Normalize routing probability rows after reconstructing topic/family probabilities.
    "normalize_routing_probabilities": True,
    # DRFP fallback (used only if not fully specified in pickle):
    "drfp_n_bits": 2048,
    "drfp_radius": 3,
    # Mordred descriptor parallelization for evaluation feature reconstruction.
    # These are applied only when Mordred descriptors must be recomputed for the
    # evaluation dataset; they do not retrain models or change pipeline.pkl.
    "mordred_parallel_enable": False,
    "mordred_parallel_n_jobs": -1,
    "mordred_parallel_backend": "loky",
    "mordred_parallel_batch_size": "auto",
    "mordred_xtqspr_explicit_h_patch_enable": "auto",
    # Logging:
    "log_file": "property_eval.log",
    # Utility:
    "help_config": False
}

def _bool_or_auto(value: Any, default: bool = False) -> bool:
    """
    Resolve bool-like values while allowing 'auto' to mean: use pipeline snapshot.
    """
    if isinstance(value, bool):
        return value

    if value is None:
        return bool(default)

    s = str(value).strip().lower()

    if s in {"auto", "none", "null", ""}:
        return bool(default)

    if s in {"yes", "y", "true", "t", "1", "on"}:
        return True

    if s in {"no", "n", "false", "f", "0", "off"}:
        return False

    return bool(default)

def parse_args() -> Dict[str, Any]:
    def _str2bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"yes", "y", "true", "t", "1", "on"}:
            return True
        if s in {"no", "n", "false", "f", "0", "off"}:
            return False
        raise argparse.ArgumentTypeError("Expected boolean value: true/false.")

    p = argparse.ArgumentParser(
        description="Evaluate XTQSPRKinetic v2.7.0 reaction-property pipeline from pipeline.pkl on a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--config", type=str, help="YAML/JSON config file with overrides.")
    # Add direct flags for all DEFAULT_CONFIG keys. Boolean options default to None
    # so an omitted CLI flag does not overwrite DEFAULT_CONFIG or config-file values.
    for k, v in DEFAULT_CONFIG.items():
        if k == "help_config":
            p.add_argument("--help_config", action="store_true", help="Print configurable keys and exit.")
            continue
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=_str2bool, default=None, metavar="{true|false}",
                           help=f"(bool) {k.replace('_',' ')}")
        else:
            p.add_argument(f"--{k}", type=type(v), help=k.replace('_', ' '))

    args = p.parse_args()
    if getattr(args, "help_config", False):
        print(json.dumps(DEFAULT_CONFIG, indent=2))
        sys.exit(0)

    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        config_path = os.path.abspath(os.path.expanduser(args.config))
        duplicate_keys = []

        if config_path.lower().endswith((".yml", ".yaml")):
            if _yaml_safe_load_with_duplicate_warnings is not None:
                cfg_file, duplicate_keys = _yaml_safe_load_with_duplicate_warnings(config_path)
            else:
                with open(config_path, "r", encoding="utf-8") as f:
                    import yaml  # type: ignore
                    cfg_file = yaml.safe_load(f)
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_file = json.load(f)

        if isinstance(cfg_file, dict):
            cfg.update({k: v for k, v in cfg_file.items() if v is not None})

        if duplicate_keys:
            cfg["_yaml_duplicate_keys"] = duplicate_keys
            cfg["_yaml_duplicate_key_count"] = len(duplicate_keys)

    # CLI overrides win
    for k, v in vars(args).items():
        if k in cfg and v is not None and k != "help_config":
            cfg[k] = v

    # Normalize paths & strings
    cfg["pipeline_pkl"] = os.path.abspath(os.path.expanduser(cfg["pipeline_pkl"]))
    cfg["input_csv"] = os.path.abspath(os.path.expanduser(cfg["input_csv"]))
    cfg["output_dir"] = os.path.abspath(os.path.expanduser(cfg["output_dir"]))
    cfg["routing_mode"] = str(cfg["routing_mode"]).lower().strip()
    cfg["hybrid_metric"] = str(cfg["hybrid_metric"]).lower().strip()
    return cfg


def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
    )


# ================================ UTILITIES ===================================

def _as_clean_list(value: Any) -> List[str]:
    """Convert config/pickle target specifications into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"none", "null", "auto"}:
            return []
        # Accept JSON/YAML-like list strings, or comma-separated strings.
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _infer_n_outputs_from_models(pipe: Dict[str, Any]) -> Optional[int]:
    """Infer model output dimension from fitted MultiOutput-style estimators when possible."""
    models = pipe.get("models", {}) or {}
    for key in ("global",) + tuple(k for k in models.keys() if k != "global"):
        model = models.get(key)
        if model is None:
            continue
        n = getattr(model, "n_outputs_", None)
        if n is not None:
            try:
                return int(n)
            except Exception:
                pass
        ests = getattr(model, "estimators_", None)
        if isinstance(ests, (list, tuple)) and len(ests) > 0:
            return int(len(ests))
    return None


def _target_cols_from_pipe(pipe: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Resolve target columns for evaluation.

    Priority:
      1. pipeline.pkl['target_cols']
      2. pipeline.pkl['config_snapshot']['target_cols']
      3. evaluator config['target_cols']
      4. pipeline.pkl['config_snapshot']['hp_target_cols']
      5. DEFAULT_TARGET_COLS for legacy Arrhenius pipelines
    """
    cfg = cfg or {}
    cs = pipe.get("config_snapshot", {}) or {}
    candidates = [
        pipe.get("target_cols", None),
        cs.get("target_cols", None),
        cfg.get("target_cols", None),
        cs.get("hp_target_cols", None),
        DEFAULT_TARGET_COLS,
    ]
    target_cols: List[str] = []
    for cand in candidates:
        target_cols = _as_clean_list(cand)
        if target_cols:
            break
    if not target_cols:
        target_cols = list(DEFAULT_TARGET_COLS)

    n_out = _infer_n_outputs_from_models(pipe)
    if n_out is not None and n_out > 0 and n_out != len(target_cols):
        logging.warning(
            "Resolved %d target names but model appears to have %d outputs; adjusting names for safe evaluation.",
            len(target_cols), n_out,
        )
        if len(target_cols) > n_out:
            target_cols = target_cols[:n_out]
        else:
            used = set(target_cols)
            j = 0
            while len(target_cols) < n_out:
                name = f"target_{j}"
                j += 1
                if name not in used:
                    target_cols.append(name)
                    used.add(name)
    return list(target_cols)


def _prediction_columns(target_cols: Sequence[str]) -> List[str]:
    return [f"{str(t)}_pred" for t in target_cols]


def _coerce_prediction_vector(pred: Any, n_targets: int) -> np.ndarray:
    """Return one finite-length prediction vector, padding/truncating to n_targets if needed."""
    arr = np.asarray(pred, dtype=float).reshape(-1)
    out = np.full(int(n_targets), np.nan, dtype=float)
    m = min(int(n_targets), int(arr.size))
    if m > 0:
        out[:m] = arr[:m]
    return out


def _has_targets(df: pd.DataFrame, target_cols: Sequence[str]) -> bool:
    return all(str(c) in df.columns for c in target_cols)


def _normalized_entropy_row(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = np.clip(p, 1e-12, 1.0)
    denom = float(p.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        p = np.ones_like(p, dtype=float) / float(max(1, p.size))
    else:
        p /= denom
    H = -(p * np.log(p)).sum()
    return float(H / math.log(p.size)) if p.size > 1 else 0.0


def _normalize_probability_frame(proba_df: pd.DataFrame, eps: float = 1e-15) -> pd.DataFrame:
    """Return finite, non-negative, row-normalized routing probabilities."""
    if proba_df is None or proba_df.empty:
        return proba_df
    arr = np.asarray(proba_df.values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0.0] = 0.0
    if arr.shape[1] == 0:
        arr = np.ones((arr.shape[0], 1), dtype=float)
    rs = arr.sum(axis=1, keepdims=True)
    zero = (rs <= eps)
    if np.any(zero):
        arr[zero.ravel(), :] = 1.0 / float(arr.shape[1])
        rs = arr.sum(axis=1, keepdims=True)
    arr = arr / rs
    return pd.DataFrame(arr, index=proba_df.index, columns=proba_df.columns)


def _pipeline_bootstrap_hybrid_requires_training_stability(pipe: Dict[str, Any]) -> bool:
    """True when stored training routing was Bootstrap-guided hybrid."""
    cs = pipe.get("config_snapshot", {}) or {}
    return (
        str(pipe.get("routing_mode", "")).lower() == "hybrid"
        and bool(cs.get("bootstrap_use_for_hybrid_routing", True))
        and pipe.get("bootstrap_stability_columns", None) is not None
    )


def _pipeline_soft_used_bootstrap_probabilities(pipe: Dict[str, Any]) -> bool:
    """True when training soft routing used bootstrap-aggregated probabilities."""
    cs = pipe.get("config_snapshot", {}) or {}
    return (
        str(pipe.get("routing_mode", "")).lower() == "soft"
        and bool(cs.get("bootstrap_use_for_soft_routing", False))
        and pipe.get("bootstrap_agg_topic_proba_shape", None) is not None
    )


def _union_features(global_feats: Sequence[str],
                    features_by_family: Optional[Dict[Union[int, str], Sequence[str]]]) -> List[str]:
    s = set(global_feats or [])
    if features_by_family:
        for v in features_by_family.values():
            s.update(list(v))
    return sorted(s)


def _infer_model_key(keys: Sequence[Union[int, str]], fam_idx: int) -> Optional[Union[int, str]]:
    """Map a family index to an available model/scaler key (int or string)."""
    candidates: List[Union[int, str]] = [
        fam_idx, str(fam_idx), f"family_{fam_idx}", f"T{fam_idx}"
    ]
    for c in candidates:
        if c in keys:
            return c
    # try any numeric-looking string keys equal to the index
    for k in keys:
        try:
            if int(k) == fam_idx:
                return k
        except Exception:
            pass
    return None


def _family_features_for_key(
    features_by_family: Optional[Dict[Union[int, str], Sequence[str]]],
    model_key: Any,
    fallback: Sequence[str],
) -> List[str]:
    """Return per-family features while tolerating int/string family keys."""
    if not features_by_family:
        return list(fallback)

    candidates = [model_key, str(model_key)]
    try:
        candidates.append(int(model_key))
    except Exception:
        pass

    for c in candidates:
        if c in features_by_family:
            return list(features_by_family[c])

    return list(fallback)


def _fallback_hydrogen_radical_features(rxn_smiles: str) -> Dict[str, float]:
    """Local v2.6.0-compatible fallback for [H].[H]>>[H][H] descriptors."""
    base = {
        "rxn_H_atom_delta": 0.0,
        "rxn_H_radical_electron_delta": 0.0,
        "rxn_HH_bond_delta": 0.0,
        "rxn_isolated_H_radical_atom_delta": 0.0,
        "rxn_is_2H_radicals_to_H2": 0.0,
    }
    if Chem is None:
        return base

    def _mol_keep_h(smi: str):
        try:
            m = Chem.MolFromSmiles(str(smi).strip(), sanitize=False)
            if m is None:
                return None
            Chem.SanitizeMol(m)
            return m
        except Exception:
            try:
                return Chem.MolFromSmiles(str(smi).strip())
            except Exception:
                return None

    def _side_stats(mols) -> np.ndarray:
        n_h_atoms = 0.0
        n_h_rad = 0.0
        n_hh = 0.0
        n_iso_h_rad = 0.0
        for mol in mols:
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    n_h_atoms += 1.0
                    n_h_rad += float(atom.GetNumRadicalElectrons())
                    if atom.GetDegree() == 0 and atom.GetNumRadicalElectrons() > 0:
                        n_iso_h_rad += 1.0
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                if a1.GetAtomicNum() == 1 and a2.GetAtomicNum() == 1:
                    n_hh += 1.0
        return np.array([n_h_atoms, n_h_rad, n_hh, n_iso_h_rad], dtype=float)

    try:
        left, right = str(rxn_smiles).split(">>", 1) if ">>" in str(rxn_smiles) else (str(rxn_smiles), "")
        R = _side_stats([_mol_keep_h(x) for x in left.split(".") if str(x).strip()])
        P = _side_stats([_mol_keep_h(x) for x in right.split(".") if str(x).strip()])
        d = P - R
        is_2h_to_h2 = (
            R[0] == 2.0 and R[1] == 2.0 and R[2] == 0.0 and R[3] == 2.0 and
            P[0] == 2.0 and P[1] == 0.0 and P[2] == 1.0 and P[3] == 0.0
        )
        base.update({
            "rxn_H_atom_delta": float(d[0]),
            "rxn_H_radical_electron_delta": float(d[1]),
            "rxn_HH_bond_delta": float(d[2]),
            "rxn_isolated_H_radical_atom_delta": float(d[3]),
            "rxn_is_2H_radicals_to_H2": float(bool(is_2h_to_h2)),
        })
    except Exception:
        pass
    return base


def _hydrogen_radical_feature_df(rxn_series: pd.Series) -> pd.DataFrame:
    """Build hydrogen-radical descriptors using package helper when available."""
    if _xt_build_hydrogen_radical_feature_df is not None:
        try:
            return _xt_build_hydrogen_radical_feature_df(rxn_series)
        except Exception:
            pass
    rows = [_fallback_hydrogen_radical_features(rxn) for rxn in rxn_series]
    return pd.DataFrame(rows, index=rxn_series.index, dtype=float)


def _repair_hydrogen_radical_zero_drfp_local(vector: np.ndarray, rxn_smiles: str) -> np.ndarray:
    """Repair all-zero DRFP for [H].[H]>>[H][H] using package helper when available."""
    if _xt_repair_hydrogen_radical_zero_drfp is not None:
        try:
            return _xt_repair_hydrogen_radical_zero_drfp(vector, rxn_smiles)
        except Exception:
            pass

    v = np.asarray(vector).copy()
    if v.ndim > 1:
        v = v.flatten()
    n = int(v.size)
    if n <= 0:
        return v
    try:
        is_zero = float(np.nansum(np.abs(v))) == 0.0
    except Exception:
        is_zero = False
    is_h2 = _fallback_hydrogen_radical_features(rxn_smiles).get("rxn_is_2H_radicals_to_H2", 0.0) > 0.5
    if is_zero and is_h2:
        for tag in (
            "XT_HRAD_2H_TO_H2",
            "XT_HRAD_RADICAL_ELECTRONS_CONSUMED",
            "XT_HRAD_HH_BOND_FORMED",
        ):
            j = int(hashlib.sha1(tag.encode("utf-8")).hexdigest(), 16) % n
            v[j] = 1
    return v


# ============================ FEATURE ENGINEERING ==============================

def _compute_mordred_agg(
    rxn_series: pd.Series,
    nan_policy: str = "mean",
    agg: str = "sum",
    parallel_enable: bool = False,
    n_jobs: int = 1,
    backend: str = "loky",
    batch_size: Union[int, str] = "auto",
    xtqspr_explicit_h_patch_enable: bool = False,
) -> pd.DataFrame:
    """
    Compute Mordred descriptors aggregated over reactants/products.

    This mirrors the behavior in XTQSPRKinetic v2.7.0:
    - uses XT_MORDRED_3D env toggle
    - supports agg in {'sum', 'concat', 'delta'}
    - applies desc_nan_policy at the descriptor level
    """
    # Prefer the XTQSPRKinetic package implementation when available so the
    # evaluator follows the same Mordred code path used during training.
    # If the import is absent or incompatible, fall back to the local sequential
    # implementation below.
    if _xt_calculate_descriptors_mordred is not None:
        try:
            return _xt_calculate_descriptors_mordred(
                rxn_series,
                nan_policy=nan_policy,
                agg=agg,
                parallel_enable=bool(parallel_enable),
                n_jobs=int(n_jobs),
                backend=str(backend),
                batch_size=batch_size,
                xtqspr_explicit_h_patch_enable=bool(xtqspr_explicit_h_patch_enable),
            )
        except TypeError:
            # Older split-package versions may not yet expose the v2.7.0 parallel
            # Mordred signature. Continue with the local fallback.
            logging.warning(
                "Imported calculate_descriptors_mordred does not support the v2.7.0 "
                "parallel signature; using evaluator local Mordred fallback."
            )
        except Exception as e:
            logging.warning(
                "Imported calculate_descriptors_mordred failed during evaluation; "
                "using evaluator local Mordred fallback. Error: %s",
                e,
            )

    if bool(parallel_enable) and int(n_jobs) != 1:
        logging.warning(
            "Evaluator local Mordred fallback is sequential. To use parallel Mordred, "
            "run the evaluator inside the XTQSPRKinetic v2.7.0 split-package environment."
        )

    if Calculator is None or descriptors is None or Chem is None:
        raise RuntimeError("Mordred/RDKit are required to compute descriptors.")

    use3d = os.environ.get("XT_MORDRED_3D", "").strip().lower() in ("1", "true", "yes", "on")

    # Full Mordred calculator
    if bool(xtqspr_explicit_h_patch_enable):
        if apply_xtqspr_mordred_patch is None:
            raise RuntimeError(
                "mordred_xtqspr_explicit_h_patch_enable=True, but "
                "XTQSPRKinetic.src.mordred_xtqspr_patch could not be imported."
            )
        apply_xtqspr_mordred_patch()
    calc = Calculator(descriptors, ignore_3D=not use3d)
    desc_objs = list(calc.descriptors)
    desc_names = [str(d) for d in desc_objs]
    ndesc = len(desc_objs)

    def safe_values(mol):
        if mol is None:
            return [np.nan] * ndesc
        try:
            m_for_calc = mol
            # Optional 3D embedding (if enabled)
            if use3d:
                try:
                    from rdkit.Chem import AllChem  # type: ignore
                    m3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(m3d, randomSeed=0xF00D)
                    AllChem.UFFOptimizeMolecule(m3d, maxIters=200)
                    m_for_calc = m3d
                except Exception:
                    m_for_calc = mol
            dd = calc(m_for_calc).asdict()
        except Exception:
            return [np.nan] * ndesc

        out = []
        for name in desc_names:
            v = dd.get(name, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            out.append(v)
        return out

    rows = []
    index = []

    agg = str(agg).lower()
    if agg not in ("sum", "concat", "delta"):
        logging.warning("Unknown mordred_agg='%s'; falling back to 'sum'.", agg)
        agg = "sum"

    for idx, rxn in rxn_series.items():
        try:
            reactant_smiles, product_smiles = rxn.split(">>")
        except ValueError:
            reactant_smiles, product_smiles = rxn, ""

        reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles.split(".") if s]
        product_mols  = [Chem.MolFromSmiles(s) for s in product_smiles.split(".") if s]

        R_mat = [safe_values(m) for m in reactant_mols] if reactant_mols else []
        P_mat = [safe_values(m) for m in product_mols]  if product_mols  else []

        R = np.nansum(R_mat, axis=0) if len(R_mat) > 0 else np.zeros(ndesc)
        P = np.nansum(P_mat, axis=0) if len(P_mat) > 0 else np.zeros(ndesc)

        if agg == "concat":
            data = np.concatenate([R, P])
            cols = [f"R_{n}" for n in desc_names] + [f"P_{n}" for n in desc_names]
        elif agg == "delta":
            data = (P - R)
            cols = [f"{n}_delta" for n in desc_names]
        else:  # "sum"
            data = (R + P)
            cols = [f"{n}_sum" for n in desc_names]

        rows.append(pd.Series(data=data, index=cols, dtype=float))
        index.append(idx)

    desc_df = pd.DataFrame(rows, index=index)
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)

    # Apply desc_nan_policy exactly as in training
    nan_policy = str(nan_policy).lower()
    if nan_policy == "mean":
        desc_df = desc_df.fillna(desc_df.mean(numeric_only=True))
    elif nan_policy == "median":
        desc_df = desc_df.fillna(desc_df.median(numeric_only=True))
    elif nan_policy == "zero":
        desc_df = desc_df.fillna(0.0)
    elif nan_policy == "drop":
        # In training this is used rarely; we keep column-wise drop
        desc_df = desc_df.dropna(axis=1, how="any")
    else:
        logging.warning("Unknown desc_nan_policy='%s'; defaulting to 'mean'.", nan_policy)
        desc_df = desc_df.fillna(desc_df.mean(numeric_only=True))

    # Keep only numeric columns
    desc_df = desc_df.select_dtypes(include=[np.number]).copy()
    return desc_df


def _compute_drfp(rxn_series: pd.Series,
                  n_bits: int,
                  radius: int,
                  want_cols: Optional[List[str]] = None,
                  hydrogen_patch: bool = False) -> pd.DataFrame:
    """
    Compute DRFP vectors for a Series of reaction SMILES, matching the behavior
    used in XTQSPRKinetic v2.7.0 (DrfpEncoder() with encode()).

    - Uses DrfpEncoder() with no __init__ arguments (new drfp API).
    - Tries encode(..., n_folded_length=n_bits, radius=radius); if that raises
      TypeError (older drfp), falls back to encode(rxn) with its own defaults.
    - Columns are named drfp_0, drfp_1, ...
    - If want_cols is provided, we ensure all of those columns exist, inserting
      missing ones as zeros, and then reorder to want_cols.
    """
    if DrfpEncoder is None:
        raise RuntimeError("DRFP is required but not installed.")

    encoder = DrfpEncoder()
    vectors = []

    for rxn in rxn_series:
        try:
            v = encoder.encode(rxn, n_folded_length=n_bits, radius=radius)
        except TypeError:
            # For older drfp versions where encode() has no n_folded_length/radius
            v = encoder.encode(rxn)
        v = np.asarray(v, dtype=float)
        if v.ndim > 1:
            v = v.flatten()
        if bool(hydrogen_patch):
            v = _repair_hydrogen_radical_zero_drfp_local(v, rxn)
        vectors.append(v)

    if not vectors:
        # No reactions → empty frame with no rows
        return pd.DataFrame(index=rxn_series.index)

    mat = np.stack(vectors)  # (N, n_bits) or whatever drfp returns
    cols = [f"drfp_{i}" for i in range(mat.shape[1])]
    df = pd.DataFrame(mat, index=rxn_series.index, columns=cols)

    if want_cols is not None:
        # add any missing requested bits as zeros
        for c in want_cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[want_cols]

    return df

def _compute_flags(rxn_series: pd.Series, want_cols: List[str], hydrogen_patch: bool = False) -> pd.DataFrame:
    flags = pd.DataFrame(index=rxn_series.index)
    if "is_unimolecular" in want_cols:
        flags["is_unimolecular"] = [1 if len(str(r).split(">>")[0].split(".")) == 1 else 0 for r in rxn_series]
    if "has_competing_products" in want_cols:
        flags["has_competing_products"] = [
            1 if (">>" in str(r) and len(str(r).split(">>", 1)[1].split(".")) > 1) else 0
            for r in rxn_series
        ]

    if bool(hydrogen_patch) and any(str(c).startswith("rxn_H") or str(c).startswith("rxn_isolated_H") or str(c) == "rxn_is_2H_radicals_to_H2" for c in want_cols):
        hrad = _hydrogen_radical_feature_df(rxn_series)
        flags = pd.concat([flags, hrad], axis=1)

    # Ensure all requested flags exist
    for c in want_cols:
        if c not in flags.columns:
            flags[c] = 0.0
    return flags[want_cols]

def compute_required_features(
    rxn_series: pd.Series,
    required_features: List[str],
    pipe: Dict[str, Any],
    eval_cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build the feature matrix for the evaluation run, using *exactly* the same
    featurization settings as the training run stored in pipeline.pkl.

    Key points for reproducibility:
    - Uses pipe['features'] / required_features to decide *exactly* which columns to build.
    - Uses config_snapshot for Mordred agg + NaN policy + 3D toggle.
    - Uses drfp_columns / flags_columns saved in the pickle to match bit/order where available.
    - Applies feature_fillna on the final merged X (mirrors training).
    """
    eval_cfg = eval_cfg or {}
    cfg_snap = pipe.get("config_snapshot", {}) or {}
    hydrogen_patch = bool(cfg_snap.get("hydrogen_radical_special_case_enable", False))

    desc_nan_policy = str(cfg_snap.get("desc_nan_policy", "mean")).lower()
    mordred_agg = str(cfg_snap.get("mordred_agg", "sum")).lower()
    feature_fillna = str(cfg_snap.get("feature_fillna", "mean")).lower()

    # Saved canonical column orders (if present)
    drfp_cols_saved = pipe.get("drfp_columns", None)
    flags_cols_saved = pipe.get("flags_columns", None)

    # Preserve order, drop dups
    req = list(dict.fromkeys(required_features))

    X = pd.DataFrame(index=rxn_series.index)

    # ------------------- Identify feature groups -------------------
    # DRFP columns: prefer the exact saved list if present
    if isinstance(drfp_cols_saved, (list, tuple)) and drfp_cols_saved:
        drfp_set = set(drfp_cols_saved)
        drfp_cols = [c for c in req if c in drfp_set]
    else:
        drfp_cols = [c for c in req if c.startswith("drfp_")]
        drfp_set = set(drfp_cols)

    # Flag columns: prefer the exact saved list if present
    if isinstance(flags_cols_saved, (list, tuple)) and flags_cols_saved:
        flag_set = set(flags_cols_saved)
        flag_cols = [c for c in req if c in flag_set]
    else:
        # Back-compat with XTQSPRKinetic defaults and v2.7.0 hydrogen-radical flags
        known_flags = {
            "is_unimolecular", "has_competing_products",
            "rxn_H_atom_delta", "rxn_H_radical_electron_delta", "rxn_HH_bond_delta",
            "rxn_isolated_H_radical_atom_delta", "rxn_is_2H_radicals_to_H2",
        }
        flag_cols = [c for c in req if c in known_flags]
        flag_set = set(flag_cols)

    # Everything else required by the model is (currently) treated as Mordred-derived
    mordred_cols = [c for c in req if (c not in drfp_set and c not in flag_set)]

    # ------------------- Mordred -------------------
    if mordred_cols:
        # XTQSPRKinetic sets XT_MORDRED_3D based on config['mordred_3d_enable'].
        # Do the same here to avoid environment-dependent mismatches.
        prev_env = os.environ.get("XT_MORDRED_3D", None)
        try:
            if bool(cfg_snap.get("mordred_3d_enable", False)):
                os.environ["XT_MORDRED_3D"] = "1"
            else:
                os.environ.pop("XT_MORDRED_3D", None)

            patch_enable = _bool_or_auto(
                eval_cfg.get("mordred_xtqspr_explicit_h_patch_enable", "auto"),
                default=bool(cfg_snap.get("mordred_xtqspr_explicit_h_patch_enable", False)),
            )

            full_desc = _compute_mordred_agg(
                rxn_series=rxn_series,
                nan_policy=desc_nan_policy,
                agg=mordred_agg,
                parallel_enable=bool(eval_cfg.get(
                    "mordred_parallel_enable",
                    cfg_snap.get("mordred_parallel_enable", False),
                )),
                n_jobs=int(eval_cfg.get(
                    "mordred_parallel_n_jobs",
                    cfg_snap.get("mordred_parallel_n_jobs", 1),
                )),
                backend=str(eval_cfg.get(
                    "mordred_parallel_backend",
                    cfg_snap.get("mordred_parallel_backend", "loky"),
                )),
                batch_size=eval_cfg.get(
                    "mordred_parallel_batch_size",
                    cfg_snap.get("mordred_parallel_batch_size", "auto"),
                ),
                xtqspr_explicit_h_patch_enable=patch_enable,
            )
        finally:
            # Restore previous env to avoid side effects for callers
            if prev_env is None:
                os.environ.pop("XT_MORDRED_3D", None)
            else:
                os.environ["XT_MORDRED_3D"] = prev_env

        present = [c for c in mordred_cols if c in full_desc.columns]
        missing = [c for c in mordred_cols if c not in full_desc.columns]

        desc_part = full_desc[present].copy() if present else pd.DataFrame(index=rxn_series.index)
        for col in missing:
            desc_part[col] = 0.0

        X = pd.concat([X, desc_part[mordred_cols]], axis=1)

    # --------------------------- DRFP bits ---------------------------
    if drfp_cols:
        cfg_snap = pipe.get("config_snapshot", {}) or {}
        # Prefer the same keys XTQSPRKinetic uses: n_bits / radius
        # (Some older snapshots used drfp_n_bits/drfp_radius; support both.)
        if isinstance(drfp_cols_saved, (list, tuple)) and drfp_cols_saved:
            implied_bits = len(drfp_cols_saved)
        else:
            try:
                implied_bits = max(int(c.split("_", 1)[1]) for c in drfp_cols) + 1
            except Exception:
                implied_bits = max(len(drfp_cols), 1)

        n_bits_cfg = int(cfg_snap.get("n_bits", cfg_snap.get("drfp_n_bits", implied_bits)))
        radius_cfg = int(cfg_snap.get("radius", cfg_snap.get("drfp_radius", DEFAULT_CONFIG["drfp_radius"])))

        want_cols = list(drfp_cols_saved) if (isinstance(drfp_cols_saved, (list, tuple)) and drfp_cols_saved) else None

        drfp_df = _compute_drfp(
            rxn_series=rxn_series,
            n_bits=n_bits_cfg,
            radius=radius_cfg,
            want_cols=want_cols,
            hydrogen_patch=hydrogen_patch,
        )

        for col in drfp_cols:
            if col not in drfp_df.columns:
                drfp_df[col] = 0.0

        X = pd.concat([X, drfp_df[drfp_cols]], axis=1)

    # ------------------- Flags ---------------------
    if flag_cols:
        flags_df = _compute_flags(rxn_series, want_cols=flag_cols, hydrogen_patch=hydrogen_patch)
        for col in flag_cols:
            if col not in flags_df.columns:
                flags_df[col] = 0.0
        X = pd.concat([X, flags_df[flag_cols]], axis=1)

    # Match training behavior on the final merged X_full
    X = X.replace([np.inf, -np.inf], np.nan)

    if feature_fillna == "zero":
        X = X.fillna(0.0)
    elif feature_fillna == "median":
        X = X.fillna(X.median(numeric_only=True))
    else:  # 'mean' or unknown
        if feature_fillna not in ("mean", "median", "zero"):
            logging.warning("Unknown feature_fillna='%s'; defaulting to 'mean'.", feature_fillna)
        X = X.fillna(X.mean(numeric_only=True))

    # Ensure all required features exist, in the correct order
    for col in req:
        if col not in X.columns:
            X[col] = 0.0

    return X[req].copy()


def _drfp_matrix_for_topics(smiles: pd.Series, pipe: Dict[str, Any]) -> pd.DataFrame:
    """Build the DRFP matrix in the exact bit order expected by topic assigners."""
    cfg_snap = pipe.get("config_snapshot", {}) or {}
    hydrogen_patch = bool(cfg_snap.get("hydrogen_radical_special_case_enable", False))
    drfp_cols_saved = pipe.get("drfp_columns", None)
    if DrfpEncoder is None:
        raise RuntimeError("DRFP is required for topic assignment but not installed.")

    if drfp_cols_saved is not None:
        n_bits = len(drfp_cols_saved)
        radius = int(cfg_snap.get("radius", 3))
        df = _compute_drfp(smiles, n_bits, radius, want_cols=list(drfp_cols_saved), hydrogen_patch=hydrogen_patch)
        return df[list(drfp_cols_saved)]
    else:
        # fall back to config
        n_bits = int(cfg_snap.get("n_bits", 2048))
        radius = int(cfg_snap.get("radius", 3))
        return _compute_drfp(smiles, n_bits, radius, want_cols=None, hydrogen_patch=hydrogen_patch)


def _lda_topic_proba(assigners: Dict[str, Any], X_drfp: pd.DataFrame) -> pd.DataFrame:
    lda = assigners.get("lda", None)
    if lda is None:
        raise RuntimeError("LDA model not present in pipeline.pkl, required for this path.")
    P = lda.transform(X_drfp.values)  # N x K
    # Match training-time convention in XTQSPRKinetic: topic_0, topic_1, ...
    return pd.DataFrame(
        P,
        index=X_drfp.index,
        columns=[f"topic_{k}" for k in range(P.shape[1])])


def topic_proba_from_pickle(pipe: Dict[str, Any], smiles: pd.Series) -> pd.DataFrame:
    """
    Compute family/topic probability for *all* supported methods.
    Honors:
      - topic_method
      - topic_assigners (dict) with possible keys: lda, fcm, gmm, hdbscan, qda
      - topic_proba_active_columns (ordering/K)
      - stored PCA objects for drfp inputs (e.g., fcm_pca, gmm['pca'], hdbscan['pca'], qda['pca'])
    """
    cfg = pipe.get("config_snapshot", {}) or {}
    method = str(pipe.get("topic_method", "lda")).lower()
    asg = pipe.get("topic_assigners", {}) or {}
    active_cols = pipe.get("topic_proba_active_columns", None)

    X_drfp = _drfp_matrix_for_topics(smiles, pipe)
    base_lda = _lda_topic_proba(asg, X_drfp) if ("lda" in asg) else None

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        if active_cols is None:
            return df
        out = df.copy()
        for c in active_cols:
            if c not in out.columns:
                out[c] = 0.0
        return out[active_cols]

    # FCM / LDA+FCM
    if method in ("fcm", "lda+fcm"):
        f = asg.get("fcm", {})
        centers = f.get("centers", None)
        if centers is None:
            raise RuntimeError("FCM centers missing in pipeline.pkl.")
        if fuzz is None:
            raise RuntimeError("scikit-fuzzy is required to evaluate FCM-based pipelines.")
        # Decide input space from the stored config. If a PCA model exists, it was fitted on DRFP.
        fcm_input = str(cfg.get("fcm_input", "lda")).lower()
        if asg.get("fcm_pca", None) is not None:
            X = asg["fcm_pca"].transform(X_drfp.values)
        elif fcm_input == "drfp":
            X = X_drfp.values
        else:
            X = base_lda.values if base_lda is not None else _lda_topic_proba(asg, X_drfp).values
        cntr = np.asarray(centers)
        _, U, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X.T, cntr, m=float(cfg.get("fcm_m", 1.1)),
            error=float(cfg.get("fcm_error", 1e-6)),
            maxiter=int(cfg.get("fcm_max_iter", 300))
        )
        P = U.T
        if active_cols is not None and len(active_cols) == P.shape[1]:
            cols = list(active_cols)
        else:
            cols = [f"topic_{k}" for k in range(P.shape[1])]
        return _align(pd.DataFrame(P, index=smiles.index, columns=cols))

    # GMM / LDA+GMM
    if method in ("gmm", "lda+gmm"):
        g = asg.get("gmm", None)
        if g is None:
            raise RuntimeError("GMM bundle missing in pipeline.pkl.")
        g_input = g.get("input", "lda")
        if g_input == "lda":
            if base_lda is None:
                base_lda = _lda_topic_proba(asg, X_drfp)
            X = base_lda.values
        else:
            X = X_drfp.values
            if g.get("pca", None) is not None:
                X = g["pca"].transform(X)
        P = g["model"].predict_proba(X)
        if active_cols is not None and len(active_cols) == P.shape[1]:
            cols = list(active_cols)
        else:
            cols = [f"topic_{k}" for k in range(P.shape[1])]
        return _align(pd.DataFrame(P, index=smiles.index, columns=cols))

    # HDBSCAN / LDA+HDBSCAN
    if method in ("hdbscan", "lda+hdbscan"):
        h = asg.get("hdbscan", None)
        if h is None:
            raise RuntimeError("HDBSCAN bundle missing in pipeline.pkl.")
        if hdbscan_prediction is None:
            raise RuntimeError("hdbscan is required to evaluate HDBSCAN-based pipelines.")
        h_input = h.get("input", "lda")
        if h_input == "lda":
            if base_lda is None:
                base_lda = _lda_topic_proba(asg, X_drfp)
            X = base_lda.values
        else:
            X = X_drfp.values
            if h.get("pca", None) is not None:
                X = h["pca"].transform(X)
        P = hdbscan_prediction.membership_vector(h["model"], X)  # N x K
        if active_cols is not None and len(active_cols) == P.shape[1]:
            cols = list(active_cols)
        else:
            cols = [f"topic_{k}" for k in range(P.shape[1])]
        return _align(pd.DataFrame(P, index=smiles.index, columns=cols))

    # LDA+QDA
    if method == "lda+qda":
        q = asg.get("qda", None)
        if q is None:
            raise RuntimeError("QDA bundle missing in pipeline.pkl.")
        q_input = q.get("input", "lda")
        if q_input == "lda":
            if base_lda is None:
                base_lda = _lda_topic_proba(asg, X_drfp)
            X = base_lda.values
        else:
            X = X_drfp.values
            if q.get("pca", None) is not None:
                X = q["pca"].transform(X)
        P = q["model"].predict_proba(X)
        if active_cols is not None and len(active_cols) == P.shape[1]:
            cols = list(active_cols)
        else:
            cols = [f"topic_{k}" for k in range(P.shape[1])]
        return _align(pd.DataFrame(P, index=smiles.index, columns=cols))

    # Plain LDA (default/fallback)
    if base_lda is None:
        base_lda = _lda_topic_proba(asg, X_drfp)
    return _align(base_lda)


# ============================== PREDICTION MODES ==============================

def _features_by_family_from_pickle(pipe: Dict[str, Any]) -> Optional[Dict[Union[int, str], List[str]]]:
    """Return per-family selected features if present, with robust int/string keys."""
    meta = (pipe.get("boruta_meta", {}) or {}).get("per_family", {})
    fbf = meta.get("features_by_family", None)
    if fbf is None:
        return None
    out: Dict[Union[int, str], List[str]] = {}
    for k, v in fbf.items():
        vals = list(v)
        out[k] = vals
        try:
            out[int(k)] = vals
        except Exception:
            pass
        out[str(k)] = vals
    return out

def _get_scaler_for_model(pipe: Dict[str, Any], scalers: Dict[Any, Any], mk: Any):
    cs = pipe.get("config_snapshot", {}) or {}
    use_family_scaler = bool(cs.get("use_family_scaler", True))

    # XT behavior: if use_family_scaler is False -> always global scaler
    if not use_family_scaler:
        return scalers.get("global")

    # else prefer family scaler, fallback to global
    return scalers.get(mk) or scalers.get("global")

def predict_hard(pipe: Dict[str, Any],
                 X: pd.DataFrame,
                 proba_df: pd.DataFrame,
                 target_cols: Sequence[str]) -> np.ndarray:
    """Hard routing: use the highest-probability family model for each sample."""
    models = pipe.get("models", {}) or {}
    scalers = pipe.get("scalers", {}) or {}
    features_global = list(pipe.get("features", []))
    fbf = _features_by_family_from_pickle(pipe)
    n_targets = len(list(target_cols))

    fam_idx = proba_df.values.argmax(axis=1)
    yhat = np.full((len(X), n_targets), np.nan, float)

    g_model = models.get("global")
    g_scaler = scalers.get("global")

    for i, k in enumerate(fam_idx):
        mk = _infer_model_key(models.keys(), int(k))
        model = models.get(mk) if mk is not None else None
        scaler = _get_scaler_for_model(pipe, scalers, mk)

        # fallback exactly like XT: if specialist missing -> use global
        if model is None or scaler is None:
            model = g_model
            scaler = g_scaler
            mk = "global"
        if model is None or scaler is None:
            continue

        feats = _family_features_for_key(fbf, mk, features_global)
        try:
            Xi = X.loc[X.index[i], feats].to_frame().T.values
            yhat[i, :] = _coerce_prediction_vector(model.predict(scaler.transform(Xi))[0], n_targets)
        except Exception as e:
            logging.warning("Hard prediction failed for row %s/family %s: %s", X.index[i], mk, e)
            continue

    return yhat


def predict_soft(pipe: Dict[str, Any],
                 X: pd.DataFrame,
                 proba_df: pd.DataFrame,
                 target_cols: Sequence[str],
                 min_weight: float = 1e-8) -> np.ndarray:
    """Soft routing: blend family-specialist predictions by topic probabilities."""
    models = pipe.get("models", {}) or {}
    scalers = pipe.get("scalers", {}) or {}
    features_global = list(pipe.get("features", []))
    fbf = _features_by_family_from_pickle(pipe)
    n_targets = len(list(target_cols))

    proba_df = _normalize_probability_frame(proba_df)
    P = proba_df.values
    K = P.shape[1]
    yhat = np.full((len(X), n_targets), np.nan, float)

    g_model = models.get("global")
    g_scaler = scalers.get("global")

    for i in range(len(X)):
        weights = P[i, :]

        idx = []
        for k in range(K):
            if weights[k] <= min_weight:
                continue
            mk = _infer_model_key(models.keys(), int(k))
            if mk is None or models.get(mk) is None:
                continue
            idx.append(k)

        if not idx:
            if g_model is None or g_scaler is None:
                continue
            try:
                Xi = X.loc[X.index[i], features_global].to_frame().T.values
                yhat[i, :] = _coerce_prediction_vector(g_model.predict(g_scaler.transform(Xi))[0], n_targets)
            except Exception as e:
                logging.warning("Global fallback prediction failed for row %s: %s", X.index[i], e)
            continue

        w = np.array([weights[k] for k in idx], float)
        w_sum = float(w.sum())
        if w_sum <= 0:
            if g_model is None or g_scaler is None:
                continue
            try:
                Xi = X.loc[X.index[i], features_global].to_frame().T.values
                yhat[i, :] = _coerce_prediction_vector(g_model.predict(g_scaler.transform(Xi))[0], n_targets)
            except Exception as e:
                logging.warning("Global fallback prediction failed for row %s: %s", X.index[i], e)
            continue

        w = w / w_sum
        agg = np.zeros(n_targets, float)
        used_weight = 0.0

        for k, wk in zip(idx, w):
            mk = _infer_model_key(models.keys(), int(k))
            model = models.get(mk) if mk is not None else None
            scaler = _get_scaler_for_model(pipe, scalers, mk)

            if model is None or scaler is None:
                model = g_model
                scaler = g_scaler
                mk = "global"
            if model is None or scaler is None:
                continue

            feats = _family_features_for_key(fbf, mk, features_global)
            try:
                Xi = X.loc[X.index[i], feats].to_frame().T.values
                yk = _coerce_prediction_vector(model.predict(scaler.transform(Xi))[0], n_targets)
                if np.all(~np.isfinite(yk)):
                    continue
                agg += float(wk) * yk
                used_weight += float(wk)
            except Exception as e:
                logging.warning("Soft prediction failed for row %s/family %s: %s", X.index[i], mk, e)
                continue

        if used_weight > 0:
            yhat[i, :] = agg / used_weight

    return yhat


def predict_hybrid(pipe: Dict[str, Any],
                   X: pd.DataFrame,
                   proba_df: pd.DataFrame,
                   target_cols: Sequence[str],
                   metric: str,
                   threshold: float) -> np.ndarray:
    """
    Hybrid switch from routing probabilities only.
      - metric='entropy'       -> hard if normalized entropy <= threshold
      - metric='vote_fraction' -> hard if max probability >= threshold
    """
    proba_df = _normalize_probability_frame(proba_df)
    P = proba_df.values
    if metric == "entropy":
        ent = np.array([_normalized_entropy_row(row) for row in P])
        hard_mask = (ent <= threshold)
    else:
        hard_mask = (P.max(axis=1) >= threshold)

    n_targets = len(list(target_cols))
    idx_all = np.arange(len(X))
    idx_hard = idx_all[hard_mask]
    idx_soft = idx_all[~hard_mask]

    yhat = np.full((len(X), n_targets), np.nan, float)
    if idx_hard.size:
        yhat[idx_hard, :] = predict_hard(pipe, X.iloc[idx_hard], proba_df.iloc[idx_hard], target_cols)
    if idx_soft.size:
        yhat[idx_soft, :] = predict_soft(pipe, X.iloc[idx_soft], proba_df.iloc[idx_soft], target_cols)
    return yhat


# ============================== METRICS / REPORT ==============================

def _stats(y_true: np.ndarray, y_pred: np.ndarray, p_features: Optional[int] = None) -> Dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return {"R2": np.nan, "AdjR2": np.nan, "MAE": np.nan, "RMSE": np.nan, "nRMSE": np.nan, "AARD": np.nan, "RAE": np.nan}
    yt = y_true[m]; yp = y_pred[m]
    n = yt.size
    
    try:
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = math.sqrt(mean_squared_error(yt, yp))
        adj_r2 = _adj_r2(r2, n, p_features)

        nrmse = _nrmse(yt, yp)
        aard = _aard(yt, yp)
        rae = _rae(yt, yp)
    except Exception:
        r2, mae, rmse, adj_r2 = np.nan, np.nan, np.nan, np.nan
        nrmse, aard, rae = np.nan, np.nan, np.nan    

    return {"R2": r2, "AdjR2": adj_r2, "MAE": mae, "RMSE": rmse, "nRMSE": nrmse, "AARD": aard, "RAE": rae}


def score_and_write(input_df: pd.DataFrame,
                    proba_df: pd.DataFrame,
                    preds: np.ndarray,
                    out_dir: str,
                    target_cols: Sequence[str],
                    pipe: Optional[Dict[str, Any]] = None,
                    excel_path: Optional[str] = None) -> None:
    """
    Write CSV artifacts and, optionally, an Excel workbook.

    Outputs are target-general:
      - prediction columns: <target>_pred
      - error columns when targets exist: <target>_abs_err, <target>_abs_rel_err, <target>_smape
      - metrics sheets loop over target_cols
    """
    os.makedirs(out_dir, exist_ok=True)
    target_cols = list(target_cols)
    pred_cols = _prediction_columns(target_cols)

    pred_arr = np.asarray(preds, dtype=float)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(-1, 1)
    if pred_arr.shape[1] != len(target_cols):
        fixed = np.full((pred_arr.shape[0], len(target_cols)), np.nan, float)
        m = min(pred_arr.shape[1], len(target_cols))
        fixed[:, :m] = pred_arr[:, :m]
        pred_arr = fixed

    pred_df = pd.DataFrame(pred_arr, columns=pred_cols, index=input_df.index)
    final = pd.concat([input_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    # Topic probabilities
    proba_df.to_csv(os.path.join(out_dir, "topic_proba_pred.csv"), index=False)

    # Hard family assignments via argmax
    fam_idx = proba_df.values.argmax(axis=1)
    fam_series = pd.Series(fam_idx, name="predicted_family")
    fam_series.to_csv(os.path.join(out_dir, "family_assignments_pred.csv"), index=False)

    has_targets = _has_targets(input_df, target_cols)
    missing_targets = [c for c in target_cols if c not in input_df.columns]

    global_rows: List[Dict[str, Any]] = []
    fam_stats: List[Dict[str, Any]] = []
    global_df = pd.DataFrame()
    fam_df = pd.DataFrame()

    p_features: Optional[int] = None
    if pipe is not None:
        feats = list(pipe.get("features", []))
        if feats:
            p_features = len(feats)

    if has_targets:
        # Always recompute metrics on the provided evaluation dataset.
        y_true_all = input_df[target_cols].apply(pd.to_numeric, errors="coerce").values
        y_pred_all = pred_arr

        for j, name in enumerate(target_cols):
            stats = _stats(y_true_all[:, j], y_pred_all[:, j], p_features=p_features)
            global_rows.append({
                "output": name,
                "R2": stats["R2"],
                "AdjR2": stats["AdjR2"],
                "MAE": stats["MAE"],
                "RMSE": stats["RMSE"],
                "nRMSE": stats["nRMSE"],
                "AARD": stats["AARD"],
                "RAE": stats["RAE"],
            })
        global_df = pd.DataFrame(global_rows)

        final_with_fam = final.copy()
        final_with_fam["predicted_family"] = fam_series.values
        for fam, sub in final_with_fam.groupby("predicted_family"):
            for name in target_cols:
                if name not in sub.columns or f"{name}_pred" not in sub.columns:
                    continue
                stats = _stats(
                    pd.to_numeric(sub[name], errors="coerce").values,
                    pd.to_numeric(sub[f"{name}_pred"], errors="coerce").values,
                    p_features=p_features,
                )
                fam_stats.append({
                    "family": int(fam),
                    "output": name,
                    "R2": stats["R2"],
                    "AdjR2": stats["AdjR2"],
                    "MAE": stats["MAE"],
                    "RMSE": stats["RMSE"],
                    "nRMSE": stats["nRMSE"],
                    "AARD": stats["AARD"],
                    "RAE": stats["RAE"],
                })
        if fam_stats:
            fam_df = pd.DataFrame(fam_stats)

    # Human-readable summary
    summary_path = os.path.join(out_dir, "eval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("XTQSPRKinetic reaction-property evaluation summary\n")
        f.write(f"Targets: {', '.join(target_cols)}\n")
        f.write(f"Rows: {len(input_df)}\n\n")
        if not has_targets:
            f.write("Predictions saved.\n")
            f.write(
                "Metrics were not computed because the input CSV is missing one or more "
                f"target columns: {', '.join(missing_targets)}\n"
            )
        else:
            f.write("GLOBAL METRICS (R2 / AdjR2 / MAE / RMSE / nRMSE / AARD / RAE):\n")
            for row in global_rows:
                f.write(
                    f"  {str(row['output']):>20s}  "
                    f"R2={row['R2']:.6f}  "
                    f"AdjR2={row['AdjR2']:.6f}  "
                    f"MAE={row['MAE']:.6f}  "
                    f"RMSE={row['RMSE']:.6f}  "
                    f"nRMSE={row['nRMSE']:.6f}  "
                    f"AARD={row['AARD']:.6f}  "
                    f"RAE={row['RAE']:.6f}\n"
                )
            if fam_stats:
                f.write("\nPER-FAMILY METRICS:\n")
                for row in fam_stats:
                    f.write(
                        f"  fam={row['family']:>3d} {str(row['output']):>20s}  "
                        f"R2={row['R2']:.6f}  "
                        f"AdjR2={row['AdjR2']:.6f}  "
                        f"MAE={row['MAE']:.6f}  "
                        f"RMSE={row['RMSE']:.6f}  "
                        f"nRMSE={row['nRMSE']:.6f}  "
                        f"AARD={row['AARD']:.6f}  "
                        f"RAE={row['RAE']:.6f}\n"
                    )

    # Excel workbook
    if excel_path:
        excel_df = final.copy()
        if has_targets:
            cfg_snap = (pipe or {}).get("config_snapshot", {}) or {}
            eps = float(cfg_snap.get("relerr_epsilon", 1e-8))
            mode = str(cfg_snap.get("relerr_mode", "smape")).lower()
            if mode not in ("abs_rel", "smape", "both"):
                mode = "smape"
            for name in target_cols:
                yt = pd.to_numeric(excel_df[name], errors="coerce").values
                yp = pd.to_numeric(excel_df[f"{name}_pred"], errors="coerce").values
                excel_df[f"{name}_abs_err"] = np.abs(yp - yt)
                if mode in ("abs_rel", "both"):
                    excel_df[f"{name}_abs_rel_err"] = np.abs(yp - yt) / (np.abs(yt) + eps)
                if mode in ("smape", "both"):
                    excel_df[f"{name}_smape"] = (2.0 * np.abs(yp - yt)) / (np.abs(yp) + np.abs(yt) + eps)

        try:
            with pd.ExcelWriter(excel_path) as writer:
                excel_df.to_excel(writer, sheet_name="Predictions", index=False)
                if has_targets and not global_df.empty:
                    global_df.to_excel(writer, sheet_name="Global_Metrics", index=False)
                if has_targets and not fam_df.empty:
                    fam_df.to_excel(writer, sheet_name="Family_Metrics", index=False)
            logging.info("Excel performance workbook written to: %s", excel_path)
        except Exception as e:
            logging.warning("Could not write Excel performance workbook '%s': %s", excel_path, e)


# =================================== MAIN =====================================

def main():
    cfg = parse_args()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    setup_logging(os.path.join(cfg["output_dir"], cfg["log_file"]))
    
    if log_yaml_duplicate_key_summary is not None:
        log_yaml_duplicate_key_summary(cfg)

    logging.info("Loading pipeline: %s", cfg["pipeline_pkl"])
    pipe: Dict[str, Any] = joblib.load(cfg["pipeline_pkl"])

    # Extract essential bits
    target_cols = _target_cols_from_pipe(pipe, cfg)
    logging.info("Evaluation targets: %s", target_cols)

    models = pipe.get("models", {}) or {}
    if not models:
        raise RuntimeError("pipeline.pkl has no 'models' block.")
    scalers = pipe.get("scalers", {}) or {}
    features_global = list(pipe.get("features", []))
    features_by_family = _features_by_family_from_pickle(pipe)

    # Input
    logging.info("Reading input CSV: %s", cfg["input_csv"])
    df = pd.read_csv(cfg["input_csv"])
    if "reaction_smiles" not in df.columns:
        raise ValueError("Input CSV must contain a 'reaction_smiles' column.")
    smiles = df["reaction_smiles"]

    # Build features required by the models (union of global + per-family)
    required = _union_features(features_global, features_by_family)
    logging.info("Computing required features (%d columns)…", len(required))
    X_all = compute_required_features(smiles, required, pipe, eval_cfg=cfg)

    # Topic probabilities for routing
    logging.info("Computing topic/family probabilities using method=%s", pipe.get("topic_method", "lda"))
    proba_df = topic_proba_from_pickle(pipe, smiles)
    # If the pickle specified an active column order, enforce it
    active_cols = pipe.get("topic_proba_active_columns", None)
    if active_cols is not None:
        for c in active_cols:
            if c not in proba_df.columns:
                proba_df[c] = 0.0
        proba_df = proba_df[active_cols]
    if bool(cfg.get("normalize_routing_probabilities", True)):
        proba_df = _normalize_probability_frame(proba_df)

    # Choose routing mode.
    # The final XTQSPRKinetic training pipeline may store routing_mode='hybrid' only
    # after Bootstrap quality gates passed. For new external reactions, however, the
    # per-sample Bootstrap stability table used during training is not available.
    # Therefore auto mode falls back to soft routing unless the user explicitly allows
    # an approximate hybrid rule based on current routing probabilities.
    routing_requested = cfg["routing_mode"]
    routing = routing_requested
    if routing == "auto":
        routing = str(pipe.get("routing_mode", "soft")).lower()
        if routing not in ("hard", "soft", "hybrid"):
            routing = "soft"

    if _pipeline_soft_used_bootstrap_probabilities(pipe):
        logging.warning(
            "The training pipeline used bootstrap-aggregated probabilities for soft routing, "
            "but bootstrap ensembles are not stored for external reactions. Evaluation will "
            "use probabilities reconstructed from the stored topic assigners."
        )

    if routing == "hybrid" and _pipeline_bootstrap_hybrid_requires_training_stability(pipe):
        allow_approx = bool(cfg.get("allow_hybrid_approximation", False))
        if routing_requested == "auto" and not allow_approx:
            logging.warning(
                "Stored pipeline routing_mode is HYBRID, but Bootstrap stability diagnostics "
                "are training-row-specific and cannot be reconstructed for new evaluation rows. "
                "Falling back to SOFT routing. Set allow_hybrid_approximation=true or "
                "--routing_mode hybrid to force approximate hybrid evaluation."
            )
            routing = "soft"
        else:
            logging.warning(
                "Using approximate HYBRID evaluation based on current routing probabilities "
                "(%s threshold %.6g). This is not identical to training-time Bootstrap-stability "
                "hybrid routing for new rows.",
                str(pipe.get("config_snapshot", {}).get("hybrid_stability_metric", cfg["hybrid_metric"])).lower(),
                float(pipe.get("config_snapshot", {}).get("hybrid_stability_threshold", cfg["hybrid_threshold"])),
            )

    logging.info("Routing mode: %s", routing.upper())

    if routing == "hard":
        preds = predict_hard(pipe, X_all, proba_df, target_cols)
    elif routing == "hybrid":
        # Use config_snapshot’s hybrid knobs when available
        cs = pipe.get("config_snapshot", {}) or {}
        metric = str(cs.get("hybrid_stability_metric", cfg["hybrid_metric"])).lower()
        thr = float(cs.get("hybrid_stability_threshold", cfg["hybrid_threshold"]))
        preds = predict_hybrid(pipe, X_all, proba_df, target_cols, metric, thr)
    else:
        preds = predict_soft(pipe, X_all, proba_df, target_cols)

    # Build Excel output path INSIDE the output_dir
    base_name = os.path.splitext(os.path.basename(cfg["input_csv"]))[0]
    excel_path = os.path.join(cfg["output_dir"], f"{base_name}_property_performance.xlsx")

    # Output & metrics (CSV + Excel)
    score_and_write(df, proba_df, preds, cfg["output_dir"],
                    target_cols=target_cols, pipe=pipe, excel_path=excel_path)
    logging.info("Done. Artifacts written under: %s", cfg["output_dir"])
    logging.info("Excel summary written at: %s", excel_path)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        sys.exit(1)
