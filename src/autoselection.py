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

from .shared_imports import *
from .config import *
from .utils import _ensure_dir, timeit_logger
from .featurization import *
from .pipeline import run_pipeline


# ------------------------- AUTO-CHOOSER WRAPPER ------------------------------

def _infer_input_token(cfg: Dict[str, Any], method: str) -> str:
    """
    Decide which *input space* fed the topic/clustering step.
    Priority: explicit config -> inference from method name.
    Recognizes: topic_input, cluster_input, method_input, topic_space, gmm_input, fcm_input, hdbscan_input, qda_input.
    """
    for k in ('topic_input','cluster_input','method_input','topic_space',
            'gmm_input','fcm_input','hdbscan_input','qda_input'):
        v = cfg.get(k, None)
        if isinstance(v, str) and v.strip():
            return f"in{_safe_token(v.strip().lower())}"
    return "inLDA" if ('lda' in method) else "inDRFP"

def _safe_token(x: Any) -> str:
    s = str(x)
    return "".join(ch for ch in s if ch.isalnum() or ch in ['_', '.', '+'])

def _flag(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(cfg.get(key, default))

def _topic_signature(cfg: Dict[str, Any]) -> str:
    """
    Build a compact, informative suffix describing the candidate:
    method, input space, PCA, LDA/FCM/GMM/HDBSCAN key params, bootstrap flags, routing, replacement.
    """
    toks: List[str] = []

    # --- Method core ---
    method = str(cfg.get('topic_method', 'lda')).lower().replace('+', '_')
    toks.append(method)

    # --- Explicit method-input token (DRFP / LDA / custom) ---
    toks.append(_infer_input_token(cfg, method))

    # --- PCA on DRFP (support both generic and GMM-specific flags) ---
    pca_enabled = bool(cfg.get('pca_enable', False)) 

    if pca_enabled:
        # Prefer variance-based token if provided; otherwise component count
        var = cfg.get('pca_var', 0.95)
        n   = cfg.get('pca_n_components', None)
        if var is not None:
            try:
                toks.append(f"pcaVar{int(round(float(var)*100))}")
            except Exception:
                toks.append(f"pcaVar{_safe_token(var)}")
        elif n is not None:
            toks.append(f"pca{_safe_token(n)}")
        else:
            toks.append("pca")

    # --- LDA knobs (when present) ---
    if 'lda' in method:
        k = cfg.get('n_topics', cfg.get('lda_n_topics', ''))
        toks.append(f"k{k}")
        lm = cfg.get('lda_learning_method', None)  # 'batch'|'online'
        if isinstance(lm, str) and lm:
            toks.append(f"lda{lm[:1]}")  # ldab / ldao
        alpha = cfg.get('lda_doc_topic_prior', None)
        if alpha not in (None, 'auto'):
            try: toks.append(f"a{float(alpha):g}")
            except: toks.append(f"a{alpha}")
        beta = cfg.get('lda_topic_word_prior', None)
        if beta not in (None, 'auto'):
            try: toks.append(f"b{float(beta):g}")
            except: toks.append(f"b{beta}")
        lit = cfg.get('lda_max_iter', None)
        if isinstance(lit, (int, float)) and lit:
            toks.append(f"lit{int(lit)}")

    # --- FCM knobs (when present) ---
    if 'fcm' in method:
        kf = cfg.get('fcm_k', cfg.get('n_topics', ''))
        toks.append(f"fcmk{kf}")
        m = cfg.get('fcm_m', None)
        if m is not None:
            try: toks.append(f"m{float(m):.2f}")
            except: toks.append(f"m{m}")

    # --- GMM knobs (when present) ---
    if 'gmm' in method:
        nc = cfg.get('gmm_n_components', cfg.get('n_topics', ''))
        toks.append(f"gmm{nc}")
        cov = str(cfg.get('gmm_covariance_type', 'full')).lower()  # full|tied|diag|spherical
        cov_map = {'full': 'f', 'tied': 't', 'diag': 'd', 'spherical': 's'}
        toks.append(f"cov{cov_map.get(cov, cov[:1])}")
        initp = str(cfg.get('gmm_init_params', 'kmeans')).lower()  # kmeans|kmeans++|random
        if '++' in initp:
            toks.append("initkpp")
        elif 'kmeans' in initp:
            toks.append("initkm")
        elif 'rand' in initp:
            toks.append("initrand")
        else:
            toks.append(f"init{initp}")
        nI = cfg.get('gmm_n_init', 1)
        try:
            nI = int(nI)
            if nI != 1:
                toks.append(f"nI{nI}")
        except Exception:
            pass
        reg = cfg.get('gmm_reg_covar', None)
        try:
            if reg is not None and abs(float(reg) - 1e-6) > 1e-9:
                toks.append(f"reg{float(reg):g}")
        except Exception:
            pass
        tol = cfg.get('gmm_tol', None)
        try:
            if tol is not None and abs(float(tol) - 1e-3) > 1e-12:
                toks.append(f"tol{float(tol):g}")
        except Exception:
            pass
        mit = cfg.get('gmm_max_iter', None)
        try:
            if mit is not None and int(mit) != 100:
                toks.append(f"mit{int(mit)}")
        except Exception:
            pass

    # --- HDBSCAN knobs (when present) ---
    if 'hdbscan' in method:
        mcs = cfg.get('hdbscan_min_cluster_size', cfg.get('min_cluster_size', None))
        if mcs is not None:
            toks.append(f"minSz{int(mcs)}")
        ms = cfg.get('hdbscan_min_samples', cfg.get('min_samples', None))
        if ms not in (None, 'None'):
            try: toks.append(f"ms{int(ms)}")
            except: pass
        eps = cfg.get('hdbscan_cluster_selection_epsilon', cfg.get('cluster_selection_epsilon', 0.0))
        try:
            epsv = float(eps)
            toks.append(f"eps{epsv:g}" if epsv != 0.0 else "eps0.00")
        except Exception:
            pass

    # --- Bootstraps 
    B  = cfg.get('bootstrap_B', 25)  
    sf = cfg.get('bootstrap_sample_frac', 0.8)  
    sf_pct = int(round(100*float(sf)))  

    if cfg.get('bootstrap_lda_enable', False):  
        toks.append(f"bsL{int(B)}x{sf_pct}")  
        if cfg.get('bootstrap_lda_topic_alignment', False):  
            toks.append("align")  

    if cfg.get('bootstrap_fcm_enable', False):  
        toks.append(f"bsF{int(B)}x{sf_pct}")  
        if cfg.get('bootstrap_fcm_align_centers', True):  
            toks.append("align")  

    if cfg.get('bootstrap_gmm_enable', False):  
        toks.append(f"bsG{int(B)}x{sf_pct}")

    if cfg.get('bootstrap_hdbscan_enable', False):  
        toks.append(f"bsG{int(B)}x{sf_pct}")

    if cfg.get('bootstrap_qda_enable', False):  
        toks.append(f"bsG{int(B)}x{sf_pct}")

    # --- Routing (prioritize hybrid if enabled) ---
    if bool(cfg.get('use_hybrid_routing', False)):
        thr = cfg.get('hybrid_stability_threshold', None)
        if thr is not None:
            try: toks.append(f"hybT{float(thr):.2f}")
            except: toks.append("hybrid")
        else:
            toks.append("hybrid")
    else:
        if bool(cfg.get('use_soft_routing', False)):
            toks.append("soft")
        else:
            toks.append("hard")

    # --- Family replacement via bootstrap stability ---
    br = bool(
        cfg.get('bootstrap_replace_family_labels', False)
        or cfg.get('bootstrap_replacement_enable', False)
        or cfg.get('bootstrap_replace_family', False)
    )
    
    if br:
        toks.append("replF")
        if bool(cfg.get('bootstrap_retrain_on_replaced', False)):
            toks.append("retrain")

    # --- HDBSCAN family derivation (if applicable) ---
    if 'hdbscan' in method:
        fam_from = str(cfg.get('hdbscan_family_from', 'proba')).lower()
        if fam_from != 'proba': toks.append("labelfam")
        if bool(cfg.get('hdbscan_noise_to_family', True)): toks.append("nz0")

    toks = [_safe_token(t) for t in toks if str(t).strip()]
    return "_".join(toks)


def _candidate_outdir(base_dir: str, idx: int, cfg: Dict[str, Any]) -> str:
    """
    Build the output directory for one Auto-Chooser candidate run.

    Folder naming:
        <candidate_name>__auto_cand_<idx>__<topic_signature>

    Behavior:
        - Uses the YAML candidate name when available.
        - Keeps the candidate index for ordering and uniqueness.
        - Adds the topic-method signature for reproducibility.
        - Avoids repeating the parent output-directory name inside the candidate
          folder name.
        - Truncates excessively long folder names and appends a short hash to
          preserve practical uniqueness.

    This affects output organization only. It does not change candidate scoring,
    family discovery, routing, training, or prediction behavior.
    """
    sig = _topic_signature(cfg)

    # Human-readable candidate name from YAML, e.g.
    cand_name = _safe_token(cfg.get('name', f'candidate_{idx:02d}'))

    # Parent folder = run identity
    # Candidate folder = candidate identity + index + method signature
    folder_name = f"auto_cand_{idx:02d}_{cand_name}_{sig}"

    # Avoid excessively long path names while preserving uniqueness.
    if len(folder_name) > 180:
        digest = hashlib.sha1(folder_name.encode("utf-8")).hexdigest()[:8]
        folder_name = folder_name[:171].rstrip("_")
        folder_name = f"{folder_name}_{digest}"

    return os.path.join(base_dir, folder_name)

def _compute_eval_score(candidate_cfg: Dict[str, Any], metric: str = 'r2_mean') -> Optional[float]:
    """
    Compute a scalar score for a candidate from its quick run outputs.

    Scoring convention:
        - Higher score is always better.
        - Goodness-of-fit metrics such as R2 or adjusted R2 are maximized directly.
        - Error metrics such as RMSE, MAE, nRMSE, AARD, and RAE are negated
          internally so that lower error corresponds to a higher score.

    This function controls candidate ranking only. It does not change model
    training, prediction equations, or the final full-strength run.

    Primary location:
        <output_dir>/metrics/global_stats.csv
    Fallbacks (searched in order):
        <output_dir>/metrics/global_metrics.csv
        <output_dir>/metrics/global_stats.json
        <output_dir>/global_stats.csv
        <output_dir>/global_metrics.csv
        <output_dir>/global_stats.json

    Scoring rule (maximize):
        - 'r2_mean'    : mean of all columns containing 'r2' (case-insensitive)
        - 'adjr2_mean' : mean of all columns containing 'adjr2'
        - 'nrmse_mean' : -mean of all columns containing 'nrmse'
        - 'aard_mean'  : -mean of all columns containing 'aard'
        - 'rae_mean'   : -mean of all columns containing 'rae'
    """
    out_dir = candidate_cfg.get('output_dir', 'network_output')
    metrics_dir = os.path.join(out_dir, 'metrics')

    # Primary paths under metrics/
    primary_csv = os.path.join(metrics_dir, 'global_stats.csv')
    alt_csv     = os.path.join(metrics_dir, 'global_metrics.csv')
    alt_json    = os.path.join(metrics_dir, 'global_stats.json')

    # Fallback paths at candidate root (mirrors)
    root_primary_csv = os.path.join(out_dir, 'global_stats.csv')
    root_alt_csv     = os.path.join(out_dir, 'global_metrics.csv')
    root_alt_json    = os.path.join(out_dir, 'global_stats.json')

    def _load_df() -> Optional[pd.DataFrame]:
        # 1) Try metrics/ (preferred)
        for p in (primary_csv, alt_csv):
            if os.path.isfile(p):
                try:
                    return pd.read_csv(p)
                except Exception:
                    pass
        if os.path.isfile(alt_json):
            try:
                with open(alt_json, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    return pd.DataFrame([obj])
                if isinstance(obj, list):
                    return pd.DataFrame(obj)
            except Exception:
                pass

        # 2) Try candidate root (fallbacks)
        for p in (root_primary_csv, root_alt_csv):
            if os.path.isfile(p):
                try:
                    return pd.read_csv(p)
                except Exception:
                    pass
        if os.path.isfile(root_alt_json):
            try:
                with open(root_alt_json, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    return pd.DataFrame([obj])
                if isinstance(obj, list):
                    return pd.DataFrame(obj)
            except Exception:
                pass

        # Keep original warning wording (points to .../metrics)
        logging.warning("Auto-chooser: metrics file not found for candidate at %s", metrics_dir)
        return None

    df = _load_df()
    if df is None or df.empty:
        return None

    m = str(metric).strip().lower()
    cols_lower = [c.lower() for c in df.columns]

    def _mean_of_cols(substr: str) -> Optional[float]:
        idx = [i for i, c in enumerate(cols_lower) if substr in c]
        if not idx:
            return None
        vals = []
        for i in idx:
            try:
                col_vals = df.iloc[:, i].values
                col_vals_float = [float(v) for v in col_vals if pd.notnull(v)]
                vals.extend(col_vals_float)
            except Exception:
                continue
        
        if not vals:
            return None
        
        return float(np.nanmean(vals))

    if m == 'r2_mean':
        val = _mean_of_cols('r2')
        return val if val is not None else None
    
    if m == 'adjr2_mean':
        val = _mean_of_cols('adjr2')
        return val if val is not None else None
    
    if m == 'nrmse_mean':
        val = _mean_of_cols('nrmse')
        return (-val) if val is not None else None
    
    if m == 'aard_mean':
        val = _mean_of_cols('aard')
        return (-val) if val is not None else None
    
    if m == 'rae_mean':
        val = _mean_of_cols('rae')
        return (-val) if val is not None else None

    # Unknown metric name: default to r2_mean
    logging.warning("Auto-chooser: unknown metric '%s'; defaulting to r2_mean.", metric)
    val = _mean_of_cols('r2')
    return val if val is not None else None


def _make_fast_eval_cfg(base_cfg: Dict[str, Any], scheme: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Build a fast-evaluation configuration for a given candidate scheme.

    Design:
    - Start from base_cfg, overlay the candidate 'scheme', and apply light/fast overrides:
        * Smaller n_estimators, lower Boruta iterations
        * Disable SHAP, CV, heavy cache writes, sidecars
        * Force a unique output_dir per candidate to avoid cross-talk
    - Do NOT mutate base_cfg in place (always copy).

    Parameters
    ----------
    base_cfg : dict
        The current merged configuration.
    scheme : dict
        Candidate dictionary containing topic_method and any method-specific keys.
    idx : int
        1-based candidate index for unique output_dir suffixing.

    Returns
    -------
    cfg : dict
        A new configuration dictionary safe to pass into run_pipeline for quick scoring.
    """
    cfg = copy.deepcopy(base_cfg)

    # 1) Overlay candidate keys.
    for k, v in (scheme or {}).items():
        cfg[k] = v
    
    # Mark this as an Auto-Chooser fast candidate run.
    # This prevents duplicate YAML-key warnings from being replayed inside every
    # candidate pipeline.log.
    cfg['_auto_chooser_candidate_run'] = True

    # 2) Unique output dir for the candidate (no collisions).
    base_out = os.path.abspath(cfg.get('output_dir', 'network_output'))
    cand_dir = _candidate_outdir(base_out, idx, cfg)
    cfg['output_dir'] = cand_dir

    # 3) Fast knobs.
    # Official v2.7.0 key is auto_eval_fast_n_estimators.
    # auto_eval_n_estimators is kept only as a backward-compatible fallback.
    fast_n_estimators = int(
        cfg.get('auto_eval_fast_n_estimators', cfg.get('auto_eval_n_estimators', 200))
    )
    fast_boruta_iter = int(cfg.get('auto_eval_boruta_max_iter', 50))

    # XGB params: only reduce n_estimators; keep all other user parameters.
    xgbp = dict(cfg.get('xgb_params', {}))
    if 'n_estimators' in xgbp:
        xgbp['n_estimators'] = min(int(xgbp['n_estimators']), fast_n_estimators)
    else:
        xgbp['n_estimators'] = fast_n_estimators
    cfg['xgb_params'] = xgbp

    # Boruta speed-up.
    cfg['boruta_max_iter'] = min(int(cfg.get('boruta_max_iter', 100)), fast_boruta_iter)

    # Disable heavy post-hoc analysis for quick candidate scoring.
    cfg['shap_sample_size'] = 0
    cfg['descriptive_figures_enable'] = False
    cfg['pipeline_dump_sidecar_meta'] = False
    cfg['pipeline_compress'] = 'auto'

    # Avoid large cache writes during quick evaluation.
    cfg['boruta_cache'] = False

    _ensure_dir(cand_dir)
    return cfg

def _materialize_candidate_defaults(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a compact list of candidate family schemes for the auto-chooser.
    Each dict is a *minimal* override that gets merged over the current config.
    Skips candidates that require unavailable optional deps (FCM/HDBSCAN).
    """
    K = int(config.get('n_topics', 6))

    candidates: List[Dict[str, Any]] = []

    # ---- LDA baseline (soft) ----
    candidates.append({
        'topic_method': 'lda',
        'n_topics': K,
        'use_soft_routing': True,
        'use_hybrid_routing': False,
        'bootstrap_lda_enable': False,
    })
    # ---- LDA (hybrid via stability from a quick bootstrap) ----
    candidates.append({
        'topic_method': 'lda',
        'n_topics': K,
        'bootstrap_lda_enable': True,
        'bootstrap_B': 20,
        'bootstrap_sample_frac': 0.8,
        'use_hybrid_routing': True,
        'hybrid_stability_metric': 'entropy',
        'hybrid_stability_threshold': 0.35,
    })

    # ---- LDA+FCM (soft memberships from FCM over LDA topic space) ----
    if HAVE_FCM:
        candidates.append({
            'topic_method': 'lda+fcm',
            'fcm_k': K,
            'fcm_input': 'lda',
            'fcm_m': 1.2,
            'use_soft_routing': True,
            'bootstrap_fcm_enable': False,
        })
        # Hybrid with bootstrap stability (uses FCM bootstrap)
        candidates.append({
            'topic_method': 'lda+fcm',
            'fcm_k': K,
            'fcm_input': 'lda',
            'fcm_m': 1.2,
            'use_soft_routing': True,
            'bootstrap_fcm_enable': True,
            'bootstrap_B': 20,
            'bootstrap_sample_frac': 0.8,
            'use_hybrid_routing': True,
            'hybrid_stability_metric': 'vote_fraction',
            'hybrid_stability_threshold': 0.75,
        })

        if bool(config.get('auto_eval_include_fcm_drfp', True)):
            # FCM directly on DRFP (soft)
            candidates.append({
                'topic_method': 'fcm',
                'fcm_input': 'drfp',
                'fcm_k': K,
                'fcm_m': 1.2,
                'pca_enable': False,
                'use_soft_routing': True,
            })

        if bool(config.get('auto_eval_include_fcm_pca', True)):
            # FCM on DRFP with PCA (soft)
            candidates.append({
                'topic_method': 'fcm',
                'fcm_input': 'drfp',
                'fcm_k': K,
                'fcm_m': 1.2,
                'pca_enable': True,
                'pca_var': 0.95,
                'use_soft_routing': True,
            })

    # ---- LDA+GMM (probabilistic clustering on topic space) ----
    candidates.append({
        'topic_method': 'lda+gmm',
        'gmm_input': 'lda',
        'gmm_n_components': K,
        'gmm_covariance_type': 'full',
        'use_soft_routing': True,
    })
    # A second GMM flavor for diversity
    candidates.append({
        'topic_method': 'lda+gmm',
        'gmm_input': 'lda',
        'gmm_n_components': K,
        'gmm_covariance_type': 'diag',
        'use_soft_routing': True,
    })

    # ---- GMM on DRFP (+ PCA) ----
    candidates.append({
        'topic_method': 'gmm',
        'gmm_input': 'drfp',
        'pca_enable': True,
        'pca_var': 0.95,
        'gmm_n_components': K,
        'gmm_covariance_type': 'full',
        'use_soft_routing': True,
    })

    # ---- LDA+GMM (bootstrap + hybrid) ----
    candidates.append({
        'topic_method': 'lda+gmm',
        'gmm_input': 'lda',
        'gmm_n_components': K,
        'gmm_covariance_type': 'full',
        'use_soft_routing': True,
        'bootstrap_gmm_enable': True,
        # reuse bootstrap knobs (same as bootstrap-LDA/FCM)
        'bootstrap_B': int(config.get('bootstrap_B', 20)),
        'bootstrap_sample_frac': float(config.get('bootstrap_sample_frac', 0.8)),
        'use_hybrid_routing': True,
        'hybrid_stability_metric': str(config.get('hybrid_stability_metric', 'vote_fraction')),
        'hybrid_stability_threshold': float(config.get('hybrid_stability_threshold', 0.75)),
    })

    # ---- GMM on DRFP (+ PCA) with bootstrap + hybrid ----
    candidates.append({
        'topic_method': 'gmm',
        'gmm_input': 'drfp',
        'pca_enable': True,
        'pca_var': 0.95,
        'gmm_n_components': K,
        'gmm_covariance_type': 'full',
        'use_soft_routing': True,
        'bootstrap_gmm_enable': True,
        'bootstrap_B': int(config.get('bootstrap_B', 20)),
        'bootstrap_sample_frac': float(config.get('bootstrap_sample_frac', 0.8)),
        'use_hybrid_routing': True,
        'hybrid_stability_metric': str(config.get('hybrid_stability_metric', 'vote_fraction')),
        'hybrid_stability_threshold': float(config.get('hybrid_stability_threshold', 0.75)),
    })

    # ---- LDA+HDBSCAN / HDBSCAN ----
    if HAVE_HDBSCAN:
        # HDBSCAN over LDA-topic space
        candidates.append({
            'topic_method': 'lda+hdbscan',
            'hdbscan_input': 'lda',
            'hdbscan_min_cluster_size': max(8, K),  # sensible default
            'hdbscan_min_samples': None,
            'hdbscan_prediction_data': True,
            'use_soft_routing': True,
        })
        # HDBSCAN over DRFP (+ PCA)
        candidates.append({
            'topic_method': 'hdbscan',
            'hdbscan_input': 'drfp',
            'pca_enable': True,
            'pca_var': 0.95,
            'hdbscan_min_cluster_size': max(8, K),
            'hdbscan_min_samples': None,
            'hdbscan_prediction_data': True,
            'use_soft_routing': True,
        })

        # LDA+HDBSCAN with bootstrap + hybrid
        candidates.append({
            'topic_method': 'lda+hdbscan',
            'hdbscan_input': 'lda',
            'hdbscan_min_cluster_size': max(8, K),
            'hdbscan_min_samples': None,
            'hdbscan_prediction_data': True,
            'use_soft_routing': True,
            'bootstrap_hdbscan_enable': True,
            'bootstrap_lda_enable': False,
            'bootstrap_B': int(config.get('bootstrap_B', 20)),
            'bootstrap_sample_frac': float(config.get('bootstrap_sample_frac', 0.8)),
            'use_hybrid_routing': True,
            'hybrid_stability_metric': str(config.get('hybrid_stability_metric', 'vote_fraction')),
            'hybrid_stability_threshold': float(config.get('hybrid_stability_threshold', 0.75)),
        })

        #HDBSCAN over DRFP (+ PCA) with bootstrap + hybrid
        candidates.append({
            'topic_method': 'hdbscan',
            'hdbscan_input': 'drfp',
            'pca_enable': True,
            'pca_var': 0.95,
            'hdbscan_min_cluster_size': max(8, K),
            'hdbscan_min_samples': None,
            'hdbscan_prediction_data': True,
            'use_soft_routing': True,
            'bootstrap_lda_enable': False,
            'bootstrap_hdbscan_enable': True,
            'bootstrap_B': int(config.get('bootstrap_B', 20)),
            'bootstrap_sample_frac': float(config.get('bootstrap_sample_frac', 0.8)),
            'use_hybrid_routing': True,
            'hybrid_stability_metric': str(config.get('hybrid_stability_metric', 'vote_fraction')),
            'hybrid_stability_threshold': float(config.get('hybrid_stability_threshold', 0.75)),
        })

    # Users can supply their own overrides in config['candidate_schemes'].
    user = config.get('candidate_schemes') or []
    if isinstance(user, list):
        for u in user:
            try:
                if isinstance(u, dict):
                    candidates.append(copy.deepcopy(u))
            except Exception:
                pass

    # Safety: filter out candidates that require libs not installed
    filtered: List[Dict[str, Any]] = []
    for c in candidates:
        tm = str(c.get('topic_method', 'lda')).lower()
        if 'fcm' in tm and not HAVE_FCM:
            continue
        if 'hdbscan' in tm and not HAVE_HDBSCAN:
            continue
        filtered.append(c)

    # Optional clamp to speed auto: auto_eval_max_candidates
    maxk = config.get('auto_eval_max_candidates', None)
    if isinstance(maxk, int) and maxk > 0:
        filtered = filtered[:maxk]

    return filtered

@timeit_logger("Auto choose family scheme")
def auto_choose_family_scheme(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-chooser for the 'best' topic/family scheme.

    Strategy
    --------
    1) Build a set of candidate schemes (either from config['candidate_schemes']
       or via _materialize_candidate_defaults(config)).
    2) For each candidate, capped by auto_eval_max_candidates:
       - Create a "fast" config with _make_fast_eval_cfg(...) that reduces runtime.
       - Run a quick pipeline using run_pipeline(cand_cfg).
       - Read the candidate metrics and reduce them to a single scalar score with
         _compute_eval_score(...), so candidates can be ranked.
    3) Pick the best-scoring candidate and build a full-strength config:
       - Overlay only the winning candidate's topic/routing keys onto the original config.
       - Restore full-strength modeling/reporting knobs from the original config.
       - Restore the original output_dir.
    4) Return the final config. The main caller then runs run_pipeline(final_cfg).

    Notes
    -----
    * Scoring metric is maximized.
    * Supported metrics depend on _compute_eval_score(...), typically including:
      {'r2_mean', 'adjr2_mean', 'rmse_mean', 'mae_mean',
       'nrmse_mean', 'aard_mean', 'rae_mean'}.
    * Error metrics are negated internally by _compute_eval_score(...) when needed,
      so higher score remains better.
    * Candidate quick-runs are isolated in unique output subfolders.
    * Candidate names from candidate_schemes are preserved in the leaderboard,
      selected-candidate metadata, and candidate output-folder names.
    * Auto-Chooser stores runtime provenance in final_cfg, including selected
      candidate name, score, signature, output directory, attempted count, and
      valid-scored count. These are generated metadata keys and do not need to
      be present in DEFAULT_CONFIG.
    * auto_eval_max_candidates=None means evaluate all materialized candidates.
    * If no candidate yields a valid score, the original config is returned unchanged.
    """

    _t0_auto = time.time()

    base_cfg = copy.deepcopy(config)
    base_out = os.path.abspath(base_cfg.get('output_dir', 'network_output'))
    _ensure_dir(base_out)

    metric = str(base_cfg.get('auto_eval_metric', 'r2_mean')).lower()

    # 1) Materialize candidate list
    user_cands = base_cfg.get('candidate_schemes', None)
    if isinstance(user_cands, list) and len(user_cands) > 0:
        candidates: List[Dict[str, Any]] = user_cands
        logging.info("[auto] Using %d user-specified candidate scheme(s).", len(candidates))
    else:
        candidates = _materialize_candidate_defaults(base_cfg)
        logging.info("[auto] Using %d default candidate scheme(s).", len(candidates))

    # Resolve candidate cap after candidates are known.
    max_cand_raw = base_cfg.get('auto_eval_max_candidates', None)

    if max_cand_raw is None:
        max_cand = len(candidates)
    else:
        max_cand = int(max_cand_raw)
        max_cand = max(1, min(max_cand, len(candidates)))

    # 2) Evaluate candidates quickly
    leaderboard: List[Dict[str, Any]] = []

    best_score: Optional[float] = None
    best_cfg: Optional[Dict[str, Any]] = None
    best_scheme: Optional[Dict[str, Any]] = None
    best_row: Optional[Dict[str, Any]] = None

    attempted = 0

    for i, scheme in enumerate(candidates, start=1):
        if attempted >= max_cand:
            logging.info("[auto] Reached evaluation cap (%d); stopping candidate loop.", max_cand)
            break

        attempted += 1

        # Build a fast evaluation config for this candidate.
        # _make_fast_eval_cfg(...) should assign a unique output_dir.
        cand_cfg = _make_fast_eval_cfg(base_cfg, scheme, i)

        _t0 = time.time()

        logging.info("[auto] Evaluating candidate %d/%d: %s", i, min(max_cand, len(candidates)), scheme)
        try:
            run_pipeline(cand_cfg)
        except AutoChooserSkipCandidate as e:
            logging.info("[auto] Candidate %d skipped: %s", i, e)
            continue
        except Exception as e:
            logging.exception("[auto] Candidate %d failed: %s", i, e)
            continue

        _dt = time.time() - _t0
        logging.info("[auto] Candidate %d finished in %.2fs", i, _dt)

        # Mirror global_stats.csv into metrics/global_stats.csv if needed.
        # This preserves compatibility with _compute_eval_score(...) if it expects metrics/.
        try:
            cand_dir = cand_cfg['output_dir']
            src = os.path.join(cand_dir, "global_stats.csv")
            dst_dir = os.path.join(cand_dir, "metrics")
            _ensure_dir(dst_dir)

            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, "global_stats.csv"))

        except Exception as e:
            logging.warning("Auto-chooser: could not mirror metrics into metrics/: %s", e)

        # Compute a scalar score from the candidate metrics.
        score = _compute_eval_score(cand_cfg, metric=metric)

        if score is None or not np.isfinite(score):
            logging.warning("[auto] Candidate %d produced no valid score.", i)
            continue

        cand_name = str(scheme.get('name', f'candidate_{i:02d}'))
        cand_sig = _topic_signature(cand_cfg)

        row = {
            'idx': i,
            'candidate_name': cand_name,
            'candidate_signature': cand_sig,
            'topic_method': scheme.get('topic_method', cand_cfg.get('topic_method', 'unknown')),
            'score_metric': metric,
            'score': float(score),
            'output_dir': cand_cfg.get('output_dir'),
            'n_estimators': cand_cfg.get('xgb_params', {}).get('n_estimators'),
            'boruta_max_iter': cand_cfg.get('boruta_max_iter'),
        }

        leaderboard.append(row)

        if (best_score is None) or (score > best_score):
            best_score = score
            best_cfg = cand_cfg
            best_scheme = scheme
            best_row = copy.deepcopy(row)

    # Persist leaderboard for audit.
    if leaderboard:
        try:
            df_lb = pd.DataFrame(leaderboard)
            df_lb.sort_values('score', ascending=False, inplace=True)

            leaderboard_path = os.path.join(base_out, 'auto_chooser_leaderboard.csv')
            df_lb.to_csv(leaderboard_path, index=False)

            logging.info(
                "[auto] Wrote leaderboard with %d scored candidate(s) to %s",
                len(df_lb),
                leaderboard_path)

        except Exception as e:
            logging.warning("[auto] Could not write leaderboard: %s", e)

    # 3) Build full-strength config from the winner.
    if best_cfg is None or best_scheme is None or best_row is None or best_score is None:
        logging.warning(
            "[auto] No candidate produced a valid score; returning original config. "
            "Attempted=%d, scored=%d.",
            attempted,
            len(leaderboard))
        return base_cfg

    final_cfg = copy.deepcopy(base_cfg)

    # Overlay ONLY the winning candidate's topic/routing keys onto the original config.
    # This intentionally does not carry over fast-eval reductions unless they were part
    # of the user-specified candidate scheme.
    for k, v in best_scheme.items():
        final_cfg[k] = v

    # Restore original output_dir.
    final_cfg['output_dir'] = base_cfg.get('output_dir', 'network_output')

    # Restore full-strength knobs from the original config.
    if 'xgb_params' in base_cfg:
        final_cfg['xgb_params'] = copy.deepcopy(base_cfg['xgb_params'])

    if 'boruta_max_iter' in base_cfg:
        final_cfg['boruta_max_iter'] = int(base_cfg['boruta_max_iter'])

    final_cfg['shap_sample_size'] = int(base_cfg.get('shap_sample_size', 250))
    final_cfg['descriptive_figures_enable'] = bool(
        base_cfg.get('descriptive_figures_enable', False))
    final_cfg['pipeline_dump_sidecar_meta'] = bool(
        base_cfg.get('pipeline_dump_sidecar_meta', True))
    final_cfg['pipeline_compress'] = base_cfg.get('pipeline_compress', 'auto')
    final_cfg['boruta_cache'] = bool(base_cfg.get('boruta_cache', False))

    # Keep the chosen metric and candidate cap for reproducibility.
    final_cfg['auto_eval_metric'] = metric
    final_cfg['auto_eval_max_candidates'] = max_cand

    # Store selected-candidate provenance.
    selected_idx = int(best_row.get('idx', -1))
    selected_name = best_row.get(
        'candidate_name',
        best_scheme.get(
            'name',
            f'candidate_{selected_idx:02d}' if selected_idx >= 0 else 'candidate_selected')
    )

    final_cfg['auto_selected_candidate_index'] = selected_idx
    final_cfg['auto_selected_candidate_name'] = selected_name
    final_cfg['auto_selected_candidate_score'] = float(best_score)
    final_cfg['auto_selected_candidate_signature'] = best_row.get(
        'candidate_signature',
        _topic_signature(best_cfg)
    )
    final_cfg['auto_selected_candidate_output_dir'] = best_cfg.get('output_dir', None)

    # Store Auto-Chooser audit summary.
    final_cfg['auto_n_candidates_attempted'] = attempted
    final_cfg['auto_n_candidates_scored'] = len(leaderboard)

    # Backward-compatible alias: previously "evaluated" meant scored candidates.
    final_cfg['auto_n_candidates_evaluated'] = len(leaderboard)

    final_cfg['auto_selection_runtime_sec'] = float(time.time() - _t0_auto)

    logging.info(
        "[auto] Winner: idx=%s name=%s method=%s score=%.6f metric=%s output_dir=%s",
        final_cfg.get('auto_selected_candidate_index'),
        final_cfg.get('auto_selected_candidate_name'),
        final_cfg.get('topic_method', 'unknown'),
        final_cfg.get('auto_selected_candidate_score'),
        final_cfg.get('auto_eval_metric'),
        final_cfg.get('auto_selected_candidate_output_dir'),
    )

    logging.info(
        "[auto] Evaluation completed in %.2fs; attempted=%d, scored=%d.",
        final_cfg.get('auto_selection_runtime_sec', float('nan')),
        final_cfg.get('auto_n_candidates_attempted', attempted),
        final_cfg.get('auto_n_candidates_scored', len(leaderboard)),
    )

    # Write compact JSON selection summary for manual inspection.
    try:
        auto_selection_meta = {
            'auto_eval_metric': final_cfg.get('auto_eval_metric'),
            'auto_eval_max_candidates': final_cfg.get('auto_eval_max_candidates'),
            'auto_selected_candidate_index': final_cfg.get('auto_selected_candidate_index'),
            'auto_selected_candidate_name': final_cfg.get('auto_selected_candidate_name'),
            'auto_selected_candidate_score': final_cfg.get('auto_selected_candidate_score'),
            'auto_selected_candidate_signature': final_cfg.get('auto_selected_candidate_signature'),
            'auto_selected_candidate_output_dir': final_cfg.get('auto_selected_candidate_output_dir'),
            'auto_n_candidates_attempted': final_cfg.get('auto_n_candidates_attempted'),
            'auto_n_candidates_scored': final_cfg.get('auto_n_candidates_scored'),
            'auto_n_candidates_evaluated': final_cfg.get('auto_n_candidates_evaluated'),
            'auto_selection_runtime_sec': final_cfg.get('auto_selection_runtime_sec'),
        }

        _ensure_dir(base_out)

        with open(
            os.path.join(base_out, 'auto_selected_candidate.json'),
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(auto_selection_meta, f, indent=2)

    except Exception as e:
        logging.warning("[auto] Could not write auto_selected_candidate.json: %s", e)

    return final_cfg
    

