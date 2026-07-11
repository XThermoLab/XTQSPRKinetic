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

"""
XTQSPRKinetic main driver.

This module is the primary command-line entry point for running XTQSPRKinetic
from the repository layout (without requiring package installation). It loads
configuration files, resolves run paths, initializes the pipeline, and executes
the selected XTQSPRKinetic workflow (including optional auto-selection modes).

For detailed execution examples and configuration usage, see:
docs/run_guide.md
"""


from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import types
import traceback
from typing import Any, Dict

try:
    import yaml  # pip install pyyaml
except Exception:
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    raise


# ----------------------------- repo paths -------------------------------------

HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "output")
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")

PKG_NAME = "XTQSPRKinetic"  # used to register pseudo-package for relative imports inside src/*


# ----------------------------- small utilities --------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (base is modified and returned)."""
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yml", ".yaml")):
            return yaml.safe_load(f)
        return json.load(f)


def _resolve_config_path(arg: str) -> str:
    """
    Resolve --config:
      - If arg is absolute or exists as given → return as-is.
      - If arg has no extension → try <CONFIGS_DIR>/<arg>.yml/.yaml/.json
      - Else → interpret relative to CONFIGS_DIR.
    """
    arg = os.path.expanduser(arg)
    if os.path.isabs(arg) and os.path.isfile(arg):
        return arg
    # as given, relative to CWD?
    if os.path.isfile(arg):
        return os.path.abspath(arg)

    # bare name without extension?
    root, ext = os.path.splitext(os.path.basename(arg))
    if not ext:
        for ext_try in (".yml", ".yaml", ".json"):
            cand = os.path.join(CONFIGS_DIR, root + ext_try)
            if os.path.isfile(cand):
                return cand

    # otherwise treat as path under configs
    cand = os.path.join(CONFIGS_DIR, arg)
    if os.path.isfile(cand):
        return cand

    raise FileNotFoundError(f"Config file not found: {arg} (searched under {CONFIGS_DIR})")


def load_user_config(path_arg: str) -> Dict[str, Any]:
    path = _resolve_config_path(path_arg)
    cfg = _load_yaml_or_json(path)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a mapping at top level.")
    return cfg


def write_effective_config(cfg: Dict[str, Any], out_dir: str, filename="resolved_config.yaml"):
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[info] Wrote effective config to: {out_path}")
    except Exception:
        print("[warn] Could not write resolved config:\n" + traceback.format_exc(), file=sys.stderr)


def _load_module_from_file(fullname: str, filepath: str) -> types.ModuleType:
    """Load a Python file as a module named `fullname`."""
    spec = importlib.util.spec_from_file_location(fullname, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {fullname} from {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


def import_src_modules():
    """
    Import the refactor modules from ../src and register a pseudo-package
    so relative imports like `from .shared_imports import ...` keep working.
    Only the modules required by main are loaded here.
    """
    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError(f"Expected src directory at: {SRC_DIR}")

    # Register pseudo-packages
    if PKG_NAME not in sys.modules:
        pkg = types.ModuleType(PKG_NAME)
        pkg.__path__ = [REPO_ROOT]
        sys.modules[PKG_NAME] = pkg
    subpkg_name = f"{PKG_NAME}.src"
    if subpkg_name not in sys.modules:
        subpkg = types.ModuleType(subpkg_name)
        subpkg.__path__ = [SRC_DIR]
        sys.modules[subpkg_name] = subpkg

    order = [
        "shared_imports",
        "config",
        "utils",
        "mordred_xtqspr_patch",
        "featurization",
        "selection",
        "prediction",
        "evaluation",
        "pipeline",
        "autoselection",
    ]
    loaded = {}
    for mod in order:
        fullname = f"{PKG_NAME}.src.{mod}"
        path = os.path.join(SRC_DIR, f"{mod}.py")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing module file: {path}")
        loaded[mod] = _load_module_from_file(fullname, path)

    # Return only what we need
    return loaded["config"], loaded["utils"], loaded["autoselection"], loaded["pipeline"]


def _is_rel(p: str) -> bool:
    return not os.path.isabs(p) and not (p.startswith("~") or p.startswith("${"))


def apply_project_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve key paths against the fixed repo folders.
    - If a path is absolute, leave it as-is.
    - If relative or bare name, map as:
        input_csv            → DATA_DIR/<value>
        output_dir           → OUTPUT_ROOT/<value>  (and ensure directory exists)
        *_cache_dir          → <output_dir>/cache   (if missing or relative)
    """
    cfg = dict(cfg)  # shallow copy

    # input_csv
    if "input_csv" in cfg and isinstance(cfg["input_csv"], str) and _is_rel(cfg["input_csv"]):
        cfg["input_csv"] = os.path.join(DATA_DIR, cfg["input_csv"])

    # output_dir
    out = cfg.get("output_dir", "output")
    if isinstance(out, str) and _is_rel(out):
        out_abs = os.path.join(OUTPUT_ROOT, out)
        cfg["output_dir"] = out_abs
    else:
        out_abs = out if isinstance(out, str) else OUTPUT_ROOT

    # cache dirs → under output_dir if not absolute
    for key in ("descriptor_cache_dir", "boruta_cache_dir"):
        if key in cfg and isinstance(cfg[key], str):
            if _is_rel(cfg[key]):
                cfg[key] = os.path.join(out_abs, cfg[key])
        else:
            # default to <output_dir>/cache when cache enabled
            if cfg.get(key.replace("_dir", ""), False):
                cfg[key] = os.path.join(out_abs, "cache")

    # (Optional) expand any tilde/env vars for safety
    for k, v in list(cfg.items()):
        if isinstance(v, str):
            cfg[k] = os.path.expandvars(os.path.expanduser(v))

    return cfg


# ---------------------------------- CLI ---------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run XTQSPRKinetic with a YAML/JSON config (fixed repo layout).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument(
        "--config",
        required=True,
        help="Config name (in ../configs) or a path (.yml/.yaml/.json). Examples: 'balanced.yml' or 'balanced'."
    )
    p.add_argument(
        "--chooser-log",
        default="auto_chooser.log",
        help="Filename for auto-chooser log (written under output_dir when auto mode is used).",
    )
    p.add_argument(
        "--dump-effective-config",
        action="store_true",
        help="Write merged+resolved config to output_dir/resolved_config.yaml.",
    )
    return p.parse_args(argv)


# ---------------------------------- main --------------------------------------

def main(argv=None):
    args = parse_args(argv)

    # Import modules from ../src
    config_mod, utils_mod, autoselect_mod, pipeline_mod = import_src_modules()

    # Merge defaults ← user config
    base = copy.deepcopy(getattr(config_mod, "DEFAULT_CONFIG"))

    config_path = _resolve_config_path(args.config)
    duplicate_keys = []

    if config_path.lower().endswith((".yml", ".yaml")) and hasattr(
        utils_mod, "_yaml_safe_load_with_duplicate_warnings"
    ):
        user, duplicate_keys = utils_mod._yaml_safe_load_with_duplicate_warnings(config_path)
    else:
        user = _load_yaml_or_json(config_path)

    if not isinstance(user, dict):
        raise ValueError("Config file must contain a mapping at top level.")

    merged = _deep_merge(base, user)

    # Runtime provenance only. These keys do not need to be in DEFAULT_CONFIG.
    if duplicate_keys:
        merged["_yaml_duplicate_keys"] = duplicate_keys
        merged["_yaml_duplicate_key_count"] = len(duplicate_keys)

    # Backward compatibility: map old reviewer_* keys to diagnostics_* keys
    # unless the new diagnostics_* key was explicitly supplied by the user.
    if hasattr(utils_mod, "_apply_diagnostics_aliases"):
        merged = utils_mod._apply_diagnostics_aliases(merged, user_keys=set(user.keys()))

    # Resolve project paths (data/output/cache) before normalization
    merged = apply_project_paths(merged)

    # Normalize/repair None-ish values with project-aware paths
    merged = utils_mod.resolve_none_config(merged)

    # Ensure output_dir exists early
    out_dir = merged.get("output_dir", OUTPUT_ROOT)
    os.makedirs(out_dir, exist_ok=True)

    # Optional separate diagnostics k-sensitivity grid. This branch launches
    # independent full-pipeline jobs and should be used only for final diagnostics.
    if bool(merged.get("diagnostics_k_sensitivity_enable", False)):
        try:
            utils_mod.setup_logging(os.path.join(out_dir, "diagnostics_k_sensitivity.log"))
            if hasattr(utils_mod, "log_yaml_duplicate_key_summary"):
                utils_mod.log_yaml_duplicate_key_summary(merged)
        except Exception:
            pass
        print("[info] Running diagnostics k-sensitivity grid ...")
        pipeline_mod.run_diagnostics_k_sensitivity_grid(merged)
        return

    # Auto-chooser if requested
    fam_mode = str(merged.get("family_selection_mode", "fixed")).lower()
    if fam_mode == "auto":
        try:
            if bool(merged.get("auto_eval_separate_log", True)):
                utils_mod.setup_logging(os.path.join(out_dir, args.chooser_log))
                if hasattr(utils_mod, "log_yaml_duplicate_key_summary"):
                    utils_mod.log_yaml_duplicate_key_summary(merged)
        except Exception:
            pass
        print("[info] Running auto-chooser sweep ...")
        merged = autoselect_mod.auto_choose_family_scheme(merged)

    # Optionally dump final effective config
    if args.dump_effective_config:
        write_effective_config(merged, out_dir)

    # Full-strength pipeline
    print("[info] Running full pipeline ...")
    pipeline_mod.run_pipeline(merged)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("[fatal] Unhandled error: " + str(e), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
