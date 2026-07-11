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
from functools import wraps

_ERRNO_HINTS = {  
    errno.EACCES: "Permission denied.",  
    errno.ENOENT: "A parent directory in the path does not exist.",  
    errno.ENOSPC: "No space left on device.",  
    errno.EROFS:  "Read-only file system.",  
    errno.ENAMETOOLONG: "Path name is too long.",  
    errno.ENOTDIR: "A path component is not a directory.",  
    errno.EEXIST: "Path exists but may not be a directory.",  
}  

# ----------------------------- CLI & LOGGING ---------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """
    Parse CLI arguments with Jupyter-friendly behavior (ignores -f).
    Only primitive keys from DEFAULT_CONFIG are exposed for quick overrides.

    Notes
    -----
    - Boolean flags require explicit values: --hp_recommender_run true or --hp_use_recommended_params false.
    - Compressor help reflects supported options: zlib, lz4, xz/lzma.
    - Avoids creating two separate flags for the same 'pipeline_compress' key.
      by reserving the dashed form as the single public CLI.
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        v = str(v).strip().lower()
        if v in ("yes","y","true","t","1"): return True
        if v in ("no","n","false","f","0"): return False
        raise argparse.ArgumentTypeError("Expected boolean (true/false).")

    parser = argparse.ArgumentParser(
        description=("(XTQSPRKinetic) QSPR Reaction Kinetics Pipeline "
                     "(LDA/FCM/GMM/HDBSCAN/QDA, Bootstrap, Auto-Chooser, Boruta-LGBM)"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument('--config', type=str, default=None, help='Path to YAML or JSON config file.')
    parser.add_argument('--help_config', action='store_true', help='Print DEFAULT_CONFIG and exit.')
    # Jupyter kernel flag; safely ignored
    parser.add_argument('-f', '--f', dest='_jupyter_kernel', default=None, help=argparse.SUPPRESS)

    # Single public compressor option uses dashed form; keeps help tidy and avoids dupes
    parser.add_argument(
        '--pipeline-compress',
        dest='pipeline_compress',
        type=str,
        default=None,
        help=("Compression for pipeline.pkl. "
              "Use 'auto', 'none', an int (0..9), "
              "'zlib[:LEVEL]', 'lz4[:LEVEL]', or 'xz[:LEVEL]'. "
              "Examples: none | 0 | 3 | zlib | zlib:6 | lz4 | lz4:3 | xz:6")
    )

    # Primitive-type overrides from DEFAULT_CONFIG (skip 'pipeline_compress' to avoid duplicate option)
    for k, d in DEFAULT_CONFIG.items():
        if k == 'pipeline_compress':
            continue
        if isinstance(d, bool):
            parser.add_argument(f'--{k}', type=str2bool, default=None, metavar='{true|false}')
        elif isinstance(d, (int, float, str)):
            parser.add_argument(f'--{k}', type=type(d), default=None)

    if argv is None:
        argv = sys.argv[1:]
    args, _unknown = parser.parse_known_args(argv)

    if args.help_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2))
        sys.exit(0)

    return args


# Centralized helper that resolves every None in config to a safe, recommended default.

def _apply_diagnostics_aliases(
    cfg: Dict[str, Any],
    user_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Backward-compatible migration from v2.5.19 reviewer_* keys to
    v2.7.0 diagnostics_* keys.

    Official public keys are diagnostics_*.
    reviewer_* keys are deprecated aliases.

    If both old and new keys are supplied by the user, the new diagnostics_*
    key takes precedence.
    """
    user_keys = set(user_keys or [])

    alias_map = {
        'reviewer_diagnostics_enable': 'diagnostics_enable',
        'reviewer_dataset_summary_enable': 'diagnostics_dataset_summary_enable',
        'reviewer_alignment_enable': 'diagnostics_alignment_enable',
        'reviewer_random_controls_mode': 'diagnostics_random_controls_mode',
        'reviewer_random_controls_enable': 'diagnostics_random_controls_enable',
        'reviewer_random_controls_n': 'diagnostics_random_controls_n',
        'reviewer_random_controls_preserve_sizes': 'diagnostics_random_controls_preserve_sizes',
        'reviewer_ablation_enable': 'diagnostics_ablation_enable',
        'reviewer_ablation_use_all_features': 'diagnostics_ablation_use_all_features',
        'reviewer_ablation_include_rmg_label_experts': 'diagnostics_ablation_include_rmg_label_experts',
        'reviewer_residual_enable': 'diagnostics_residual_enable',
        'reviewer_residual_top_n': 'diagnostics_residual_top_n',
        'reviewer_stability_summary_enable': 'diagnostics_stability_summary_enable',
        'reviewer_k_sensitivity_enable': 'diagnostics_k_sensitivity_enable',
        'reviewer_k_values': 'diagnostics_k_values',
        'reviewer_k_sensitivity_shap_sample_size': 'diagnostics_k_sensitivity_shap_sample_size',
    }

    for old_key, new_key in alias_map.items():
        if old_key in cfg and new_key not in user_keys:
            cfg[new_key] = cfg[old_key]
            logging.warning(
                "Config key '%s' is deprecated; use '%s' instead.",
                old_key,
                new_key,
            )

    return cfg

# =============================================================================
#                    YAML LOADING WITH DUPLICATE-KEY WARNINGS
# =============================================================================

class DuplicateKeyWarningLoader(yaml.SafeLoader):
    """
    YAML loader that detects duplicate mapping keys during config-file loading.

    Behavior:
        - Logs duplicate YAML keys with line/column location.
        - Preserves standard PyYAML behavior: the final/later key value wins.
        - Does not merge duplicate keys automatically.

    This is mainly intended to prevent silent user mistakes such as defining
    candidate_schemes: more than once in the same YAML file.
    """
    pass

def _construct_mapping_with_duplicate_warning(loader, node, deep=False):
    """
    Construct a YAML mapping while recording duplicate keys.

    Duplicate-key policy:
        - Detect duplicate keys during YAML parsing.
        - Store duplicate-key metadata on the loader for later provenance.
        - Preserve standard PyYAML behavior: the final/later value overwrites
          the earlier value.
        - Do not merge duplicate dictionaries or lists.

    Notes:
        This function intentionally avoids merging duplicate dictionaries or 
        lists to prevent unsafe overrides for keys like output_dir, xgb_params, 
        target_cols, or candidate_schemes; similarly, it does not log directly, 
        but instead defers logging to log_yaml_duplicate_key_summary(config)—executed 
        after setup_logging(...)—to ensure warnings are properly captured within the 
        official pipeline/Auto-Chooser log.
    """
    if not isinstance(node, yaml.MappingNode):
        raise yaml.constructor.ConstructorError(
            None,
            None,
            f"Expected a mapping node, but found {node.id}",
            node.start_mark,
        )

    # Supports YAML merge keys such as << if they are used.
    loader.flatten_mapping(node)

    mapping = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)

        if key in mapping:
            duplicate_info = {
                "key": str(key),
                "line": int(key_node.start_mark.line + 1),
                "column": int(key_node.start_mark.column + 1),
            }

            if not hasattr(loader, "_duplicate_keys"):
                loader._duplicate_keys = []

            loader._duplicate_keys.append(duplicate_info)

        value = loader.construct_object(value_node, deep=deep)
        # Standard PyYAML-style overwrite behavior:
        # the final/later duplicate key value becomes the effective config value.
        mapping[key] = value

    return mapping


DuplicateKeyWarningLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_with_duplicate_warning,
)

def _yaml_safe_load_with_duplicate_warnings(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load a YAML configuration file while recording duplicate keys.

    Returns:
        data:
            Loaded YAML mapping. If duplicate keys are present, the final/later
            value is retained.
        duplicates:
            List of duplicate-key records, each containing key, line, and column.

    Duplicate-key behavior:
        - Final/later values overwrite earlier values.
        - Duplicate lists or dictionaries are not merged.
        - Duplicate-key metadata is returned for later official logging after
          setup_logging(...) is active.

    This preserves normal YAML/PyYAML semantics while making accidental duplicate
    config blocks visible in logs and runtime provenance.
    """
    with open(path, "r", encoding="utf-8") as f:
        loader = DuplicateKeyWarningLoader(f)

        try:
            data = loader.get_single_data()
            duplicates = getattr(loader, "_duplicate_keys", [])
        finally:
            loader.dispose()

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML config must be a mapping/dictionary: {path}")

    return data, duplicates

def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge DEFAULT_CONFIG, optional config file (YAML/JSON), and CLI overrides.

    Priority:
        CLI > file config > DEFAULT_CONFIG

    Additional behavior:
        - Expands ~ and environment variables in common path fields.
        - Handles pipeline_compress through the dedicated --pipeline-compress
        CLI option.
        - Supports backward compatibility from old reviewer_* YAML keys to
        new diagnostics_* keys in v2.6.0.
        - Ensures required Boruta parameter dictionaries are present.
        - Detects duplicate YAML keys during config-file loading and logs a warning
        with the duplicated key name and YAML line/column location.
        - Preserves standard YAML/PyYAML overwrite behavior for duplicate keys:
        the final/later value is used and earlier values are not automatically
        merged.
        - Stores duplicate-key metadata as internal runtime provenance using:
            _yaml_duplicate_keys
            _yaml_duplicate_key_count
        These metadata keys do not need to be added to DEFAULT_CONFIG.
    """
    cfg = resolve_none_config(copy.deepcopy(DEFAULT_CONFIG))

    # -------------------------------------------------------------------------
    # 1) File config
    # -------------------------------------------------------------------------
    file_cfg: Dict[str, Any] = {}
    cfg_path = getattr(args, 'config', None)

    if cfg_path:
        cfg_path = os.path.expandvars(os.path.expanduser(cfg_path))

        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        try:
            duplicate_keys: List[Dict[str, Any]] = []

            if cfg_path.lower().endswith(('.yml', '.yaml')):
                loaded, duplicate_keys = _yaml_safe_load_with_duplicate_warnings(cfg_path)
            else:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)

            if isinstance(loaded, dict):
                file_cfg = {k: v for k, v in loaded.items() if v is not None}
            else:
                logging.warning("Config file did not contain a dict; ignoring.")

        except Exception as e:
            raise RuntimeError(f"Failed to read config file '{cfg_path}': {e}")

        cfg.update(file_cfg)

        # Runtime provenance only. 
        if duplicate_keys:
            cfg['_yaml_duplicate_keys'] = duplicate_keys
            cfg['_yaml_duplicate_key_count'] = len(duplicate_keys)

    # -------------------------------------------------------------------------
    # 2) CLI overrides
    # -------------------------------------------------------------------------
    # Track keys supplied explicitly by the user through CLI. This is needed so
    # deprecated reviewer_* aliases do not overwrite explicit diagnostics_* CLI
    # settings.
    cli_keys: Set[str] = set()

    # Dedicated compressor flag: --pipeline-compress.
    # This is handled separately because parse_args() skips pipeline_compress
    # in the automatic DEFAULT_CONFIG argument loop to avoid duplicate options.
    if getattr(args, 'pipeline_compress', None) is not None:
        cfg['pipeline_compress'] = args.pipeline_compress
        cli_keys.add('pipeline_compress')

    # Primitive-type overrides from DEFAULT_CONFIG.
    for k, d in DEFAULT_CONFIG.items():
        if k == 'pipeline_compress':
            continue

        if isinstance(d, (bool, int, float, str)):
            v = getattr(args, k, None)
            if v is not None:
                cfg[k] = v
                cli_keys.add(k)

    # -------------------------------------------------------------------------
    # 3) Backward compatibility for old reviewer_* YAML/CLI keys
    # -------------------------------------------------------------------------
    # If an old reviewer_* key is present, map it to the corresponding
    # diagnostics_* key unless the user explicitly supplied the diagnostics_*
    # key in either the config file or CLI.
    #
    # Requires _apply_diagnostics_aliases(cfg, user_keys=...) to be defined.
    user_keys = set(file_cfg.keys()) | cli_keys
    cfg = _apply_diagnostics_aliases(cfg, user_keys=user_keys)

    # -------------------------------------------------------------------------
    # 4) Path hygiene
    # -------------------------------------------------------------------------
    for path_key in (
        'input_csv',
        'output_dir',
        'descriptor_cache_dir',
        'boruta_cache_dir',
    ):
        if path_key in cfg and isinstance(cfg[path_key], str):
            cfg[path_key] = os.path.expandvars(os.path.expanduser(cfg[path_key]))

    # -------------------------------------------------------------------------
    # 5) Sanity checks for Boruta parameter blocks
    # -------------------------------------------------------------------------
    # Ensure required Boruta parameter dictionaries exist. This protects against
    # incomplete YAML files or accidental null values.
    if not isinstance(cfg.get('boruta_lgbm_params'), dict):
        cfg['boruta_lgbm_params'] = copy.deepcopy(
            DEFAULT_CONFIG['boruta_lgbm_params']
        )

    if not isinstance(cfg.get('boruta_rf_params'), dict):
        cfg['boruta_rf_params'] = copy.deepcopy(
            DEFAULT_CONFIG['boruta_rf_params']
        )

    return cfg

def log_yaml_duplicate_key_summary(config: Dict[str, Any]) -> None:
    """
    Replay duplicate-YAML-key warnings after setup_logging(...) is active.

    Duplicate-key detection happens during load_config(...), often before the
    final file/stdout log handlers are initialized. This helper re-logs the
    stored duplicate-key provenance so the warning appears in the official run
    log and console output.
    """
    duplicates = config.get('_yaml_duplicate_keys', [])

    if not duplicates:
        return

    logging.warning(
        "[config] YAML duplicate-key audit: %d duplicate key occurrence(s) were "
        "detected during config loading. The final/later value was used for each.",
        len(duplicates),
    )

    for d in duplicates:
        logging.warning(
            "[config] Duplicate YAML key '%s' at line %s, column %s. "
            "The later value overwrote the earlier value.",
            d.get('key', 'unknown'),
            d.get('line', 'unknown'),
            d.get('column', 'unknown'),
        )

def _to_int(x, default: Optional[int]=None, min_val: Optional[int]=None) -> Optional[int]:
    try:
        v = int(x)
        if (min_val is not None) and (v < min_val):
            return min_val
        return v
    except Exception:
        return default

def _to_float(x, default: Optional[float]=None, min_val: Optional[float]=None) -> Optional[float]:
    try:
        v = float(x)
        if (min_val is not None) and (v < min_val):
            return min_val
        return v
    except Exception:
        return default
    
def resolve_none_config(cfg_in: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve `None` (or 'None') placeholders in config to recommended defaults.

    Strategy:
      • Inherit values from related keys (e.g., gmm_n_components ← n_topics).
      • Provide robust defaults for bootstrap mirrors and PCA random states.
      • Normalize enumerations and guard invalid choices.
      • Log a compact summary of changes.

    Returns
    -------
    dict : a NEW dict with all replacements applied (input is not mutated).
    """
    cfg = copy.deepcopy(cfg_in)
    changes = []

    def _is_none(v) -> bool:
        return v is None or (isinstance(v, str) and v.strip().lower() == "none")

    def _set(key, val):
        old = cfg.get(key, None)
        if old != val:
            cfg[key] = val
            changes.append(f"{key}={val!r}")

    # --- Core dependencies / inheritance -------------------------------------
    rs = cfg.get('random_state', 42)
    if _is_none(rs):
        _set('random_state', int(42))

    # Topic counts
    n_topics = cfg.get('n_topics', 6)
    if _is_none(n_topics):
        _set('n_topics', int(6))
    n_topics = cfg.get('n_topics', 6)

    # LDA: components defaults
	# n_components
    n_comp = cfg.get('lda_n_topics', None)
    if _is_none(n_comp):
        _set('lda_n_topics', int(cfg.get('n_topics', 6)))

    # learning method
    lm = str(cfg.get('lda_learning_method', 'batch')).lower()
    if lm not in ('batch', 'online'):
        lm = 'batch'
    _set('lda_learning_method', lm)
    
    # priors (scikit-learn expects float or None; if user passed "auto", coerce to None)
    alpha = cfg.get('lda_doc_topic_prior', None)
    beta  = cfg.get('lda_topic_word_prior', None)
    if isinstance(alpha, str) and alpha.lower() == 'auto':
        logging.warning("LDA: 'auto' prior not supported by sklearn; using None.")
        alpha = None
    if isinstance(beta, str) and beta.lower() == 'auto':
        logging.warning("LDA: 'auto' prior not supported by sklearn; using None.")
        beta = None
    alpha = None if alpha in (None, 'None', '') else _to_float(alpha, None, None)
    beta  = None if beta  in (None, 'None', '') else _to_float(beta, None, None)

    _set('lda_doc_topic_prior', alpha)
    _set('lda_topic_word_prior', beta)
    
    if _is_none(cfg.get('lda_max_iter', None)):
        _set('lda_max_iter', int(50))

    # GMM: components defaults
    if _is_none(cfg.get('gmm_n_components', None)):
        _set('gmm_n_components', int(cfg.get('n_topics', 6)))

    cov = str(cfg.get('gmm_covariance_type', 'full')).lower()
    if cov not in ('full', 'tied', 'diag', 'spherical'):
        cov = 'full'
    _set('gmm_covariance_type', cov)

    initp = str(cfg.get('gmm_init_params', 'kmeans')).lower().strip()
    if '++' in initp:  # sklearn doesn’t support 'kmeans++' here
        logging.warning("GMM: 'kmeans++' init not supported; using 'kmeans'.")
        initp = 'kmeans'
    if initp not in ('kmeans', 'random'):
        logging.warning("GMM: unknown init '%s'; using 'kmeans'.", initp)
        initp = 'kmeans'
    _set('gmm_init_params', initp)

    if _is_none(cfg.get('gmm_reg_covar', None)):
        _set('gmm_reg_covar', float(1e-6))

    if _is_none(cfg.get('gmm_n_init', None)):
        _set('gmm_n_init', int(1))

    if _is_none(cfg.get('gmm_tol', None)):
        _set('gmm_tol', float(1e-3))
    
    if _is_none(cfg.get('gmm_max_iter', None)):
        _set('gmm_max_iter', int(100))

    if _is_none(cfg.get('gmm_random_state', None)):
        _set('gmm_random_state', int(cfg.get('random_state', 42)))

    # GMM alignment cost normalization (defensive)
    cost = str(cfg.get('bootstrap_gmm_cost', 'mean_euclid')).lower()
    if cost not in ('mean_euclid', 'mean_mahal', 'gaussian_kl'):
        _set('bootstrap_gmm_cost', 'mean_euclid')

    # FCM: k defaults to n_topics
    if _is_none(cfg.get('fcm_k', None)):
        _set('fcm_k', n_topics)

    # PCA: if enabled & using DRFP & n_components is None → set heuristic
    if bool(cfg.get('pca_enable', False)):
        if _is_none(cfg.get('pca_n_components', None)):
            n_bits = int(cfg.get('n_bits', 2048))
            # Heuristic: quarter of DRFP bits, clamped to [16, 256]
            suggested = max(16, min(256, n_bits // 4))
            _set('pca_n_components', suggested)

    if _is_none(cfg.get('pca_random_state', None)):
        _set('pca_random_state', int(cfg.get('random_state', 42)))

    # HDBSCAN: min_samples defaults to min_cluster_size when None
    if _is_none(cfg.get('hdbscan_min_samples', None)):
        _set('hdbscan_min_samples', int(cfg.get('hdbscan_min_cluster_size', 10)))

    # Bootstrap: mirror enable flags from bootstrap_lda_enable when None
    base_boot = bool(cfg.get('bootstrap_lda_enable', False))
    for k in ('bootstrap_gmm_enable', 'bootstrap_hdbscan_enable', 'bootstrap_fcm_enable', 'bootstrap_qda_enable'):
        if _is_none(cfg.get(k, None)):
            _set(k, base_boot)

    # Auto-chooser: cap candidates if None (keeps sweeps bounded)
    if _is_none(cfg.get('auto_eval_max_candidates', None)):
        _set('auto_eval_max_candidates', 20)

    # --- Enumerations / normalization ----------------------------------------
    # topic_method
    valid_topic = {'lda','fcm','lda+fcm','gmm','lda+gmm','hdbscan','lda+hdbscan','lda+qda'}
    tm = str(cfg.get('topic_method', 'lda')).lower().replace(' ', '')
    if tm not in valid_topic:
        logging.warning("Invalid topic_method=%r; defaulting to 'lda'.", tm)
        _set('topic_method', 'lda')
    else:
        if cfg.get('topic_method') != tm:
            _set('topic_method', tm)

    # family_selection_mode
    fsm = str(cfg.get('family_selection_mode', 'fixed')).lower()
    if fsm not in {'fixed','auto'}:
        _set('family_selection_mode', 'fixed')
    else:
        if cfg.get('family_selection_mode') != fsm:
            _set('family_selection_mode', fsm)

    # relerr_mode
    rem = str(cfg.get('relerr_mode', 'smape')).lower()
    if rem not in {'abs_rel','smape','both'}:
        _set('relerr_mode', 'smape')
    else:
        if cfg.get('relerr_mode') != rem:
            _set('relerr_mode', rem)

    # boruta_single_target: must be one of the configured continuous target columns.
    if not bool(cfg.get('boruta_per_target', True)):
        target_cols_cfg = _get_target_cols(cfg, required=False)
        bst = str(cfg.get('boruta_single_target', target_cols_cfg[0] if target_cols_cfg else 'Ea'))
        if bst not in set(target_cols_cfg):
            _set('boruta_single_target', target_cols_cfg[0] if target_cols_cfg else 'Ea')
    
    # Normalize per-family Boruta enable flag to bool
    val_pfb = cfg.get('Experimental_Per_Family_Boruta', False)
    if isinstance(val_pfb, str):
        _set('Experimental_Per_Family_Boruta', val_pfb.strip().lower() in ('1','true','yes','on'))
    elif val_pfb is None:
        # leave as None to allow env-var fallback (if you chose None in DEFAULT_CONFIG)
        pass
    else:
        _set('Experimental_Per_Family_Boruta', bool(val_pfb))

    # Paths (expand at this stage if still “None”-ish)
    for path_key in ('input_csv','output_dir','descriptor_cache_dir','boruta_cache_dir'):
        pv = cfg.get(path_key, None)
        if isinstance(pv, str) and pv:
            # expand ~ and env vars
            expanded = os.path.expandvars(os.path.expanduser(pv))
            if expanded != pv:
                _set(path_key, expanded)

    # Sanity for nested Boruta params
    if not isinstance(cfg.get('boruta_lgbm_params'), dict):
        cfg['boruta_lgbm_params'] = copy.deepcopy(DEFAULT_CONFIG['boruta_lgbm_params'])
        changes.append("boruta_lgbm_params=<default>")
    if not isinstance(cfg.get('boruta_rf_params'), dict):
        cfg['boruta_rf_params'] = copy.deepcopy(DEFAULT_CONFIG['boruta_rf_params'])
        changes.append("boruta_rf_params=<default>")

    # Log summary (single line)
    if changes:
        logging.info("Config None-resolver applied: %s", ", ".join(changes))
    else:
        logging.info("Config None-resolver: no changes needed.")

    return cfg

def setup_logging(log_path: str):
    """
    Initialize logging to both file and stdout with a consistent format.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root = logging.getLogger(); root.setLevel(logging.INFO)
    for h in list(root.handlers): root.removeHandler(h)
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8'); fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    root.addHandler(fh); root.addHandler(ch)
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    # scikit-learn >=1.6 emits this FutureWarning through HDBSCAN's internal
    # validation path. It is dependency-level deprecation noise, not a pipeline
    # data-quality warning. Keep the filter narrow so other FutureWarnings remain visible.
    warnings.filterwarnings(
        "ignore",
        message=r"'force_all_finite' was renamed to 'ensure_all_finite' in 1\.6 and will be removed in 1\.8\.",
        category=FutureWarning,
    )

# --- Timing utilities ---------------------------------------------------------

def _format_elapsed(seconds: float) -> str:
    """
    Human-readable elapsed-time formatter.

    Uses milliseconds for sub-second operations so fast steps do not appear as
    misleading '0.00s' entries in the log.
    """
    try:
        seconds = float(seconds)
    except Exception:
        return "nan"

    if not math.isfinite(seconds):
        return "nan"

    if seconds < 1.0:
        return f"{seconds * 1000.0:.2f}ms"

    if seconds < 60.0:
        return f"{seconds:.2f}s"

    minutes = int(seconds // 60)
    rem = seconds - 60.0 * minutes

    if minutes < 60:
        return f"{minutes}m {rem:.2f}s"

    hours = int(minutes // 60)
    minutes = minutes % 60
    return f"{hours}h {minutes}m {rem:.2f}s"


def _write_run_runtime_artifacts(
    *,
    config: Dict[str, Any],
    label: str,
    started_utc: str,
    ended_utc: str,
    elapsed_seconds: float,
    status: str,
    error: Optional[str] = None,
) -> None:
    """
    Write total run-time artifacts for one full configuration run.

    This is intentionally lightweight and does not affect scientific outputs.
    """
    try:
        out_dir = str(config.get('output_dir', '')).strip()
        if not out_dir:
            return

        _ensure_dir(out_dir)

        payload = {
            'label': str(label),
            'pipeline_version': str(config.get('pipeline_version', 'unknown')),
            'status': str(status),
            'started_utc': started_utc,
            'ended_utc': ended_utc,
            'elapsed_seconds': float(elapsed_seconds),
            'elapsed_minutes': float(elapsed_seconds / 60.0),
            'elapsed_human': _format_elapsed(elapsed_seconds),
            'input_csv': config.get('input_csv', None),
            'output_dir': config.get('output_dir', None),
            'target_cols': list(config.get('target_cols', []))
                           if isinstance(config.get('target_cols', []), (list, tuple))
                           else config.get('target_cols', None),
            'topic_method': config.get('topic_method', None),
            'n_topics': config.get('n_topics', None),
            'mordred_parallel_enable': bool(config.get('mordred_parallel_enable', False)),
            'mordred_parallel_n_jobs': config.get('mordred_parallel_n_jobs', None),
            'boruta_parallel_enable': bool(config.get('boruta_parallel_enable', False)),
            'boruta_parallel_n_jobs': config.get('boruta_parallel_n_jobs', None),
            'shap_sample_size': config.get('shap_sample_size', None),
            'descriptive_figures_enable': bool(config.get('descriptive_figures_enable', False)),
            'figure_label_index_base': config.get('figure_label_index_base', 1),
        }

        if error is not None:
            payload['error'] = str(error)

        json_path = os.path.join(out_dir, 'run_runtime.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        csv_path = os.path.join(out_dir, 'run_runtime_summary.csv')
        pd.DataFrame([payload]).to_csv(csv_path, index=False)

        logging.info(
            "Total configuration run time written to %s and %s.",
            json_path,
            csv_path,
        )

    except Exception as e:
        logging.warning("Could not write total run-time artifacts: %s", e)


@contextmanager
def log_timing(label: str):
    """Context manager to log the wall-clock duration of a code block."""
    _t0 = time.perf_counter()
    try:
        yield
    finally:
        _dt = time.perf_counter() - _t0
        logging.info("%s completed in %s", label, _format_elapsed(_dt))


def timeit_logger(label: str | None = None):
    """
    Decorator to log function wall-clock time without changing behavior.

    For the full run_pipeline(...) call, this also writes:
        run_runtime.json
        run_runtime_summary.csv
    """
    def _wrap(func):
        @wraps(func)
        def _inner(*args, **kwargs):
            run_label = label or func.__name__
            started_dt = datetime.now(timezone.utc)
            started_utc = started_dt.isoformat().replace('+00:00', 'Z')
            _t0 = time.perf_counter()
            status = 'success'
            err_text = None

            try:
                return func(*args, **kwargs)

            except Exception as exc:
                status = 'failed'
                err_text = repr(exc)
                raise

            finally:
                ended_dt = datetime.now(timezone.utc)
                ended_utc = ended_dt.isoformat().replace('+00:00', 'Z')
                _dt = time.perf_counter() - _t0

                logging.info("%s finished in %s", run_label, _format_elapsed(_dt))

                # Only write total run artifacts for the full pipeline call.
                if str(run_label).lower() == 'run pipeline':
                    cfg = None
                    if 'config' in kwargs and isinstance(kwargs['config'], dict):
                        cfg = kwargs['config']
                    elif args and isinstance(args[0], dict):
                        cfg = args[0]

                    if isinstance(cfg, dict):
                        _write_run_runtime_artifacts(
                            config=cfg,
                            label=run_label,
                            started_utc=started_utc,
                            ended_utc=ended_utc,
                            elapsed_seconds=float(_dt),
                            status=status,
                            error=err_text,
                        )

        return _inner
    return _wrap

# --- Model validation utilities ----------------------------------------------------------

def _abs_err(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """
    Absolute error in natural units: |pred - true|.

    This metric is scale-dependent but denominator-free, so it remains
    statistically well-defined even when the reference target contains zeros.
    """
    y = pd.Series(y_true, copy=False).astype(float)
    yp = pd.Series(y_pred, copy=False).astype(float)

    if len(y) != len(yp):
        logging.warning(
            "Absolute error: length mismatch between y_true (%d) and y_pred (%d). "
            "Using the common prefix length.",
            len(y),
            len(yp),
        )

    n = min(len(y), len(yp))
    yv = y.iloc[:n].to_numpy(dtype=float)
    ypv = yp.iloc[:n].to_numpy(dtype=float)

    out = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(yv) & np.isfinite(ypv)

    if not np.all(valid):
        logging.warning(
            "Absolute error: %d/%d rows contain non-finite y_true or y_pred; "
            "their absolute errors are written as NaN.",
            int(n - np.sum(valid)),
            int(n),
        )

    out[valid] = np.abs(ypv[valid] - yv[valid])
    return pd.Series(out, index=y.index[:n])


def _abs_rel_err(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-6) -> pd.Series:
    """
    Absolute relative error: |pred - true| / |true|.

    Strict zero-aware reporting policy:
    - If |true| <= eps, the relative error is mathematically undefined.
    - Such rows are returned as NaN rather than divided by eps.
    - This prevents artificially huge AARD/relative-error values for true-zero
      or near-zero targets.
    - Use sMAPE as the bounded zero-tolerant relative-error diagnostic.
    """
    y = pd.Series(y_true, copy=False).astype(float)
    yp = pd.Series(y_pred, copy=False).astype(float)

    if len(y) != len(yp):
        logging.warning(
            "Absolute Relative Error: length mismatch between y_true (%d) and y_pred (%d). "
            "Using the common prefix length.",
            len(y),
            len(yp),
        )

    n = min(len(y), len(yp))
    yv = y.iloc[:n].to_numpy(dtype=float)
    ypv = yp.iloc[:n].to_numpy(dtype=float)

    out = np.full(n, np.nan, dtype=float)
    finite_pair = np.isfinite(yv) & np.isfinite(ypv)
    valid = finite_pair & (np.abs(yv) > float(eps))

    invalid_zero = int(np.sum(finite_pair & (np.abs(yv) <= float(eps))))
    if invalid_zero > 0:
        logging.warning(
            "Absolute Relative Error: %d/%d finite rows have |y_true| <= %.3g. "
            "Relative error is undefined for these rows and is written as NaN.",
            invalid_zero,
            int(n),
            float(eps),
        )

    if np.any(valid):
        out[valid] = np.abs(ypv[valid] - yv[valid]) / np.abs(yv[valid])
    else:
        logging.warning(
            "Absolute Relative Error: no rows with finite |y_true| > %.3g. "
            "All relative-error values are NaN.",
            float(eps),
        )

    return pd.Series(out, index=y.index[:n])


def _smape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-6) -> pd.Series:
    """
    Symmetric MAPE:
        2 * |pred - true| / (|pred| + |true| + eps)

    sMAPE is bounded and zero-tolerant. When true = pred = 0, the numerator is
    zero and the value is effectively 0. Range is approximately [0, 2].
    """
    y = pd.Series(y_true, copy=False).astype(float)
    yp = pd.Series(y_pred, copy=False).astype(float)

    if len(y) != len(yp):
        logging.warning(
            "sMAPE: length mismatch between y_true (%d) and y_pred (%d). "
            "Using the common prefix length.",
            len(y),
            len(yp),
        )

    n = min(len(y), len(yp))
    yv = y.iloc[:n].to_numpy(dtype=float)
    ypv = yp.iloc[:n].to_numpy(dtype=float)

    out = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(yv) & np.isfinite(ypv)

    if not np.all(valid):
        logging.warning(
            "sMAPE: %d/%d rows contain non-finite y_true or y_pred; "
            "their sMAPE values are written as NaN.",
            int(n - np.sum(valid)),
            int(n),
        )

    num = 2.0 * np.abs(ypv[valid] - yv[valid])
    den = np.abs(ypv[valid]) + np.abs(yv[valid]) + float(eps)
    out[valid] = num / den

    zero_zero_count = int(np.sum(valid & (np.abs(yv) <= float(eps)) & (np.abs(ypv) <= float(eps))))
    if zero_zero_count > 0:
        logging.debug(
            "sMAPE: %d rows have both y_true and y_pred approximately zero; "
            "sMAPE is effectively 0 for these rows.",
            zero_zero_count,
        )

    return pd.Series(out, index=y.index[:n])


def _metric_finite_arrays(y_true, y_pred=None, weights=None):
    """
    Return finite y_true/y_pred/(weights) arrays for robust metric computation.

    Negative weights are clipped to zero because negative averaging weights are
    not meaningful for the weighted error metrics used here.
    """
    try:
        y = np.asarray(y_true, dtype=float).reshape(-1)
    except Exception:
        return np.array([], dtype=float), None, None

    if y_pred is None:
        if weights is None:
            mask = np.isfinite(y)
            return y[mask], None, None

        try:
            w = np.asarray(weights, dtype=float).reshape(-1)
        except Exception:
            return np.array([], dtype=float), None, None

        n = min(y.size, w.size)
        y = y[:n]
        w = w[:n]

        if np.any(np.isfinite(w) & (w < 0)):
            logging.warning(
                "Metric calculation encountered negative weights. "
                "Negative weights are clipped to zero."
            )

        mask = np.isfinite(y) & np.isfinite(w)
        return y[mask], None, np.clip(w[mask], 0.0, None)

    try:
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float), None

    if weights is None:
        n = min(y.size, yp.size)
        y = y[:n]
        yp = yp[:n]
        mask = np.isfinite(y) & np.isfinite(yp)
        return y[mask], yp[mask], None

    try:
        w = np.asarray(weights, dtype=float).reshape(-1)
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float), None

    n = min(y.size, yp.size, w.size)
    y = y[:n]
    yp = yp[:n]
    w = w[:n]

    if np.any(np.isfinite(w) & (w < 0)):
        logging.warning(
            "Metric calculation encountered negative weights. "
            "Negative weights are clipped to zero."
        )

    mask = np.isfinite(y) & np.isfinite(yp) & np.isfinite(w)
    return y[mask], yp[mask], np.clip(w[mask], 0.0, None)


def _relative_valid_mask(values, eps: float = 1e-12):
    """
    Mask samples whose true value can support a relative-error denominator.

    Values with |true| <= eps are treated as undefined for AARD/ARE-like
    metrics and are omitted from the relative-error average.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    return np.isfinite(v) & (np.abs(v) > float(eps))


def _safe_r2_score(y_true, y_pred, eps: float = 1e-12) -> float:
    """
    Ordinary R² with strict statistical handling.

    R² = 1 - SSE / SST

    Returns np.nan if:
    - fewer than two finite paired samples are available,
    - the target variance is zero or numerically negligible,
    - the computed value is non-finite.

    Note:
    This strict behavior is more suitable for scientific reporting than forcing
    sklearn-style finite fallback values for degenerate target subsets.
    """
    y, yp, _ = _metric_finite_arrays(y_true, y_pred)

    if y.size < 2:
        logging.warning(
            "R2 calculation skipped: fewer than two finite paired samples. Returning NaN."
        )
        return np.nan

    den = float(np.sum((y - np.mean(y)) ** 2))
    if not np.isfinite(den) or den <= float(eps):
        logging.warning(
            "R2 calculation skipped: target variance is zero or numerically negligible. "
            "R2 is undefined for this subset. Returning NaN."
        )
        return np.nan

    try:
        val = float(r2_score(y, yp))
    except Exception as e:
        logging.warning("R2 calculation failed: %s. Returning NaN.", e)
        return np.nan

    if not np.isfinite(val):
        logging.warning("R2 calculation produced a non-finite value. Returning NaN.")
        return np.nan

    if val < 0:
        logging.warning(
            "R2 calculation: result is negative (%.4f). The model performs worse "
            "than a mean-baseline predictor for this subset.",
            val,
        )

    return val


def _safe_mae(y_true, y_pred) -> float:
    """
    Mean Absolute Error over finite paired samples.
    Returns NaN if no finite paired samples are available.
    """
    y, yp, _ = _metric_finite_arrays(y_true, y_pred)
    if y.size == 0:
        logging.warning("MAE calculation skipped: no finite paired samples. Returning NaN.")
        return np.nan
    return float(np.mean(np.abs(y - yp)))


def _safe_rmse(y_true, y_pred) -> float:
    """
    Root Mean Squared Error over finite paired samples.
    Returns NaN if no finite paired samples are available.
    """
    y, yp, _ = _metric_finite_arrays(y_true, y_pred)
    if y.size == 0:
        logging.warning("RMSE calculation skipped: no finite paired samples. Returning NaN.")
        return np.nan
    return float(np.sqrt(np.mean((y - yp) ** 2)))


def _adj_r2(r2_value: float, n: int, p: int) -> float:
    """
    Adjusted R² given ordinary R², sample size n, and number of predictors p.

    R²_adj = 1 - (1 - R²) * (n - 1) / (n - p - 1)

    Returns np.nan if not well-defined:
    - non-finite ordinary R²,
    - invalid n or p,
    - n <= p + 1, where the denominator is zero or negative.
    """
    try:
        r2 = float(r2_value)
    except Exception:
        return np.nan

    if not np.isfinite(r2):
        logging.warning(
            "Adjusted R2 calculation failed: input R2 is not finite (%s). Returning NaN.",
            r2_value,
        )
        return np.nan

    try:
        n = int(n)
        p = int(p)
    except Exception:
        return np.nan

    if n <= 1 or p < 0:
        logging.warning(
            "Adjusted R2 calculation failed: invalid n=%d or p=%d. Returning NaN.",
            n,
            p,
        )
        return np.nan

    if n <= p + 1:
        logging.warning(
            "Adjusted R2 calculation failed: sample size n=%d is too small for "
            "number of predictors p=%d. Condition n > p + 1 is violated. "
            "This indicates insufficient degrees of freedom or severe "
            "over-parameterization. Returning NaN.",
            n,
            p,
        )
        return np.nan

    try:
        val = 1.0 - (1.0 - r2) * ((n - 1.0) / (n - p - 1.0))
        return float(val) if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


def _nrmse(y_true, y_pred, eps: float = 1e-12) -> float:
    """
    Normalized RMSE:

        nRMSE = RMSE / |mean(y_true)|

    If |mean(y_true)| <= eps, nRMSE is undefined and NaN is returned rather
    than dividing by eps.
    """
    y, yp, _ = _metric_finite_arrays(y_true, y_pred)

    if y.size == 0:
        logging.warning("nRMSE calculation skipped: no finite paired samples. Returning NaN.")
        return np.nan

    rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
    mean_true = float(np.mean(y))

    if not np.isfinite(mean_true) or abs(mean_true) <= float(eps):
        logging.warning(
            "nRMSE is undefined because |mean(y_true)| <= %.3g. Returning NaN.",
            float(eps),
        )
        return np.nan

    return float(rmse / abs(mean_true))


def _aard(y_true, y_pred, eps: float = 1e-12) -> float:
    """
    Average Absolute Relative Deviation.

        AARD = mean(|y_pred - y_true| / |y_true|)

    Strict zero-aware policy:
    - Rows with |y_true| <= eps are omitted because their relative error is
      undefined.
    - They are not divided by eps.
    - If no valid denominator remains, NaN is returned.
    """
    y, yp, _ = _metric_finite_arrays(y_true, y_pred)

    if y.size == 0:
        logging.warning("AARD calculation skipped: no finite paired samples. Returning NaN.")
        return np.nan

    valid = _relative_valid_mask(y, eps=eps) & np.isfinite(yp)
    omitted = int(y.size - np.sum(valid))

    if omitted > 0:
        logging.warning(
            "AARD: omitted %d/%d rows with non-finite values or |y_true| <= %.3g; "
            "relative error is undefined for these rows.",
            omitted,
            int(y.size),
            float(eps),
        )

    if not np.any(valid):
        logging.warning(
            "AARD could not be computed: no valid non-zero true values. Returning NaN."
        )
        return np.nan

    rel = np.abs(y[valid] - yp[valid]) / np.abs(y[valid])
    rel = rel[np.isfinite(rel)]

    if rel.size == 0:
        logging.warning("AARD could not be computed after filtering invalid rows. Returning NaN.")
        return np.nan

    return float(np.mean(rel))


def _rae(y_true, y_pred, eps: float = 1e-12) -> float:
    """
    Relative Absolute Error.

        RAE = sum(|y_true - y_pred|) / sum(|y_true - mean(y_true)|)

    If y_true is constant, the denominator is zero and RAE is statistically
    undefined. In that case, NaN is returned.
    """
    y, yp, _ = _metric_finite_arrays(y_true, y_pred)

    if y.size == 0:
        logging.warning("RAE calculation skipped: no finite paired samples. Returning NaN.")
        return np.nan

    numerator = float(np.sum(np.abs(y - yp)))
    denominator = float(np.sum(np.abs(y - np.mean(y))))

    if not np.isfinite(denominator) or denominator <= float(eps):
        logging.warning(
            "RAE denominator is zero or numerically negligible because y_true is "
            "constant. RAE is undefined. Returning NaN."
        )
        return np.nan

    val = numerator / denominator
    return float(val) if np.isfinite(val) else np.nan


def _w_nrmse(y_true, y_pred, weights, eps: float = 1e-12) -> float:
    """
    Weighted normalized RMSE:

        w_nRMSE = weighted_RMSE / |weighted_mean(y_true)|

    If |weighted_mean(y_true)| <= eps, the metric is undefined and NaN is
    returned rather than dividing by eps.
    """
    y, yp, w = _metric_finite_arrays(y_true, y_pred, weights)

    if y.size == 0 or w is None:
        logging.warning("Weighted nRMSE calculation skipped: no finite weighted samples. Returning NaN.")
        return np.nan

    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= float(eps):
        logging.warning(
            "Weighted nRMSE calculation skipped: sum of non-negative weights is "
            "zero or numerically negligible. Returning NaN."
        )
        return np.nan

    rmse = float(np.sqrt(np.average((y - yp) ** 2, weights=w)))
    mean_true = float(np.average(y, weights=w))

    if not np.isfinite(mean_true) or abs(mean_true) <= float(eps):
        logging.warning(
            "Weighted nRMSE is undefined because |weighted mean(y_true)| <= %.3g. "
            "Returning NaN.",
            float(eps),
        )
        return np.nan

    return float(rmse / abs(mean_true))


def _w_aard(y_true, y_pred, weights, eps: float = 1e-12) -> float:
    """
    Weighted Average Absolute Relative Deviation.

    Rows with |y_true| <= eps are omitted from the weighted average because
    their relative error is undefined.
    """
    y, yp, w = _metric_finite_arrays(y_true, y_pred, weights)

    if y.size == 0 or w is None:
        logging.warning("Weighted AARD calculation skipped: no finite weighted samples. Returning NaN.")
        return np.nan

    valid = (
        _relative_valid_mask(y, eps=eps)
        & np.isfinite(yp)
        & np.isfinite(w)
        & (w > 0)
    )

    omitted = int(y.size - np.sum(valid))
    if omitted > 0:
        logging.warning(
            "Weighted AARD: omitted %d/%d rows with non-finite values, zero weight, "
            "or |y_true| <= %.3g.",
            omitted,
            int(y.size),
            float(eps),
        )

    if not np.any(valid):
        logging.warning(
            "Weighted AARD could not be computed: no valid weighted non-zero true values. "
            "Returning NaN."
        )
        return np.nan

    w_sum = float(np.sum(w[valid]))
    if not np.isfinite(w_sum) or w_sum <= float(eps):
        logging.warning(
            "Weighted AARD calculation skipped: sum of valid weights is zero or "
            "numerically negligible. Returning NaN."
        )
        return np.nan

    rel = np.abs(y[valid] - yp[valid]) / np.abs(y[valid])
    rel_valid = np.isfinite(rel)

    if not np.any(rel_valid):
        logging.warning("Weighted AARD could not be computed after filtering. Returning NaN.")
        return np.nan

    return float(np.average(rel[rel_valid], weights=w[valid][rel_valid]))


def _w_rae(y_true, y_pred, weights, eps: float = 1e-12) -> float:
    """
    Weighted Relative Absolute Error.

        wRAE = sum(w * |y_true - y_pred|)
               / sum(w * |y_true - weighted_mean(y_true)|)

    If the weighted target variance/dispersion is zero, the metric is undefined
    and NaN is returned.
    """
    y, yp, w = _metric_finite_arrays(y_true, y_pred, weights)

    if y.size == 0 or w is None:
        logging.warning("Weighted RAE calculation skipped: no finite weighted samples. Returning NaN.")
        return np.nan

    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= float(eps):
        logging.warning(
            "Weighted RAE calculation skipped: sum of non-negative weights is "
            "zero or numerically negligible. Returning NaN."
        )
        return np.nan

    numerator = float(np.sum(w * np.abs(y - yp)))
    mean_true = float(np.average(y, weights=w))
    denominator = float(np.sum(w * np.abs(y - mean_true)))

    if not np.isfinite(denominator) or denominator <= float(eps):
        logging.warning(
            "Weighted RAE denominator is zero or numerically negligible because "
            "weighted y_true is constant. Weighted RAE is undefined. Returning NaN."
        )
        return np.nan

    val = numerator / denominator
    return float(val) if np.isfinite(val) else np.nan


def _w_r2(y, yhat, w, eps: float = 1e-12) -> float:
    """
    Weighted ordinary R².

        wR² = 1 - weighted_SSE / weighted_SST

    Returns NaN if:
    - no finite weighted samples are available,
    - total positive weight is zero,
    - weighted target variance is zero or numerically negligible.

    This is weighted ordinary R², not weighted adjusted R². Therefore, no
    n <= p + 1 degrees-of-freedom check is applied here.
    """
    y, yhat, w = _metric_finite_arrays(y, yhat, w)

    if y.size == 0 or w is None:
        logging.warning("Weighted R2 calculation skipped: no finite weighted samples. Returning NaN.")
        return np.nan

    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= float(eps):
        logging.warning(
            "Weighted R2 calculation failed: sum of non-negative weights is zero "
            "or numerically negligible. Returning NaN."
        )
        return np.nan

    mu = float(np.sum(w * y) / w_sum)
    den = float(np.sum(w * (y - mu) ** 2))

    if not np.isfinite(den) or den <= float(eps):
        logging.warning(
            "Weighted R2 calculation failed: weighted variance of true values is "
            "zero or numerically negligible. Target variable is constant under "
            "the supplied weights. Weighted R2 is undefined. Returning NaN."
        )
        return np.nan

    num = float(np.sum(w * (y - yhat) ** 2))
    if not np.isfinite(num):
        logging.warning(
            "Weighted R2 calculation failed: weighted residual sum of squares is "
            "not finite. Returning NaN."
        )
        return np.nan

    r2 = 1.0 - num / den

    if not np.isfinite(r2):
        logging.warning(
            "Weighted R2 calculation failed: computed weighted R2 is not finite. "
            "Returning NaN."
        )
        return np.nan

    if r2 < 0:
        logging.warning(
            "Weighted R2 calculation: result is negative (%.4f). The model performs "
            "worse than a weighted mean baseline for this weighted subset.",
            r2,
        )

    return float(r2)

# --- File utilities ----------------------------------------------------------

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def _ensure_dir(path: str) -> None:  
    """  
    Ensure the given directory exists using os.makedirs.  
    - Creates parents recursively.  
    - No error if it already exists.  
    - Logs and re-raises unexpected failures.  
    """ 
    try:  
        os.makedirs(path, exist_ok=True)  
    except OSError as e:  
        # Logs full stack trace for diagnostics
        hint = _ERRNO_HINTS.get(e.errno, "Unexpected OS error.")  
        logging.exception("Failed to ensure directory '%s' [errno=%s]: %s",  
                         path, e.errno, hint)  

# --- Compression utilities ----------------------------------------------------

# --- robust, config-driven compression (zlib / lz4 / lzma) ----------
def _normalize_compress(c):
    """
    Accepts:
          - None / 'auto' -> prefer lz4:3 if available, else lzma:3, else zlib:3
          - 'none'/'off'/0/False -> 0 (no compression)
          - int 0..9 -> zlib with that level
          - 'zlib' or ('zlib', lvl)  -> zlib
          - 'lz4'  or ('lz4', lvl)   -> lz4 (converted to zlib if HAVE_LZ4 is False)
          - 'lzma'/'xz' or ('lzma', lvl)/('xz', lvl) -> lzma/xz (converted to zlib if HAVE_LZMA False)
        Returns a value suitable for joblib.dump(compress=...).
    """
    # auto mode
    if c is None or (isinstance(c, str) and c.strip().lower() == 'auto'):
        if HAVE_LZ4:
            return ('lz4', 3)
        if HAVE_LZMA:
            return ('xz', 3)
        return ('zlib', 3)
    # booleans
    if isinstance(c, bool):
        return 0 if not c else _normalize_compress('auto')
    # integers -> zlib level
    if isinstance(c, int):
        return max(0, min(9, c))
    # strings
    if isinstance(c, str):
        s = c.strip().lower()
        if s in ('none', 'no', 'false', 'off', '0'): return 0
        if s == 'lz4':                  return ('lz4', 3) if HAVE_LZ4 else ('zlib', 3)
        if s in ('lzma', 'xz'):         return ('xz', 3) if HAVE_LZMA else ('zlib', 3)
        if s in ('zlib', 'gzip'):       return ('zlib', 3)
        # unknown -> safe default
        return ('zlib', 3)
    # tuples/lists: (method, level)
    if isinstance(c, (list, tuple)) and len(c) >= 1:
        method = str(c[0]).lower()
        level = 3
        if len(c) >= 2:
            try:
                level = int(c[1])
            except Exception:
                level = 3
        level = max(0, min(9, level))
        if method == 'lz4':
            return ('lz4', level) if HAVE_LZ4 else ('zlib', level)
        if method in ('lzma', 'xz'):
            return ('xz', level) if HAVE_LZMA else ('zlib', level)
        if method in ('zlib', 'gzip'):
            return ('zlib', level)
        # unknown method -> treat as zlib level
        return level
    # fallback
    return ('zlib', 3)

# --- Tiny helper to decode the chosen compressor ----------
def _compress_to_method_level(compress):
    """
    Normalize joblib 'compress' parameter into ('method', level) for metadata.
    Accepts: 0..9 | ('zlib', L) | ('lz4', L) | ('xz', L)
             or 0 / None => ('none', 0)
    """
    if compress in (None, 0, False):
        return ('none', 0)
    if isinstance(compress, int):
        return ('zlib', int(max(0, min(9, compress))))
    if isinstance(compress, (list, tuple)) and len(compress) >= 1:
        method = str(compress[0]).lower()
        lvl = 3
        if len(compress) >= 2:
            try: lvl = int(compress[1])
            except Exception: lvl = 3
        lvl = max(0, min(9, lvl))
        if method in ('zlib', 'gzip'):
            return ('zlib', lvl)
        if method in ('lz4',):
            return ('lz4', lvl)
        if method in ('xz', 'lzma'):
            return ('xz', lvl)
        return ('zlib', lvl)
    # Fallback
    return ('zlib', 3)

# Magic-bytes detector (for programs not using joblib)
def detect_file_compressor(path: str) -> str:
    """
    Best-effort detector from file magic bytes.
    Returns one of: 'zstd'|'lz4'|'xz'|'gzip'|'zlib'|'none'|'unknown'
    """
    with open(path, 'rb') as f:
        head = f.read(8)

    # Zstandard magic: 28 B5 2F FD
    if head.startswith(b'\x28\xB5\x2F\xFD'):
        return 'zstd'
    # LZ4 frame magic: 04 22 4D 18, plus skippable frames 184D2A50..5F (little endian)
    if head.startswith(b'\x04\x22\x4D\x18') or head[:4] in [bytes([0x50,0x2A,0x4D,0x18])]:
        return 'lz4'
    # XZ/LZMA container magic: FD 37 7A 58 5A 00
    if head.startswith(b'\xFD7zXZ\x00'):
        return 'xz'
    # GZIP magic: 1F 8B
    if head.startswith(b'\x1F\x8B'):
        return 'gzip'
    # Common zlib headers: 78 01/5E/9C/DA
    if len(head) >= 2 and head[0] == 0x78 and head[1] in (0x01, 0x5E, 0x9C, 0xDA):
        return 'zlib'
    # Pickle (no compression) often starts with 0x80 (PROTO opcode), but not guaranteed
    if head[:1] == b'\x80':
        return 'none'
    return 'unknown'

def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dictionaries (used for scheme overrides)."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def _deep_merged(base: dict, override: dict) -> dict:
    """Recursively update dict `base` in-place with `override`."""
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merged(base[k], v)
        else:
            base[k] = v
    return base

# ------------------------- TARGET CONFIGURATION HELPERS -----------------------

DEFAULT_TARGET_COLS = ['lnA', 'n', 'Ea']

def _as_clean_list(value: Any) -> List[str]:
    """Normalize comma/space separated strings or sequences into a clean string list."""
    if value is None:
        return []
    if isinstance(value, str):
        s = value.replace(',', ' ').strip()
        return [x.strip() for x in s.split() if x.strip()]
    if isinstance(value, (list, tuple, set, pd.Index, np.ndarray)):
        return [str(x).strip() for x in list(value) if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _get_target_cols(config: Optional[Dict[str, Any]] = None,
                     df: Optional[pd.DataFrame] = None,
                     required: bool = False) -> List[str]:
    """
    Return canonical supervised target columns.

    Priority:
        1) config['target_cols']      (new canonical key)
        2) config['hp_target_cols']   (legacy/backward-compatible key)
        3) ['lnA', 'n', 'Ea']         (default Arrhenius case study)

    If df is supplied and required=True, all target columns must be present.
    """
    cfg = config or {}
    cols = _as_clean_list(cfg.get('target_cols', None))
    if not cols:
        cols = _as_clean_list(cfg.get('hp_target_cols', None))
    if not cols:
        cols = list(DEFAULT_TARGET_COLS)

    # De-duplicate while preserving order.
    cols = list(dict.fromkeys(cols))
    if required and isinstance(df, pd.DataFrame):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Input CSV is missing required target column(s): {missing}. Available columns: {list(df.columns)}")
    return cols


def _target_cols_from_frame(Y: Optional[pd.DataFrame],
                            fallback: Optional[Sequence[str]] = None) -> List[str]:
    """Infer target columns from a target DataFrame, with a backward-compatible fallback."""
    if isinstance(Y, pd.DataFrame) and len(Y.columns):
        return [str(c) for c in Y.columns]
    fb = _as_clean_list(fallback)
    return fb if fb else list(DEFAULT_TARGET_COLS)


def _safe_target_token(target: Any) -> str:
    """Return a file-name-safe token for a target name."""
    s = str(target)
    out = ''.join(ch if (ch.isalnum() or ch in ['_', '.', '-']) else '_' for ch in s)
    out = out.strip('_')
    return out or 'target'


def _prediction_columns(target_cols: Sequence[str]) -> List[str]:
    """Prediction column names corresponding to target_cols."""
    return [f"{str(t)}_pred" for t in target_cols]


def _coerce_prediction_vector(yhat: Any, n_targets: int) -> List[float]:
    """Convert a model prediction row to length n_targets, padding/truncating safely."""
    arr = np.asarray(yhat, dtype=float).ravel()
    out = np.full(int(n_targets), np.nan, dtype=float)
    m = min(out.size, arr.size)
    if m > 0:
        out[:m] = arr[:m]
    return [float(x) if np.isfinite(x) else np.nan for x in out]


def _resolve_prediction_target_cols(target_cols: Optional[Sequence[str]],
                                    models: Optional[Dict[Any, Any]] = None) -> List[str]:
    """Resolve target names for prediction outputs, using model output count when needed."""
    cols = _as_clean_list(target_cols)
    if cols:
        return cols
    n_out = None
    try:
        if isinstance(models, dict):
            m = models.get('global', None)
            n_out = getattr(m, 'n_outputs_', None)
            if n_out is None and hasattr(m, 'estimators_'):
                n_out = len(getattr(m, 'estimators_', []))
    except Exception:
        n_out = None
    if n_out is None:
        n_out = len(DEFAULT_TARGET_COLS)
    if int(n_out) == len(DEFAULT_TARGET_COLS):
        return list(DEFAULT_TARGET_COLS)
    return [f'target_{i}' for i in range(int(n_out))]