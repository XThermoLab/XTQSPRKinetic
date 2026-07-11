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
from .utils import (
    _smape, _abs_err, 
    _abs_rel_err, 
    _adj_r2, _rae, 
    _aard, _nrmse, 
    _w_r2, _w_rae, 
    _w_aard, _w_nrmse, 
    _compress_to_method_level, 
    _normalize_compress, 
    _ensure_dir,
    _get_target_cols,
    _target_cols_from_frame,
    _safe_target_token,
    _prediction_columns,
    _resolve_prediction_target_cols,
    setup_logging, timeit_logger,
    _format_elapsed,
    _safe_r2_score,
    _safe_mae,
    _safe_rmse,
    log_yaml_duplicate_key_summary,
    DEFAULT_TARGET_COLS,
)
from .featurization import *
from .featurization import (
    _bootstrap_method_enabled,
    _qda_feasibility_report,
    _format_qda_feasibility_message,
    _normalize_probability_rows,
)
from .selection import *
from .prediction import *
from .evaluation import *
from .evaluation import _publication_ready_rcparams, _save_current_figure_publication

# --- Package version lookup (lightweight, no heavy imports) ---
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
except Exception:  # Python < 3.8 fallback if you have importlib-metadata installed
    from importlib_metadata import version as _pkg_version, PackageNotFoundError  # type: ignore

# ------------------------------ PIPELINE -------------------------------------

def _safe_pkg_version(name: str) -> str:
    """Return installed package version string without importing the package."""
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return 'unavailable'
    except Exception:
        return 'unknown'
    
def _pipeline_version_from(cfg: Dict[str, Any]) -> str:
    """Return the pipeline version from config, falling back to DEFAULT_CONFIG or 'unknown'."""
    try:
        return str((cfg or {}).get('pipeline_version', DEFAULT_CONFIG.get('pipeline_version', 'unknown')))
    except Exception:
        return str(DEFAULT_CONFIG.get('pipeline_version', 'unknown'))

def _log_pipeline_banner(config: Dict[str, Any]) -> None:
    """Log a consistent banner with the configured pipeline version."""
    try:
        logging.info("QSPR/Reaction Kinetics Pipeline (v%s)", _pipeline_version_from(config))
    except Exception:
        pass

def zero_variance_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove zero-variance columns. Safe and deterministic.

    Args:
        df: DataFrame [N × D]

    Returns:
        DataFrame with constant columns removed.
    """
    nunique = df.nunique(dropna=False)
    keep = nunique[nunique > 1].index
    dropped = set(df.columns) - set(keep)
    if dropped:
        logging.info(f"Zero-variance filter removed {len(dropped)} columns.")
    return df.loc[:, keep]

def build_pipeline_components(
    *,
    version: str,
    config: Dict[str, Any],
    models: Dict[Any, Any],
    scalers: Dict[Any, Any],
    confirmed_features: List[str],
    X_full: pd.DataFrame,
    df: pd.DataFrame,
    topic_method_actual: str,
    routing_mode_actual: Optional[str] = None,
    lda_model: Optional[Any] = None,
    drfp_df: Optional[pd.DataFrame] = None,
    flags_df: Optional[pd.DataFrame] = None,
    fcm_centers: Optional[np.ndarray] = None,
    prefilter_report: Optional[Dict[str, Any]] = None,
    boruta_meta: Optional[Dict[str, Any]] = None,
    topic_proba_active_df: Optional[pd.DataFrame] = None,
    bootstrap_agg_topic_proba: Optional[pd.DataFrame] = None,
    bootstrap_stability: Optional[pd.DataFrame] = None,
    global_stats_df: Optional[pd.DataFrame] = None,
    family_stats_df: Optional[pd.DataFrame] = None,
    fcm_pca_model: Optional[Any] = None,
    gmm_model: Optional[Dict[str, Any]] = None,
    hdbscan_model: Optional[Dict[str, Any]] = None,
    qda_model: Optional[Dict[str, Any]] = None, 
) -> Dict[str, Any]:
    """
    Construct the serializable pipeline package (pickle payload).

    Notes
    -----
    * Schema remains aligned with current codebase.
    * Only attach artifacts that exist (guarded by `is not None`).
    * `boruta_meta` is made safe if missing.
    """
    routing_mode_requested = (
        'hybrid' if bool(config.get('use_hybrid_routing', False))
        else ('soft' if bool(config.get('use_soft_routing', True)) else 'hard')
    )
    routing_mode = str(routing_mode_actual or routing_mode_requested)
    
    safe_boruta = boruta_meta or {
        'mode': ('per_target' if bool(config.get('boruta_per_target', True))
                 else f"single:{config.get('boruta_single_target', 'Ea')}"),
        'estimator': str(config.get('boruta_estimator', 'rf')).lower(),
        'max_iter': int(config.get('boruta_max_iter', 100)),
        'n_confirmed': int(len(confirmed_features)),
        'feature_names': list(confirmed_features),
    }

    # Prefer the true training labels if provided; otherwise fall back to df['family']
    if isinstance(df, pd.DataFrame) and ('_training_family' in df.columns):
        _training_family = list(df['_training_family'])
    elif isinstance(df, pd.DataFrame) and ('family' in df.columns):
        _training_family = list(df['family'])
    else:
        _training_family = None

    pkg: Dict[str, Any] = {
        'version': str(version),
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'config_snapshot': copy.deepcopy(config),
        'routing_mode': routing_mode,
        'routing_mode_requested': routing_mode_requested,
        'topic_method': str(topic_method_actual),
        'target_cols': _get_target_cols(config, required=False),
        'features': list(confirmed_features),
        'models': models,
        'scalers': scalers,
        'prefilter_report': (copy.deepcopy(prefilter_report) if prefilter_report else None),
        'boruta_meta': safe_boruta,
        'training_index': list(X_full.index),   # lightweight provenance
        'training_family': _training_family,    # true training labels if available
    }

    # Topic assigners / clusterers (only include what exists)
    topic_bundle: Dict[str, Any] = {}
    if lda_model is not None:
        topic_bundle['lda'] = lda_model
    if fcm_centers is not None:
        topic_bundle['fcm'] = {'centers': np.array(fcm_centers)}
    if fcm_pca_model is not None:
        topic_bundle['fcm_pca'] = fcm_pca_model
    if gmm_model is not None:
        topic_bundle['gmm'] = gmm_model
    if hdbscan_model is not None:
        topic_bundle['hdbscan'] = hdbscan_model
    if qda_model is not None:                       
        topic_bundle['qda'] = qda_model             # expected shape: {'model': QDA, 'input': 'lda'|'drfp', 'pca': PCA|None}
    if topic_bundle:
        pkg['topic_assigners'] = topic_bundle

    # DRFP / flags are sizeable; include only if present and you intend a comprehensive pickle
    if drfp_df is not None:
        pkg['drfp_columns'] = list(drfp_df.columns)
    if flags_df is not None:
        pkg['flags_columns'] = list(flags_df.columns)

    # Probabilities used for routing in this run + bootstrap summaries if any
    if topic_proba_active_df is not None:
        pkg['topic_proba_active_columns'] = list(topic_proba_active_df.columns)
    if bootstrap_agg_topic_proba is not None:
        pkg['bootstrap_agg_topic_proba_shape'] = list(bootstrap_agg_topic_proba.shape)
    if bootstrap_stability is not None:
        pkg['bootstrap_stability_columns'] = list(bootstrap_stability.columns)

    # Metrics block (optional, size-aware)
    metrics_block: Dict[str, Any] = {}
    include_full = bool(config.get('pipeline_include_metrics', False))

    if isinstance(global_stats_df, pd.DataFrame) and not global_stats_df.empty:
        if include_full:
            metrics_block['global_stats'] = global_stats_df
        else:
            try:
                metrics_block['global_stats_summary'] = global_stats_df.describe(include='all')
            except Exception:
                metrics_block['global_stats_shape'] = list(global_stats_df.shape)

    if isinstance(family_stats_df, pd.DataFrame) and not family_stats_df.empty:
        if include_full:
            metrics_block['family_stats'] = family_stats_df
        else:
            try:
                metrics_block['family_stats_summary'] = family_stats_df.describe(include='all')
            except Exception:
                metrics_block['family_stats_shape'] = list(family_stats_df.shape)

    if metrics_block:
        pkg['metrics'] = metrics_block

    return pkg

@timeit_logger("Save pipeline components from config")
def save_pipeline_components_from_config(
    *,
    version: str,
    config: Dict[str, Any],
    models: Dict[Any, Any],
    scalers: Dict[Any, Any],
    confirmed_features: List[str],
    X_full: pd.DataFrame,
    df: pd.DataFrame,                                          # must contain the final 'family' column
    topic_method_actual: str,                                  # e.g. config['topic_method'] ('lda'|'fcm'|'lda+fcm')
    routing_mode_actual: Optional[str] = None,                 # actual executed routing mode after quality gates
    lda_model: Optional[Any] = None,
    drfp_df: Optional[pd.DataFrame] = None,
    flags_df: Optional[pd.DataFrame] = None,
    fcm_centers: Optional[np.ndarray] = None,
    prefilter_report: Optional[Dict[str, Any]] = None,
    boruta_meta: Optional[Dict[str, Any]] = None,
    topic_proba_active_df: Optional[pd.DataFrame] = None,      # the probabilities actually used for routing (if any)
    bootstrap_agg_topic_proba: Optional[pd.DataFrame] = None,
    bootstrap_stability: Optional[pd.DataFrame] = None,
    global_stats_df: Optional[pd.DataFrame] = None,
    family_stats_df: Optional[pd.DataFrame] = None,
    output_path: str,
    fcm_pca_model: Optional[Any] = None,
    gmm_model: Optional[Dict[str, Any]] = None,                # {'model': GMM, 'input': 'lda'|'drfp', 'pca': PCA|None}
    hdbscan_model: Optional[Dict[str, Any]] = None,            # {'model': HDBSCAN, 'input': 'lda'|'drfp', 'pca': PCA|None}
    qda_model: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Build the pipeline package and persist it to disk using joblib with compression.
    """
    pkg = build_pipeline_components(
        version=version,
        config=config,
        models=models,
        scalers=scalers,
        confirmed_features=confirmed_features,
        X_full=X_full,
        df=df,                                             
        topic_method_actual=topic_method_actual,
        routing_mode_actual=routing_mode_actual,
        lda_model=lda_model,
        drfp_df=drfp_df,
        flags_df=flags_df,
        fcm_centers=fcm_centers,
        prefilter_report=prefilter_report,
        boruta_meta=boruta_meta,
        topic_proba_active_df=topic_proba_active_df,
        bootstrap_agg_topic_proba=bootstrap_agg_topic_proba,
        bootstrap_stability=bootstrap_stability,
        global_stats_df=global_stats_df,
        family_stats_df=family_stats_df,
        fcm_pca_model=fcm_pca_model,
        gmm_model=gmm_model,
        hdbscan_model=hdbscan_model,
        qda_model=qda_model,
    )
    # Ensure parent directory exists before dumping
    _ensure_dir(os.path.dirname(output_path))

    # Compress to keep size reasonable while preserving completeness
    compress_cfg = config.get('pipeline_compress', 'auto')
    compress = _normalize_compress(compress_cfg)
    
    
    pipeline_ver = _pipeline_version_from(config)


    # Save with normalized compressor; if joblib doesn't know the method name,
    # fall back silently to zlib:3 (rare older joblib versions).
    try:
        method, level = _compress_to_method_level(compress)
            # Embed minimal dump metadata inside the pickle payload
        pkg['dump_info'] = {
            'pipeline_version': pipeline_ver,
            'compression': {'method': method, 'level': int(level)},
            'created_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'joblib':       _safe_pkg_version('joblib'),
            'numpy':        _safe_pkg_version('numpy'),
            'scikit-learn': _safe_pkg_version('scikit-learn'),
            'xgboost':      _safe_pkg_version('xgboost'),
            'lightgbm':     _safe_pkg_version('lightgbm'),
            'lz4':          _safe_pkg_version('lz4'),
            'xz':           _safe_pkg_version('lzma'),
            'python': sys.version.split()[0],
            'platform': platform.platform(),
        }
        joblib.dump(pkg, output_path, compress=compress)
        logging.info("Saved pipeline components to '%s' (compress=%s | HAVE_LZ4=%s, HAVE_LZMA=%s)",
                     output_path, compress, HAVE_LZ4, HAVE_LZMA)

    except ValueError as e:
        # Compatibility fallback for older joblib not recognizing 'lz4'/'xz' names.
        logging.info("Compressor '%s' not supported by joblib in this env; falling back to zlib:3. Details: %s",
                     compress, e)
        compress=('zlib', 3)
        method, level = _compress_to_method_level(compress)
            # Embed minimal dump metadata inside the pickle payload
        pkg['dump_info'] = {
            'pipeline_version': pipeline_ver,
            'compression': {'method': method, 'level': int(level)},
            'created_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'joblib':       _safe_pkg_version('joblib'),
            'numpy':        _safe_pkg_version('numpy'),
            'scikit-learn': _safe_pkg_version('scikit-learn'),
            'xgboost':      _safe_pkg_version('xgboost'),
            'lightgbm':     _safe_pkg_version('lightgbm'),
            'lz4':          _safe_pkg_version('lz4'),
            'xz':           _safe_pkg_version('lzma'),
            'python': sys.version.split()[0],
            'platform': platform.platform(),
        }
        joblib.dump(pkg, output_path, compress=compress)
        logging.info("Saved pipeline components to '%s' (compress=%s | HAVE_LZ4=%s, HAVE_LZMA=%s)",
                     output_path, compress, HAVE_LZ4, HAVE_LZMA)

    # Optional sidecar JSON with checksum for external tools
    if bool(config.get('pipeline_dump_sidecar_meta', True)):
        try:
            sha = hashlib.sha256()
            with open(output_path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    sha.update(chunk)
            sidecar = {
                'file': os.path.basename(output_path),
                'size_bytes': os.path.getsize(output_path),
                'sha256': sha.hexdigest(),
                'compression': {'method': method, 'level': int(level)},
                'created_utc': pkg['dump_info']['created_utc'],
                'pipeline_version': pkg['dump_info']['pipeline_version'],
                'joblib': pkg['dump_info']['joblib'],
                'numpy': pkg['dump_info']['numpy'],
                'scikit-learn':pkg['dump_info']['scikit-learn'],
                'xgboost': pkg['dump_info']['xgboost'],
                'lightgbm': pkg['dump_info']['lightgbm'],
                'lz4': pkg['dump_info']['lz4'],
                'xz': pkg['dump_info']['xz'],
                'python': pkg['dump_info']['python'],
                'platform': pkg['dump_info']['platform'],
            }
            sidecar_path = output_path + '.meta.json'
            with open(sidecar_path, 'w', encoding='utf-8') as f:
                json.dump(sidecar, f, indent=2)
            logging.info("Wrote pipeline sidecar meta: %s", sidecar_path)
        except Exception as e:
            logging.warning("Could not write pipeline sidecar meta: %s", e)

def _texts_from_drfp_bits(drfp_df: 'pd.DataFrame') -> 'List[List[str]]':
    """
    Build a text corpus from DRFP bit presence: each reaction -> list of 'b{idx}' tokens.
    Handles pandas DataFrame, NumPy arrays, lists-of-lists, and SciPy sparse matrices.
    """
    # Fast path for SciPy sparse matrices
    if hasattr(drfp_df, "tocsr"):
        X = drfp_df.tocsr()
        texts = []
        for i in range(X.shape[0]):
            row = X.getrow(i)
            toks = [f"b{int(j)}" for j in row.indices] or ["_empty_"]
            texts.append(toks)
        return texts

    # Pandas DataFrame -> .values; otherwise assume array-like
    try:
        X = drfp_df.values
    except Exception:
        X = drfp_df

    texts = []
    for row in X:
        if isinstance(row, np.ndarray):
            nz = np.nonzero(row > 0)[0]
            toks = [f"b{int(j)}" for j in nz]
        else:
            toks = [f"b{j}" for j, v in enumerate(row) if (v is not None and float(v) > 0.0)]
        texts.append(toks if toks else ["_empty_"])
    return texts

def _texts_from_drfp_ngrams(config: 'Dict[str, Any]', drfp_df: 'pd.DataFrame') -> 'Optional[List[List[str]]]':
    """
    Build per-reaction tokens from **unhashed DRFP n-grams** using DrfpEncoder (mapping=True).

    Uses your existing inputs:
      - config['input_csv']  -> CSV with 'reaction_smiles'
      - config['n_bits']     -> DRFP bit length (fallback: drfp_df.shape[1] or 2048)
      - config['radius']     -> DRFP radius (fallback: 3)
      - config['ring']       -> include ring features (DEFAULT: True). Also accepts legacy 'rings'.
    """
    try:
        from drfp import DrfpEncoder  # type: ignore
    except Exception:
        logging.info("drfp not available; cannot build drfp_ngrams texts.")
        return None

    input_csv = str(config.get('input_csv', '')).strip()
    if not input_csv or not os.path.exists(input_csv):
        logging.info("input_csv not found; cannot build drfp_ngrams texts.")
        return None

    try:
        df_src = pd.read_csv(input_csv)
    except Exception as e:
        logging.warning("Failed to read input_csv '%s': %s", input_csv, e)
        return None

    if 'reaction_smiles' not in df_src.columns:
        logging.warning("input_csv missing required 'reaction_smiles' column.")
        return None

    rxn_smiles = df_src['reaction_smiles'].astype(str).tolist()

    try:
        n_bits = int(config.get('n_bits',
                    (drfp_df.shape[1] if hasattr(drfp_df, 'shape') and len(getattr(drfp_df, 'shape', [])) > 1 else 2048)))
    except Exception:
        n_bits = 2048

    try:
        radius = int(config.get('radius', 3))
    except Exception:
        radius = 3

    ring = bool(config.get('ring', config.get('rings', True)))

    texts: List[List[str]] = []
    try:
        for rxn in rxn_smiles:
            fps, fmap = DrfpEncoder.encode([rxn],
                                           n_folded_length=n_bits,
                                           radius=radius,
                                           rings=ring,
                                           mapping=True)
            fp = fps[0]
            active = [i for i, v in enumerate(fp) if v]
            toks: set[str] = set()
            for j in active:
                grams = fmap.get(j)
                if grams:
                    toks.update(str(g) for g in grams)
            texts.append(sorted(toks) if toks else ["_empty_"])
        return texts
    except Exception as e:
        logging.warning("Failed to construct drfp_ngrams texts: %s", e)
        return None

def _brics_tokens_from_smiles(smi: str, to_smarts: bool = False, min_fragment_size: int = 1) -> 'List[str]':
    """Return BRICS fragment tokens for a single molecule SMILES."""
    if not globals().get('HAVE_RDKIT', False) or not smi:
        return []
    try:
        from rdkit import Chem
        from rdkit.Chem import BRICS
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return []
        frags = BRICS.BRICSDecompose(mol, minFragmentSize=int(min_fragment_size))  # set[str]
        if not frags:
            return []
        if not to_smarts:
            return sorted(frags)
        out = []
        for s in frags:
            fm = Chem.MolFromSmiles(s)
            if fm:
                out.append(Chem.MolToSmarts(fm))
        return sorted(set(out))
    except Exception:
        return []

def _brics_tokens_from_reaction_smiles(rxn: str, to_smarts: bool = False, min_fragment_size: int = 1) -> 'List[str]':
    """Tokenize a reaction SMILES (reactants>>products or reactants>reagents>products) into BRICS tokens."""
    if not rxn:
        return ["_empty_"]
    parts = rxn.split(">")
    if len(parts) == 2:      # reactants>>products
        reactants, products = parts[0], parts[1]
    elif len(parts) == 3:    # reactants>reagents>products
        reactants, _, products = parts
    else:
        reactants, products = rxn, ""
    toks = set()
    for smi in filter(None, (reactants.split(".") if reactants else [])):
        toks.update(_brics_tokens_from_smiles(smi, to_smarts=to_smarts, min_fragment_size=min_fragment_size))
    for smi in filter(None, (products.split(".") if products else [])):
        toks.update(_brics_tokens_from_smiles(smi, to_smarts=to_smarts, min_fragment_size=min_fragment_size))
    return sorted(toks) if toks else ["_empty_"]

def _texts_from_brics_smarts(config: 'Dict[str, Any]') -> 'Optional[List[List[str]]]':
    """
    Build texts from BRICS fragments using existing input source:
      - config['input_csv'] with column 'reaction_smiles'
    Config knobs:
      - 'brics_to_smarts': bool (default False)  -> SMARTS vs fragment SMILES
      - 'brics_min_fragment_size': int (default 1)
    """
    if not globals().get('HAVE_RDKIT', False):
        logging.info("RDKit not available; cannot build brics_smarts texts.")
        return None

    input_csv = str(config.get('input_csv', '')).strip()
    if not input_csv or not os.path.exists(input_csv):
        logging.info("input_csv not found; cannot build brics_smarts texts.")
        return None

    try:
        df_src = pd.read_csv(input_csv)
    except Exception as e:
        logging.warning("Failed to read input_csv '%s': %s", input_csv, e)
        return None

    if 'reaction_smiles' not in df_src.columns:
        logging.warning("input_csv missing required 'reaction_smiles' column for brics_smarts.")
        return None

    to_smarts = bool(config.get('brics_to_smarts', False))
    min_fragment_size = int(config.get('brics_min_fragment_size', 1))

    texts: List[List[str]] = []
    for rxn in df_src['reaction_smiles'].astype(str):
        toks = _brics_tokens_from_reaction_smiles(rxn, to_smarts=to_smarts, min_fragment_size=min_fragment_size)
        texts.append(toks if toks else ["_empty_"])
    logging.info("Coherence texts: brics_smarts (%d docs).", len(texts))
    return texts

def _get_coherence_texts(drfp_df: 'pd.DataFrame', config: 'Dict[str, Any]') -> 'List[List[str]]':
    """Choose token source per config['lda_coherence_token_source'] using your existing inputs."""
    src = str(config.get('lda_coherence_token_source', 'drfp_bits')).lower()

    if src == 'drfp_ngrams':
        texts = _texts_from_drfp_ngrams(config, drfp_df)
        if texts:
            logging.info("Coherence texts: drfp_ngrams (%d docs).", len(texts))
            return texts
        logging.info("drfp_ngrams requested but unavailable; falling back to drfp_bits.")

    elif src == 'brics_smarts':
        texts = _texts_from_brics_smarts(config)
        if texts:
            return texts
        logging.info("brics_smarts requested but unavailable; falling back to drfp_bits.")

    # Default / fallback
    texts = _texts_from_drfp_bits(drfp_df)
    logging.info("Coherence texts: drfp_bits (%d docs).", len(texts))
    return texts

def compute_lda_coherence_bits(lda_model,
                               drfp_df: 'pd.DataFrame',
                               config: 'Dict[str, Any]',
                               out_dir: str) -> None:
    """
    Compute topic coherence (c_v / c_npmi / u_mass) using selectable token sources:
      - lda_coherence_token_source: {'drfp_bits','drfp_ngrams','brics_smarts'}
    
    Writes:
      out_dir/lda_coherence_<metric>.json
      out_dir/lda_coherence_<metric>_per_topic.csv

    IMPORTANT:
    - If token_source == 'drfp_bits', topic words are built from LDA components_ as 'b{idx}'.
    - Otherwise, topic words are **derived from texts** using doc-topic weights (lda.transform),
      ensuring the topic tokens live in the same Dictionary as the texts (prevents mismatch errors).
    """
    if not globals().get('HAVE_GENSIM', False):
        logging.info("Gensim not available; skipping LDA coherence.")
        return

    try:
        # 1) Build texts
        texts = _get_coherence_texts(drfp_df, config)
        if not texts or all((len(t) == 0 or (len(t) == 1 and t[0] == "_empty_")) for t in texts):
            logging.info("Coherence skipped: empty texts.")
            return

        # 2) Dictionary & filtering
        dictionary = Dictionary(texts)
        min_count = max(1, int(config.get('lda_coherence_min_count', 2)))
        max_df = float(config.get('lda_coherence_max_df', 0.95))
        if min_count > 1 or max_df < 1.0:
            dictionary.filter_extremes(no_below=min_count, no_above=max_df)
        if len(dictionary) == 0:
            logging.info("Coherence skipped: dictionary empty after filtering.")
            return

        # 3) Build BoW corpus from texts (used by UMass and by the text-derived topic tokens)
        corpus = [dictionary.doc2bow(t) for t in texts]

        # 4) Topic token lists
        comps = getattr(lda_model, 'components_', None)
        if comps is None:
            logging.warning("LDA model has no components_; cannot compute coherence.")
            return

        token_source = str(config.get('lda_coherence_token_source', 'drfp_bits')).lower()

        # Bound topn appropriately
        if token_source == 'drfp_bits':
            n_features = comps.shape[1] if hasattr(comps, "shape") else len(comps[0])
            topn = max(1, min(int(config.get('lda_coherence_topn', 20)), int(n_features)))
        else:
            # For text-derived path we’ll cap later by available scored tokens
            topn = max(1, int(config.get('lda_coherence_topn', 20)))

        if token_source == 'drfp_bits':
            # (A) components_ -> top-N bit indices -> 'b{idx}'
            topic_tokens = [[f"b{int(i)}" for i in np.argsort(row)[-topn:][::-1]] for row in comps]
        else:
            # (B) derive topic tokens from texts using doc-topic weights
            try:
                Theta = lda_model.transform(drfp_df)  # shape: (n_docs, n_topics)
            except Exception as e:
                logging.warning("Could not compute lda.transform(..) to build text-based topic tokens: %s", e)
                return

            from collections import defaultdict

            n_docs_theta = int(Theta.shape[0])
            n_docs_texts = len(corpus)
            L = min(n_docs_theta, n_docs_texts)
            if n_docs_theta != n_docs_texts:
                logging.warning(
                    "Coherence token-source doc-count mismatch: texts=%d, lda.transform=%d; truncating to %d.",
                    n_docs_texts, n_docs_theta, L
                )

            K = int(Theta.shape[1])
            topic_tokens: List[List[str]] = []
            for k in range(K):
                scores = defaultdict(float)
                for d in range(L):
                    w = float(Theta[d, k])
                    if w <= 0.0:
                        continue
                    # weight each token by its count in doc * topic weight
                    for token_id, cnt in corpus[d]:
                        scores[token_id] += w * float(cnt)
                if not scores:
                    topic_tokens.append(["_empty_"])
                    continue
                top_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
                topic_tokens.append([dictionary[int(tid)] for tid, _ in top_ids])

        # 5) Coherence model setup
        metric = str(config.get('lda_coherence_metric', 'c_v')).lower().replace('-', '_')
        if metric == 'umass': metric = 'u_mass'
        if metric == 'npmi' : metric = 'c_npmi'
        processes   = max(1, int(config.get('lda_coherence_processes', 1)))
        window_size = int(config.get('lda_coherence_window_size', 110))

        cm_kwargs = dict(topics=topic_tokens, coherence=metric)
        if metric == 'u_mass':
            if not any(len(b) > 0 for b in corpus):
                logging.info("Coherence skipped: empty corpus for u_mass.")
                return
            cm_kwargs.update(dictionary=dictionary, corpus=corpus, processes=processes)
            metric_tag = 'u_mass'
        elif metric == 'c_npmi':
            cm_kwargs.update(dictionary=dictionary, texts=texts, processes=processes, window_size=window_size)
            metric_tag = 'c_npmi'
        else:  # c_v
            cm_kwargs.update(dictionary=dictionary, texts=texts, processes=processes, window_size=window_size)
            metric_tag = 'c_v'

        cm = CoherenceModel(**cm_kwargs)
        per_topic = cm.get_coherence_per_topic()
        overall   = float(cm.get_coherence())

        # 6) Outputs
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "metric": metric_tag,
            "mean_coherence": float(overall),
            "per_topic": [float(x) for x in per_topic],
            "topn": int(topn),
            "window_size": int(window_size),
            "n_docs": int(len(texts)),
            "n_tokens": int(len(dictionary)),
            "token_source": token_source,
        }
        with open(os.path.join(out_dir, f"lda_coherence_{metric_tag}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        import pandas as _pd
        _pd.DataFrame({"topic": list(range(len(per_topic))), "coherence": [float(x) for x in per_topic]})\
           .to_csv(os.path.join(out_dir, f"lda_coherence_{metric_tag}_per_topic.csv"), index=False)

        logging.info("LDA %s coherence (token_source=%s): mean=%.4f",
                     metric_tag, meta["token_source"], overall)

    except Exception as e:
        logging.warning("LDA coherence computation failed: %s", e)


# =============================================================================
#                      HYPERPARAMETER RECOMMENDER (OFFLINE CV)
# =============================================================================

def recommend_xgb_hyperparams(
    X_full: pd.DataFrame,
    Y: pd.DataFrame,
    topics: List[int],
    features: List[str],
    cfg: Dict[str, Any],
    features_by_family: Optional[Dict[int, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Offline hyperparameter recommender for the specialist XGB models, tuned *per target*.

    Output JSON structure (high level)
    ---------------------------------
    - global.targets[target_name].params
    - regimes[regime_label].targets[target_name].params   (if size_regime)
    - families[family_id].targets[target_name].params

    IMPORTANT SCIENTIFIC NOTE
    -------------------------
    This function is intentionally *not* used to "compare the final model to held-out data".
    Instead, it uses CV only to recommend hyperparameters, after which the main training run
    is expected to fit ONCE on the full clustered network (your stated modeling philosophy).

    That said, the CV procedure here is still useful because it searches for XGB settings
    that yield robust fits under resampling on the training network, without changing your
    fundamental final-model approach.
    """
   
    # Targets
    target_cols = _get_target_cols(cfg, Y, required=True)
    # Ensure these columns exist
    for c in target_cols:
        if c not in Y.columns:
            raise ValueError(f"Target column '{c}' not found in Y. Available: {list(Y.columns)}")

    # Config
    base_params = copy.deepcopy(cfg.get('xgb_params', DEFAULT_CONFIG['xgb_params']))
    if 'random_state' not in base_params:
        base_params['random_state'] = int(cfg.get('random_state', 42))

    param_grid_cfg = cfg.get('hp_recommender_param_grid', None) or DEFAULT_CONFIG.get('hp_recommender_param_grid', {})
    if not isinstance(param_grid_cfg, dict) or not param_grid_cfg:
        raise ValueError("hp_recommender_param_grid must be a non-empty dict.")

    mode = str(cfg.get('hp_recommender_mode', 'size_regime')).strip().lower()
    n_folds = int(cfg.get('hp_recommender_n_folds', 5))
    min_fold_size = int(cfg.get('hp_recommender_min_fold_size', 20))
    min_family_size = int(cfg.get('hp_recommender_min_family_size', 30))
    use_robust_scaler = cfg.get('hp_recommender_use_robust_scaler', cfg.get('use_robust_scaler', True))

    # If per-family feature sets are used, size-regime pooling becomes ambiguous → force per-family mode.
    if features_by_family is not None and mode == 'size_regime':
        mode = 'per_family'

    size_bins = cfg.get('hp_recommender_size_bins', [25, 50, 100, 200, 500])

    # Scoring for GridSearchCV: choose objective via cfg['hp_recommender_scoring'].
    # GridSearchCV always *maximizes* the score. For error metrics (RMSE/MAE), we therefore use negative scorers.
    scoring_name = str(cfg.get('hp_recommender_scoring', 'neg_rmse')).strip().lower()
    _aliases = {
        # RMSE variants
        'rmse': 'neg_rmse',
        'neg_rmse': 'neg_rmse',
        'neg_mean_rmse': 'neg_rmse',
        'neg_root_mean_squared_error': 'neg_rmse',
        # MAE variants
        'mae': 'neg_mae',
        'neg_mae': 'neg_mae',
        'neg_mean_absolute_error': 'neg_mae',
        # R2 variants
        'r2': 'r2',
    }
    scoring_effective = _aliases.get(scoring_name, scoring_name)
    if scoring_effective == 'neg_rmse':
        scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
        scoring_metric = 'rmse'
        scoring_greater_is_better = False
    elif scoring_effective == 'neg_mae':
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        scoring_metric = 'mae'
        scoring_greater_is_better = False
    elif scoring_effective == 'r2':
        scorer = make_scorer(r2_score, greater_is_better=True)
        scoring_metric = 'r2'
        scoring_greater_is_better = True
    else:
        raise ValueError(
            f"Unknown hp_recommender_scoring='{scoring_name}'. Supported: rmse|neg_rmse, mae|neg_mae, r2, neg_r2."
        )

    def _make_pipeline(xgb_params: Dict[str, Any]):
        scaler = RobustScaler() if bool(use_robust_scaler) else StandardScaler()
        return Pipeline([('scaler', scaler), ('model', XGBRegressor(**xgb_params))])

    # Grid for Pipeline model step
    grid = {f"model__{k}": v for k, v in param_grid_cfg.items()}

    def _choose_n_splits(n: int) -> int:
        if n <= 0:
            return 0
        max_splits = n // max(1, min_fold_size)
        return int(min(n_folds, max(2, max_splits)))

    def _tune_single_target(X_block: pd.DataFrame, y_vec: np.ndarray, label: str, tname: str) -> Dict[str, Any]:
        Xb = X_block.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        yb = np.asarray(y_vec).ravel()

        n = int(len(Xb))
        n_splits = _choose_n_splits(n)
        if n_splits < 2:
            return {
                "status": "skipped_small",
                "n_samples": n,
                "cv_best_score": None,
                "params": copy.deepcopy(base_params),
                "cv_best_metric": None,
                "metric": scoring_metric,
                "score_name": scoring_effective,
            }

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.get('random_state', 42)))
        pipe = _make_pipeline(copy.deepcopy(base_params))

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scorer,
            cv=cv,
            n_jobs=int(cfg.get('hp_recommender_n_jobs', -1)),
            refit=True,
            verbose=0,
        )
        try:
            gs.fit(Xb, yb)
            # Convert best_params_ back to plain XGB params
            best_overrides = {k.replace('model__', ''): v for k, v in (gs.best_params_ or {}).items()}
            tuned = copy.deepcopy(base_params)
            tuned.update(best_overrides)
            return {
                "status": "tuned",
                "n_samples": n,
                "cv_best_score": float(gs.best_score_) if gs.best_score_ is not None else None,
                "cv_best_metric": (
                    (-float(gs.best_score_) if gs.best_score_ is not None else None)
                    if scoring_effective in ('neg_rmse', 'neg_mae', 'r2')
                    else (float(gs.best_score_) if gs.best_score_ is not None else None)
                ),
                "metric": scoring_metric,
                "score_name": scoring_effective,
                "params": tuned,
            }
        except Exception as e:
            logging.warning("Tuning failed for %s / %s: %s", label, tname, e)
            return {
                "status": "failed",
                "n_samples": n,
                "cv_best_score": None,
                "params": copy.deepcopy(base_params),
                "cv_best_metric": None,
                "metric": scoring_metric,
                "score_name": scoring_effective,
            }

    def _tune_block(X_block: pd.DataFrame, Y_block: pd.DataFrame, label: str) -> Dict[str, Any]:
        out_targets = {}
        scores = []
        metrics = []
        statuses = []
        for tname in target_cols:
            res_t = _tune_single_target(X_block, Y_block[tname].values, label=label, tname=tname)
            out_targets[tname] = {
                "params": res_t["params"],
                "cv_best_score": res_t.get("cv_best_score"),
                "cv_best_metric": res_t.get("cv_best_metric"),
                "metric": res_t.get("metric"),
                "score_name": res_t.get("score_name"),
                "status": res_t.get("status"),
                "n_samples": res_t.get("n_samples"),
            }
            if res_t.get("cv_best_score") is not None:
                scores.append(float(res_t["cv_best_score"]))
            if res_t.get("cv_best_metric") is not None:
                metrics.append(float(res_t["cv_best_metric"]))
            statuses.append(str(res_t.get("status")))

        agg = float(np.mean(scores)) if scores else None
        agg_metric = float(np.mean(metrics)) if metrics else None
        # Status: tuned if any target tuned; skipped_small if all skipped; else mixed/failed.
        if all(s == 'skipped_small' for s in statuses):
            status = 'skipped_small'
        elif any(s == 'tuned' for s in statuses):
            status = 'tuned'
        elif any(s == 'failed' for s in statuses):
            status = 'failed'
        else:
            status = 'unknown'

        return {
            "status": status,
            "n_samples": int(len(X_block)),
            "aggregate_cv_best_score": agg,
            "aggregate_cv_best_metric": agg_metric,
            "targets": out_targets,
        }

    # ---- GLOBAL tuning on full network
    global_block = _tune_block(X_full[features], Y, label='GLOBAL')
    global_targets = global_block.get('targets', {}) or {}

    # Helper: build a global-by-target mapping indexed by output position
    # (Used in training consumption)
    global_by_index = {}
    for i, tname in enumerate(target_cols):
        tinfo = global_targets.get(tname, {})
        if isinstance(tinfo, dict) and isinstance(tinfo.get('params'), dict):
            global_by_index[i] = dict(tinfo['params'])

    fam_ids = sorted(list(set(int(x) for x in topics)))
    topics_s = np.asarray([int(x) for x in topics], dtype=int)

    regimes_out: Dict[str, Any] = {}
    families_out: Dict[str, Any] = {}

    # ---- SIZE-REGIME mode
    def _size_label(n: int) -> str:
        for b in size_bins:
            if n < int(b):
                lo = 0
                prev = 0
                for bb in size_bins:
                    if int(bb) == int(b):
                        lo = prev
                        break
                    prev = int(bb)
                return f"{lo}-{int(b)-1}"
        return f"{int(size_bins[-1])}+" if size_bins else "all"

    if mode == 'size_regime':
        # Group families by size bin
        fam_sizes = {int(f): int((topics_s == int(f)).sum()) for f in fam_ids}
        bins = {}
        for f, n in fam_sizes.items():
            lbl = _size_label(n)
            bins.setdefault(lbl, []).append(f)

        # Tune pooled data per bin (only families with sufficient size)
        for lbl, fam_list in bins.items():
            idx = np.isin(topics_s, np.asarray([int(x) for x in fam_list], dtype=int))
            n_pool = int(idx.sum())
            # only tune if there is enough pooled data AND at least one family >= min_family_size
            if n_pool < max(min_family_size, min_fold_size * 2):
                regimes_out[lbl] = {
                    "status": "skipped_small",
                    "n_samples": n_pool,
                    "aggregate_cv_best_score": None,
                    "targets": {t: {"params": copy.deepcopy(global_targets.get(t, {}).get('params', base_params)),
                                     "cv_best_score": None,
                                     "status": "fallback_global",
                                     "n_samples": n_pool} for t in target_cols},
                }
                continue

            # Build pooled block
            X_pool = X_full.loc[idx, features]
            Y_pool = Y.loc[idx, target_cols]
            block = _tune_block(X_pool, Y_pool, label=f"REGIME[{lbl}]")
            regimes_out[lbl] = block

        # Assign each family to a regime's target params (fallback to global)
        for fam in fam_ids:
            idx_f = topics_s == int(fam)
            n_f = int(idx_f.sum())
            lbl = _size_label(n_f)

            if n_f < min_family_size:
                families_out[str(int(fam))] = {
                    "n_samples": n_f,
                    "regime": lbl,
                    "status": "fallback_global",
                    "targets": copy.deepcopy(global_targets),
                }
                continue

            reg = regimes_out.get(lbl, None)
            if isinstance(reg, dict) and isinstance(reg.get('targets'), dict) and reg.get('status') in ('tuned', 'unknown', 'failed'):
                # If the regime tuning failed, still use its targets (they might contain fallbacks),
                # but mark status accordingly.
                fam_targets = reg.get('targets', {}) or {}
                st = 'tuned_regime' if reg.get('status') == 'tuned' else 'fallback_global'
                # If some targets were tuned in the regime, use them; otherwise fallback
                if any(isinstance(fam_targets.get(t, {}), dict) and fam_targets[t].get('status') == 'tuned' for t in target_cols):
                    st = 'tuned_regime'
                families_out[str(int(fam))] = {
                    "n_samples": n_f,
                    "regime": lbl,
                    "status": st,
                    "targets": copy.deepcopy(fam_targets) if fam_targets else copy.deepcopy(global_targets),
                }
            else:
                families_out[str(int(fam))] = {
                    "n_samples": n_f,
                    "regime": lbl,
                    "status": "fallback_global",
                    "targets": copy.deepcopy(global_targets),
                }

    else:
        # ---- PER-FAMILY mode
        for fam in fam_ids:
            idx_f = topics_s == int(fam)
            n_f = int(idx_f.sum())
            cols_f = (features_by_family.get(int(fam), features) if features_by_family else features)

            if n_f < min_family_size:
                families_out[str(int(fam))] = {
                    "n_samples": n_f,
                    "regime": None,
                    "status": "fallback_global",
                    "targets": copy.deepcopy(global_targets),
                }
                continue

            block = _tune_block(X_full.loc[idx_f, cols_f], Y.loc[idx_f, target_cols], label=f"FAMILY[{int(fam)}]")
            families_out[str(int(fam))] = {
                "n_samples": n_f,
                "regime": None,
                "status": "tuned_family" if block.get('status') == 'tuned' else "fallback_global",
                "targets": copy.deepcopy(block.get('targets', {})) if isinstance(block.get('targets'), dict) else copy.deepcopy(global_targets),
            }

    out = {
        "meta": {
            "mode": mode,
            "target_cols": target_cols,
            "n_folds": n_folds,
            "min_fold_size": min_fold_size,
            "min_family_size": min_family_size,
            "use_robust_scaler": bool(use_robust_scaler),
            "param_grid": copy.deepcopy(param_grid_cfg),
            "scoring": {
                "requested": scoring_name,
                "effective": scoring_effective,
                "metric": scoring_metric,
                "greater_is_better": bool(scoring_greater_is_better),
            },
            "features_common": list(features),
        },
        "global": {
            "status": global_block.get("status", "unknown"),
            "n_samples": int(global_block.get("n_samples", 0)),
            "aggregate_cv_best_score": global_block.get("aggregate_cv_best_score", None),
            "aggregate_cv_best_metric": global_block.get("aggregate_cv_best_metric", None),
            "targets": copy.deepcopy(global_targets),
            "by_target_index": global_by_index,
        },
        "regimes": regimes_out,
        "families": families_out,
    }
    return out


def write_hyperparams_artifacts(reco: Dict[str, Any], out_json_path: str, out_csv_path: Optional[str] = None) -> None:
    """Write the recommender JSON plus a compact CSV summary (optional)."""

    _ensure_dir(os.path.dirname(out_json_path))
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(reco, f, indent=2)

    if out_csv_path:
        rows = []
        fam = reco.get("families", {}) or {}
        meta = reco.get("meta", {}) or {}
        target_cols = list(meta.get("target_cols", DEFAULT_TARGET_COLS))

        for fam_id, info in fam.items():
            tblock = (info or {}).get("targets", {}) or {}
            for tname in target_cols:
                tinfo = tblock.get(tname, {}) if isinstance(tblock, dict) else {}
                rows.append({
                    "family": fam_id,
                    "target": tname,
                    "n_samples": info.get("n_samples"),
                    "regime": info.get("regime"),
                    "status": info.get("status"),
                    "cv_best_score": tinfo.get("cv_best_score"),
                    "target_status": tinfo.get("status"),
                    "params_json": json.dumps(tinfo.get("params", {}), ensure_ascii=False),
                })

        df = pd.DataFrame(rows)
        _ensure_dir(os.path.dirname(out_csv_path))
        df.to_csv(out_csv_path, index=False)
        
# =============================================================================
#                    REVIEWER-RESPONSE DIAGNOSTICS (v2.5.19)
# =============================================================================

def _safe_label_series(df: pd.DataFrame, requested_col: str) -> Optional[pd.Series]:
    """
    Return the reference mechanistic/RMG family label column if available.
    Falls back to common alternative names used in reaction-mechanism tables.
    """
    candidates = [
        requested_col,
        'rmg_family',
        'RMG_family',
        'reaction_family',
        'ReactionFamily',
        'mechanism_label',
        'mechanistic_label',
        'family_rmg',
        'family_RMG',
        'RMG_reaction_family',
        'Reaction_Family',
    ]
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if c in df.columns:
            ser = df[c].copy()
            ser = ser.where(~pd.isna(ser), other='Unknown')
            return ser.astype(str)
    return None


def _classification_entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy for one row of class-count values."""
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _evaluate_prediction_table(
    Y_true: pd.DataFrame,
    preds: pd.DataFrame,
    label: str,
    n_features: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute the XTQSPRKinetic metric set for all configured target columns.

    This intentionally reuses the pipeline's own helper functions (_adj_r2,
    _nrmse, _aard, _rae) so diagnostics are numerically consistent
    with global_stats.csv and family_stats.csv.
    """
    rows = []
    p_features = int(n_features or 0)

    for out_name in _target_cols_from_frame(Y_true):
        pred_col = f"{out_name}_pred"
        if out_name not in Y_true.columns or pred_col not in preds.columns:
            rows.append({
                'model': label,
                'output': out_name,
                'R2': np.nan,
                'AdjR2': np.nan,
                'MAE': np.nan,
                'RMSE': np.nan,
                'nRMSE': np.nan,
                'AARD': np.nan,
                'RAE': np.nan,
                'n_samples': 0,
                'n_features': p_features,
                'status': 'missing columns',
            })
            continue

        y = pd.Series(Y_true[out_name]).astype(float)
        yp = pd.Series(preds[pred_col], index=Y_true.index).astype(float)
        valid = np.isfinite(y.values) & np.isfinite(yp.values)
        yv = y.values[valid]
        ypv = yp.values[valid]
        n_valid = int(valid.sum())

        if n_valid == 0:
            r2 = adj_r2 = mae = rmse = nrmse = aard = rae = np.nan
        else:
            try:
                r2 = _safe_r2_score(yv, ypv)
            except Exception:
                r2 = np.nan
            try:
                mae = float(mean_absolute_error(yv, ypv))
            except Exception:
                mae = np.nan
            try:
                rmse = float(math.sqrt(mean_squared_error(yv, ypv)))
            except Exception:
                rmse = np.nan
            try:
                adj_r2 = float(_adj_r2(r2, n_valid, p_features))
            except Exception:
                adj_r2 = np.nan
            try:
                nrmse = float(_nrmse(yv, ypv))
            except Exception:
                nrmse = np.nan
            try:
                aard = float(_aard(yv, ypv))
            except Exception:
                aard = np.nan
            try:
                rae = float(_rae(yv, ypv))
            except Exception:
                rae = np.nan

        rows.append({
            'model': label,
            'output': out_name,
            'R2': r2,
            'AdjR2': adj_r2,
            'MAE': mae,
            'RMSE': rmse,
            'nRMSE': nrmse,
            'AARD': aard,
            'RAE': rae,
            'n_samples': n_valid,
            'n_features': p_features,
            'status': 'ok' if n_valid else 'no finite paired values',
        })

    return pd.DataFrame(rows)


def _write_dataset_summary(
    df: pd.DataFrame,
    Y: pd.DataFrame,
    out_dir: str,
    rmg_family_col: str,
) -> None:
    """Write concise data-summary tables."""
    summary_dir = os.path.join(out_dir, "diagnostics")
    _ensure_dir(summary_dir)

    target_summary = Y[_target_cols_from_frame(Y)].describe().T
    target_summary.to_csv(os.path.join(summary_dir, "dataset_target_summary.csv"))

    if 'family' in df.columns:
        df['family'].value_counts().sort_index().rename_axis('learned_family') \
            .reset_index(name='count') \
            .to_csv(os.path.join(summary_dir, "learned_family_counts.csv"), index=False)

    rmg = _safe_label_series(df, rmg_family_col)
    if rmg is not None:
        rmg.value_counts().rename_axis('rmg_family') \
            .reset_index(name='count') \
            .to_csv(os.path.join(summary_dir, "rmg_family_counts.csv"), index=False)

    meta = {
        'n_reactions': int(len(df)),
        'has_rmg_family_labels': bool(rmg is not None),
        'rmg_family_column_used': None if rmg is None else rmg.name,
        'target_columns': _target_cols_from_frame(Y),
        'interpretation': (
            'Network-level diagnostics for an RMG-derived reaction mechanism; '
            'not random reaction-level external generalization metrics.'
        ),
    }
    with open(os.path.join(summary_dir, "dataset_summary_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _write_family_mechanism_alignment(
    df: pd.DataFrame,
    out_dir: str,
    rmg_family_col: str,
) -> None:
    """
    Compare learned XTQSPRKinetic families with original RMG/mechanistic labels.
    This is a mechanistic diagnostic and is not used during unsupervised gating.
    """
    summary_dir = os.path.join(out_dir, "diagnostics")
    _ensure_dir(summary_dir)

    if 'family' not in df.columns:
        logging.warning("Alignment skipped: df has no active 'family' column.")
        return

    rmg = _safe_label_series(df, rmg_family_col)
    if rmg is None:
        logging.warning(
            "Alignment skipped: no RMG/mechanistic family column found. "
            "Set config['rmg_family_col'] to the correct column name."
        )
        pd.DataFrame([{
            'status': 'skipped',
            'reason': 'No RMG/mechanistic family column found',
            'requested_column': rmg_family_col,
        }]).to_csv(os.path.join(summary_dir, "family_rmg_alignment_SKIPPED.csv"), index=False)
        return

    learned = df['family'].where(~pd.isna(df['family']), other='Unknown').astype(str)

    contingency = pd.crosstab(
        learned,
        rmg,
        rownames=['learned_family'],
        colnames=['rmg_family']
    )
    contingency.to_csv(os.path.join(summary_dir, "family_rmg_contingency_matrix.csv"))

    learned_codes = pd.Categorical(learned).codes
    rmg_codes = pd.Categorical(rmg).codes

    total = contingency.values.sum()
    purity = float(contingency.max(axis=1).sum() / total) if total > 0 else np.nan

    try:
        h, c, v = homogeneity_completeness_v_measure(rmg_codes, learned_codes)
    except Exception:
        h = c = v = np.nan

    metrics = {
        'n_reactions': int(len(df)),
        'n_learned_families': int(pd.Series(learned).nunique()),
        'n_rmg_families': int(pd.Series(rmg).nunique()),
        'purity': purity,
        'adjusted_rand_index': float(adjusted_rand_score(rmg_codes, learned_codes)),
        'normalized_mutual_information': float(normalized_mutual_info_score(rmg_codes, learned_codes)),
        'adjusted_mutual_information': float(adjusted_mutual_info_score(rmg_codes, learned_codes)),
        'homogeneity': float(h),
        'completeness': float(c),
        'v_measure': float(v_measure_score(rmg_codes, learned_codes)),
        'rmg_family_column_used': rmg.name,
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(summary_dir, "family_rmg_alignment_metrics.csv"),
        index=False
    )
    with open(os.path.join(summary_dir, "family_rmg_alignment_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rows = []
    for fam, row in contingency.iterrows():
        n_fam = int(row.sum())
        dominant_label = str(row.idxmax()) if n_fam > 0 else None
        dominant_count = int(row.max()) if n_fam > 0 else 0
        entropy = _classification_entropy_from_counts(row.values)

        rows.append({
            'learned_family': fam,
            'n_reactions': n_fam,
            'dominant_rmg_family': dominant_label,
            'dominant_rmg_count': dominant_count,
            'purity_within_family': float(dominant_count / n_fam) if n_fam > 0 else np.nan,
            'rmg_label_entropy': entropy,
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(summary_dir, "family_rmg_enrichment_by_family.csv"),
        index=False
    )


def _write_stability_summary(
    df: pd.DataFrame,
    stability_df: Optional[pd.DataFrame],
    out_dir: str,
    rmg_family_col: str,
) -> None:
    """Summarize bootstrap stability metrics by learned and RMG family."""
    if stability_df is None or not isinstance(stability_df, pd.DataFrame) or stability_df.empty:
        logging.info("Stability summary skipped: no bootstrap stability_df available.")
        return

    summary_dir = os.path.join(out_dir, "diagnostics")
    _ensure_dir(summary_dir)

    stab = stability_df.copy()
    try:
        stab = stab.reindex(df.index)
    except Exception:
        stab.index = df.index[:len(stab)]
        stab = stab.reindex(df.index)

    cols = [c for c in ['modal_vote', 'norm_entropy', 'entropy'] if c in stab.columns]
    if not cols:
        logging.info("Stability summary skipped: expected stability columns absent.")
        return

    tmp = pd.DataFrame(index=df.index)
    tmp['family'] = df['family'].values if 'family' in df.columns else 'Unknown'
    for c in cols:
        tmp[c] = stab[c].values

    by_learned = tmp.groupby('family')[cols].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
    by_learned.to_csv(os.path.join(summary_dir, "bootstrap_stability_by_learned_family.csv"))

    rmg = _safe_label_series(df, rmg_family_col)
    if rmg is not None:
        tmp['rmg_family'] = rmg.values
        by_rmg = tmp.groupby('rmg_family')[cols].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
        by_rmg.to_csv(os.path.join(summary_dir, "bootstrap_stability_by_rmg_family.csv"))

    stab[cols].describe().T.to_csv(
        os.path.join(summary_dir, "bootstrap_stability_overall_summary.csv")
    )


def _write_residual_error_tables(
    df: pd.DataFrame,
    Y: pd.DataFrame,
    preds: pd.DataFrame,
    out_dir: str,
    rmg_family_col: str,
    stability_df: Optional[pd.DataFrame],
    top_n: int = 25,
) -> None:
    """Write residual/failure-mode tables."""
    summary_dir = os.path.join(out_dir, "diagnostics")
    _ensure_dir(summary_dir)

    rmg = _safe_label_series(df, rmg_family_col)

    base_cols = ['reaction_smiles', 'family']
    out = df[[c for c in base_cols if c in df.columns]].copy()
    if rmg is not None:
        out['rmg_family'] = rmg.values

    if stability_df is not None and isinstance(stability_df, pd.DataFrame):
        stab = stability_df.copy()
        try:
            stab = stab.reindex(df.index)
        except Exception:
            stab.index = df.index[:len(stab)]
            stab = stab.reindex(df.index)
        for c in ['modal_vote', 'norm_entropy', 'entropy']:
            if c in stab.columns:
                out[c] = stab[c].values

    for target in _target_cols_from_frame(Y):
        out[f'{target}_true'] = Y[target].values
        out[f'{target}_pred'] = preds[f'{target}_pred'].values
        out[f'{target}_residual'] = out[f'{target}_pred'] - out[f'{target}_true']
        out[f'{target}_abs_error'] = np.abs(out[f'{target}_residual'])

    out.to_csv(os.path.join(summary_dir, "residuals_all_reactions.csv"), index=False)

    for target in _target_cols_from_frame(Y):
        token = _safe_target_token(target)
        top = out.sort_values(f'{target}_abs_error', ascending=False).head(int(top_n))
        top.to_csv(
            os.path.join(summary_dir, f"top_{int(top_n)}_residuals_{token}.csv"),
            index=False
        )

    agg_cols = {f'{target}_abs_error': ['count', 'mean', 'median', 'max'] for target in _target_cols_from_frame(Y)}

    if 'family' in out.columns:
        out.groupby('family').agg(agg_cols).to_csv(
            os.path.join(summary_dir, "residual_summary_by_learned_family.csv")
        )

    if 'rmg_family' in out.columns:
        out.groupby('rmg_family').agg(agg_cols).to_csv(
            os.path.join(summary_dir, "residual_summary_by_rmg_family.csv")
        )

def _predict_moe_with_configured_routing(
    models: Dict[Any, Any],
    X_full: pd.DataFrame,
    hard_labels: Sequence[int],
    features: List[str],
    scalers: Dict[Any, Any],
    config: Dict[str, Any],
    *,
    topic_proba_df: Optional[pd.DataFrame] = None,
    stability_df: Optional[pd.DataFrame] = None,
    routing_mode_actual: Optional[str] = None,
    hard_topic_series: Optional[Union[pd.Series, Sequence[int]]] = None,
    target_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Use the same hard / soft / hybrid routing logic for diagnostic ablation rows.
    """

    mode = str(
        routing_mode_actual or (
            'hybrid'
            if bool(config.get('use_hybrid_routing', False))
            else ('soft' if bool(config.get('use_soft_routing', True)) else 'hard')
        )
    ).strip().lower()

    if (
        mode == 'hybrid'
        and isinstance(topic_proba_df, pd.DataFrame)
        and isinstance(stability_df, pd.DataFrame)
        and not topic_proba_df.empty
        and not stability_df.empty
    ):
        return predict_arrhenius_hybrid(
            models,
            X_full,
            topic_proba_df,
            stability_df,
            features,
            scalers,
            bool(config.get('use_family_scaler', True)),
            metric=str(config.get('hybrid_stability_metric', 'entropy')).lower(),
            threshold=float(config.get('hybrid_stability_threshold', 0.35)),
            features_by_family=None,
            target_cols=target_cols,
            hard_topic_series=hard_topic_series,
        )

    if (
        mode in {'soft', 'hybrid'}
        and bool(config.get('use_soft_routing', True))
        and isinstance(topic_proba_df, pd.DataFrame)
        and not topic_proba_df.empty
    ):
        return predict_arrhenius_weighted(
            models,
            X_full,
            topic_proba_df,
            features,
            scalers,
            bool(config.get('use_family_scaler', True)),
            features_by_family=None,
            target_cols=target_cols,
        )

    return predict_arrhenius(
        models,
        X_full,
        list(hard_labels),
        features,
        scalers,
        bool(config.get('use_family_scaler', True)),
        features_by_family=None,
        target_cols=target_cols,
    )

def _predict_global_model(
    models: Dict[Any, Any],
    scalers: Dict[Any, Any],
    X_full: pd.DataFrame,
    features: List[str],
    target_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Predict using only the global fallback model already trained by XTQSPRKinetic."""
    g_model = models.get('global')
    g_scaler = scalers.get('global')
    if g_model is None or g_scaler is None:
        raise RuntimeError("Global model or global scaler is missing.")

    target_cols = _resolve_prediction_target_cols(target_cols, models)
    Xg = X_full[features]
    Xg_scaled = g_scaler.transform(Xg)
    Xg_scaled_df = pd.DataFrame(Xg_scaled, columns=features, index=Xg.index)
    yhat = g_model.predict(Xg_scaled_df)

    return pd.DataFrame(
        yhat,
        index=X_full.index,
        columns=_prediction_columns(target_cols)
    )


def _run_random_cluster_controls(
    X_full: pd.DataFrame,
    Y: pd.DataFrame,
    df: pd.DataFrame,
    confirmed: List[str],
    config: Dict[str, Any],
    out_dir: str,
) -> None:
    """
    Random-cluster negative controls for diagnostic audit.

    diagnostics_random_controls_mode controls the null model:

    - 'size_matched':
        Randomly permutes the observed learned-family labels.
        This preserves the exact family-size distribution and destroys the
        chemical association between reactions and learned families.
        This is the recommended mode for manuscript/reviewer diagnostics.

    - 'empirical_label_sampling':
        Samples pseudo-family labels from the empirical learned-family
        probabilities. This does not preserve exact family sizes and should
        be treated as an exploratory sensitivity analysis.

    - 'off':
        Should normally be handled by the dispatcher before this function is
        called. If received here, the function returns without running.
    """
    summary_dir = os.path.join(out_dir, "diagnostics")
    _ensure_dir(summary_dir)

    if 'family' not in df.columns:
        logging.warning("Random-cluster controls skipped: df has no 'family' column.")
        return

    random_control_mode = str(
        config.get('diagnostics_random_controls_mode', 'size_matched')
    ).strip().lower()

    # Backward-compatible fallback if old keys are used and the new mode key
    # was not explicitly supplied.
    if 'diagnostics_random_controls_mode' not in config:
        preserve_sizes = bool(config.get('diagnostics_random_controls_preserve_sizes', True))
        random_control_mode = 'size_matched' if preserve_sizes else 'empirical_label_sampling'

    if random_control_mode in {'false', 'none', 'no', '0'}:
        random_control_mode = 'off'
    elif random_control_mode in {'true', 'yes', '1'}:
        random_control_mode = 'size_matched'

    valid_modes = {'off', 'size_matched', 'empirical_label_sampling'}
    if random_control_mode not in valid_modes:
        raise ValueError(
            "diagnostics_random_controls_mode must be one of "
            f"{sorted(valid_modes)}, got '{random_control_mode}'."
        )

    if random_control_mode == 'off':
        logging.info("Random-cluster controls skipped inside function: mode='off'.")
        return

    n_controls = int(config.get('diagnostics_random_controls_n', 25))
    rng = np.random.RandomState(int(config.get('random_state', 42)))

    observed_labels = np.asarray(df['family'].tolist())
    unique_labels, label_counts = np.unique(observed_labels, return_counts=True)
    label_probs = label_counts / label_counts.sum()

    rows = []
    xgb_params_base = copy.deepcopy(config.get('xgb_params', DEFAULT_CONFIG['xgb_params']))

    logging.info(
        "Running random-cluster controls: mode=%s, n_controls=%d.",
        random_control_mode,
        n_controls,
    )

    for b in range(n_controls):
        if random_control_mode == 'size_matched':
            random_labels = rng.permutation(observed_labels)
            control_label = f'random_size_matched_{b}'
        else:
            random_labels = rng.choice(
                unique_labels,
                size=len(observed_labels),
                replace=True,
                p=label_probs,
            )
            control_label = f'random_empirical_label_sampling_{b}'

        try:
            ctrl_models, ctrl_scalers = train_topic_models(
                X_full,
                Y,
                random_labels.tolist(),
                confirmed,
                random_state=int(config.get('random_state', 42)) + b + 1,
                use_family_scaler=bool(config.get('use_family_scaler', True)),
                family_min_size=int(config.get('family_min_size', 5)),
                xgb_params=xgb_params_base,
                use_robust_scaler=bool(config.get('use_robust_scaler', True)),
                target_cols=_get_target_cols(config, Y, required=False),
            )

            ctrl_preds = predict_arrhenius(
                ctrl_models,
                X_full,
                random_labels.tolist(),
                confirmed,
                ctrl_scalers,
                bool(config.get('use_family_scaler', True)),
                target_cols=_get_target_cols(config, Y, required=False),
            )

            m = _evaluate_prediction_table(
                Y,
                ctrl_preds,
                label=control_label,
                n_features=len(confirmed),
            )
            m['random_control_id'] = b
            m['random_control_mode'] = random_control_mode
            m['preserve_family_sizes'] = (random_control_mode == 'size_matched')
            rows.append(m)

        except Exception as e:
            logging.warning("Random-cluster control %d failed: %s", b, e)

    if rows:
        all_ctrl = pd.concat(rows, ignore_index=True)
        all_ctrl.to_csv(
            os.path.join(summary_dir, "random_cluster_controls_all_runs.csv"),
            index=False
        )

        summary = all_ctrl.groupby('output')[[
            'R2', 'MAE', 'RMSE', 'nRMSE', 'AARD', 'RAE'
        ]].agg(['mean', 'std', 'min', 'max'])

        summary.to_csv(
            os.path.join(summary_dir, "random_cluster_controls_summary.csv")
        )

        meta = {
            'random_control_mode': random_control_mode,
            'n_controls_requested': int(n_controls),
            'n_controls_completed': int(all_ctrl['random_control_id'].nunique()),
            'preserve_family_sizes': bool(random_control_mode == 'size_matched'),
            'output_files': [
                'random_cluster_controls_all_runs.csv',
                'random_cluster_controls_summary.csv',
                'random_cluster_controls_meta.json',
            ],
        }
        with open(
            os.path.join(summary_dir, "random_cluster_controls_meta.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2)
    else:
        logging.warning("Random-cluster controls produced no successful runs.")

def _run_component_ablation_diagnostics(
    X_full: pd.DataFrame,
    Y: pd.DataFrame,
    df: pd.DataFrame,
    preds_current: pd.DataFrame,
    models: Dict[Any, Any],
    scalers: Dict[Any, Any],
    confirmed: List[str],
    config: Dict[str, Any],
    out_dir: str,
    topic_proba_df: Optional[pd.DataFrame] = None,
    stability_df: Optional[pd.DataFrame] = None,
    routing_mode_actual: Optional[str] = None,
) -> None:
    """
    Component-wise diagnostics:
    current MoE, global-only model, RMG-label experts, and optional all-feature MoE.
    """
    summary_dir = os.path.join(out_dir, "diagnostics")
    _ensure_dir(summary_dir)

    rows = []

    rows.append(_evaluate_prediction_table(
        Y,
        preds_current,
        label='proposed_MoE_current_routing',
        n_features=len(confirmed),
    ))

    try:
        gpred = _predict_global_model(models, scalers, X_full, confirmed, target_cols=_get_target_cols(config, Y, required=False))
        rows.append(_evaluate_prediction_table(
            Y,
            gpred,
            label='global_XGBoost_confirmed_features',
            n_features=len(confirmed),
        ))
    except Exception as e:
        logging.warning("Global-model ablation failed: %s", e)

    if bool(config.get('diagnostics_ablation_include_rmg_label_experts', True)):
        rmg = _safe_label_series(df, str(config.get('rmg_family_col', 'rmg_family')))
        if rmg is not None:
            try:
                rmg_codes = pd.Categorical(rmg).codes.tolist()
                xgb_params_base = copy.deepcopy(config.get('xgb_params', DEFAULT_CONFIG['xgb_params']))

                rmg_models, rmg_scalers = train_topic_models(
                    X_full,
                    Y,
                    rmg_codes,
                    confirmed,
                    random_state=int(config.get('random_state', 42)),
                    use_family_scaler=bool(config.get('use_family_scaler', True)),
                    family_min_size=int(config.get('family_min_size', 5)),
                    xgb_params=xgb_params_base,
                    use_robust_scaler=bool(config.get('use_robust_scaler', True)),
                    target_cols=_get_target_cols(config, Y, required=False),
                )

                rmg_preds = predict_arrhenius(
                    rmg_models,
                    X_full,
                    rmg_codes,
                    confirmed,
                    rmg_scalers,
                    bool(config.get('use_family_scaler', True)),
                    target_cols=_get_target_cols(config, Y, required=False),
                )

                rows.append(_evaluate_prediction_table(
                    Y,
                    rmg_preds,
                    label='RMG_label_experts_confirmed_features',
                    n_features=len(confirmed),
                ))

            except Exception as e:
                logging.warning("RMG-label expert ablation failed: %s", e)
        else:
            logging.info("RMG-label expert ablation skipped: no reference label column found.")

    if bool(config.get('diagnostics_ablation_use_all_features', False)):
        try:
            all_features = list(X_full.columns)
            xgb_params_base = copy.deepcopy(
                config.get('xgb_params', DEFAULT_CONFIG['xgb_params'])
            )
            target_cols = _get_target_cols(config, Y, required=False)

            # Use the same family labels that were actually used for specialist training
            # when bootstrap label replacement / training-family logic is active.
            if '_training_family' in df.columns:
                active_family_labels = df['_training_family'].tolist()
                active_family_series = df['_training_family']
            else:
                active_family_labels = df['family'].tolist()
                active_family_series = df['family']

            all_models, all_scalers = train_topic_models(
                X_full,
                Y,
                active_family_labels,   # <-- revised: not df['family'].tolist()
                all_features,
                random_state=int(config.get('random_state', 42)),
                use_family_scaler=bool(config.get('use_family_scaler', True)),
                family_min_size=int(config.get('family_min_size', 5)),
                xgb_params=xgb_params_base,
                use_robust_scaler=bool(config.get('use_robust_scaler', True)),
                target_cols=target_cols,
            )

            all_preds = _predict_moe_with_configured_routing(
                all_models,
                X_full,
                active_family_labels,
                all_features,
                all_scalers,
                config,
                topic_proba_df=topic_proba_df,
                stability_df=stability_df,
                routing_mode_actual=routing_mode_actual,
                hard_topic_series=active_family_series,
                target_cols=target_cols,
            )

            rows.append(_evaluate_prediction_table(
                Y,
                all_preds,
                label='MoE_all_prefiltered_features',
                n_features=len(all_features),
            ))

            gpred_all = _predict_global_model(
                all_models,
                all_scalers,
                X_full,
                all_features,
                target_cols=target_cols,
            )

            rows.append(_evaluate_prediction_table(
                Y,
                gpred_all,
                label='global_XGBoost_all_prefiltered_features',
                n_features=len(all_features),
            ))

        except Exception as e:
            logging.warning("All-feature ablation failed: %s", e)

    if rows:
        ablation_df = pd.concat(rows, ignore_index=True)
        ablation_df.to_csv(
            os.path.join(summary_dir, "component_ablation_metrics.csv"),
            index=False
        )


def run_diagnostics(
    *,
    df: pd.DataFrame,
    Y: pd.DataFrame,
    X_full: pd.DataFrame,
    preds: pd.DataFrame,
    models: Dict[Any, Any],
    scalers: Dict[Any, Any],
    confirmed: List[str],
    config: Dict[str, Any],
    stability_df: Optional[pd.DataFrame] = None,
    topic_proba_df: Optional[pd.DataFrame] = None,
    routing_mode_actual: Optional[str] = None,
) -> None:
    """Central dispatcher for network-preserving diagnostics."""
    if not bool(config.get('diagnostics_enable', False)):
        return

    out_dir = config['output_dir']
    rmg_family_col = str(config.get('rmg_family_col', 'rmg_family'))

    t0 = time.time()
    logging.info("Running diagnostics.")

    if bool(config.get('diagnostics_dataset_summary_enable', True)):
        _write_dataset_summary(df, Y, out_dir, rmg_family_col)

    if bool(config.get('diagnostics_alignment_enable', True)):
        _write_family_mechanism_alignment(df, out_dir, rmg_family_col)

    if bool(config.get('diagnostics_stability_summary_enable', True)):
        _write_stability_summary(df, stability_df, out_dir, rmg_family_col)

    if bool(config.get('diagnostics_residual_enable', True)):
        _write_residual_error_tables(
            df,
            Y,
            preds,
            out_dir,
            rmg_family_col,
            stability_df,
            top_n=int(config.get('diagnostics_residual_top_n', 25)),
        )

    if bool(config.get('diagnostics_ablation_enable', True)):
        _run_component_ablation_diagnostics(
            X_full,
            Y,
            df,
            preds,
            models,
            scalers,
            confirmed,
            config,
            out_dir,
            topic_proba_df=topic_proba_df,
            stability_df=stability_df,
            routing_mode_actual=routing_mode_actual,
        )
    # Random-cluster controls are potentially expensive, so they are controlled
    # by an explicit mode. The 'auto' mode preserves compatibility with the old
    # diagnostics_random_controls_enable Boolean key.
    random_control_mode = str(
        config.get('diagnostics_random_controls_mode', 'auto')
    ).strip().lower()

    if random_control_mode in {'', 'auto'}:
        random_control_mode = (
            'size_matched'
            if bool(config.get('diagnostics_random_controls_enable', False))
            else 'off'
        )

    if random_control_mode in {'false', 'none', 'no', '0'}:
        random_control_mode = 'off'
    elif random_control_mode in {'true', 'yes', '1'}:
        random_control_mode = 'size_matched'

    valid_random_control_modes = {
        'off',
        'size_matched',
        'empirical_label_sampling',
    }

    if random_control_mode not in valid_random_control_modes:
        raise ValueError(
            "diagnostics_random_controls_mode must be one of "
            f"{sorted(valid_random_control_modes)} or 'auto', "
            f"got '{random_control_mode}'."
        )

    if random_control_mode != 'off':
        logging.info(
            "Random-cluster controls enabled; mode=%s. "
            "For manuscript diagnostics, use mode='size_matched'.",
            random_control_mode,
        )
        config['diagnostics_random_controls_mode'] = random_control_mode
        _run_random_cluster_controls(
            X_full,
            Y,
            df,
            confirmed,
            config,
            out_dir,
        )
    else:
        logging.info(
            "Random-cluster controls skipped: diagnostics_random_controls_mode='off'."
        )

    runtime = {
        'diagnostics_runtime_seconds': float(time.time() - t0),
        'diagnostics_runtime_minutes': float((time.time() - t0) / 60.0),
    }

    summary_dir = os.path.join(out_dir, "diagnostics")
    _ensure_dir(summary_dir)

    with open(
        os.path.join(summary_dir, "diagnostics_runtime.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(runtime, f, indent=2)

    logging.info("Diagnostics completed in %.2fs.", time.time() - t0)

def run_diagnostics_k_sensitivity_grid(base_config: Dict[str, Any]) -> None:
    """
    Run separate full pipeline jobs for k-sensitivity diagnostics.

    Each k requires refitting family discovery, Boruta/model training, routing,
    and diagnostics; therefore this function is intentionally separate
    from one normal run_pipeline call.
    """
    if not bool(base_config.get('diagnostics_k_sensitivity_enable', False)):
        return

    k_values = base_config.get('diagnostics_k_values', [3, 5, 7, 9, 11])
    base_out = base_config.get('output_dir', 'network_output')
    grid_out = os.path.join(base_out, "diagnostics_k_sensitivity")
    _ensure_dir(grid_out)

    rows_global = []
    rows_alignment = []

    for k in k_values:
        cfg = copy.deepcopy(base_config)
        cfg['n_topics'] = int(k)
        # Keep FCM/GMM component counts tied to k unless the user explicitly fixed them.
        if cfg.get('fcm_k', None) is not None:
            cfg['fcm_k'] = int(k)
        if cfg.get('gmm_n_components', None) is not None:
            cfg['gmm_n_components'] = int(k)
        cfg['output_dir'] = os.path.join(grid_out, f"k_{int(k)}")
        cfg['family_selection_mode'] = 'fixed'

        # Avoid repeated expensive SHAP in k sweeps unless requested.
        cfg['shap_sample_size'] = int(cfg.get('diagnostics_k_sensitivity_shap_sample_size', 0))

        _ensure_dir(cfg['output_dir'])
        try:
            run_pipeline(cfg)

            gpath = os.path.join(cfg['output_dir'], "global_stats.csv")
            if os.path.exists(gpath):
                g = pd.read_csv(gpath)
                g['k'] = int(k)
                rows_global.append(g)

            apath = os.path.join(
                cfg['output_dir'],
                "diagnostics",
                "family_rmg_alignment_metrics.csv"
            )
            if os.path.exists(apath):
                a = pd.read_csv(apath)
                a['k'] = int(k)
                rows_alignment.append(a)

        except Exception as e:
            logging.exception("k-sensitivity run failed for k=%s: %s", k, e)

    if rows_global:
        pd.concat(rows_global, ignore_index=True).to_csv(
            os.path.join(grid_out, "k_sensitivity_global_stats_summary.csv"),
            index=False
        )
    if rows_alignment:
        pd.concat(rows_alignment, ignore_index=True).to_csv(
            os.path.join(grid_out, "k_sensitivity_alignment_summary.csv"),
            index=False
        )

@timeit_logger("Run pipeline")
def run_pipeline(config: Dict[str, Any]):
    """
    Full training + evaluation pipeline for the given configuration.

    Side-effects: writes numerous artifacts under config['output_dir'].
    """
    # 1) Setup
    setup_logging(os.path.join(config['output_dir'], 'pipeline.log'))

    # Replay duplicate-YAML-key warnings after logging is active.
    # Skip Auto-Chooser fast candidate runs to avoid repeated warnings in every
    # candidate subfolder.
    if not bool(config.get('_auto_chooser_candidate_run', False)):
        log_yaml_duplicate_key_summary(config)

    _log_pipeline_banner(config)
    logging.info(f"Starting pipeline with config: {config}")
    _ensure_dir(config['output_dir'])
    np.random.seed(config['random_state']); random.seed(config['random_state'])

    # Metrics scratch dir for extra diagnostics
    try:
        metrics_dir = os.path.join(config['output_dir'], "metrics")
        _ensure_dir(metrics_dir)
    except Exception as e:
        logging.warning("Could not write extra diagnostics: %s", e)

    # 2) Load Data
    logging.info(f"Loading data from {config['input_csv']}.")
    df = pd.read_csv(config['input_csv'])
    target_cols = _get_target_cols(config, df, required=True)
    config['target_cols'] = list(target_cols)  # canonicalize for downstream artifacts/pipeline.pkl
    required = {'reaction_smiles', *target_cols}
    if not required.issubset(df.columns):
        raise KeyError(f"Input CSV is missing required columns: {required - set(df.columns)}")
    reac = df['reaction_smiles']
    Y = df[target_cols].copy()

    # 3) Featurization
    # Optional 3D Mordred toggle via env; 
    # If 'mordred_3d_enable' is truthy, set env so descriptor builder can honor it.
    try:
        if bool(config.get('mordred_3d_enable', False)):
            os.environ['XT_MORDRED_3D'] = '1'
            logging.info("Mordred 3D descriptors: ENABLED via XT_MORDRED_3D=1")
        else:
            os.environ.pop('XT_MORDRED_3D', None)
            logging.info("Mordred 3D descriptors: DISABLED")
    except Exception as _e:
        logging.warning("Could not set XT_MORDRED_3D env flag: %s", _e)

    logging.info("Calculating Mordred descriptors (with optional cache).")
    desc_df = get_or_compute_mordred(
        reac_series=reac,
        desc_nan_policy=config['desc_nan_policy'],
        agg=config.get('mordred_agg', 'sum'),
        use_cache=bool(config.get('descriptor_cache', False)),
        cache_dir=config.get('descriptor_cache_dir', os.path.join(config['output_dir'], 'cache')),
        mordred_parallel_enable=bool(config.get('mordred_parallel_enable', False)),
        mordred_parallel_n_jobs=int(config.get('mordred_parallel_n_jobs', 1)),
        mordred_parallel_backend=str(config.get('mordred_parallel_backend', 'loky')),
        mordred_parallel_batch_size=config.get('mordred_parallel_batch_size', 'auto'),
        mordred_xtqspr_explicit_h_patch_enable=bool(
            config.get('mordred_xtqspr_explicit_h_patch_enable', False)
        ),
    )

    logging.info("Generating simple reaction flags.")
    flags_df = pd.DataFrame({
        'is_unimolecular': [1 if len(r.split('>>')[0].split('.')) == 1 else 0 for r in reac],
        'has_competing_products': [1 if len(r.split('>>')[1].split('.')) > 1 else 0 for r in reac]
    }, index=reac.index)

    if bool(config.get('hydrogen_radical_special_case_enable', True)):
        logging.info("Adding hydrogen-radical special-case descriptors.")
        hrad_df = build_hydrogen_radical_feature_df(reac)
        flags_df = pd.concat([flags_df, hrad_df], axis=1)

    logging.info("Generating DRFP and reference LDA.")
    drfp_df, topics_baseline_lda, lda, topic_proba_df_lda = generate_drfp_topics(reac, config)

    # If generate_drfp_topics dropped zero-DRFP rows, realign all precomputed frames
    if bool(config.get('drfp_drop_zero_rows', False)) and not reac.index.equals(drfp_df.index):
        keep_idx = drfp_df.index
        missing_n = len(reac) - len(keep_idx)
        logging.warning(
            "Dropped %d reactions due to all-zero DRFP; aligning df/reac/Y/descriptors/flags to DRFP index.",
            missing_n
        )
        df      = df.loc[keep_idx].copy()
        reac    = reac.loc[keep_idx].copy()
        Y       = Y.loc[keep_idx].copy()
        try:
            desc_df  = desc_df.loc[keep_idx].copy()
            flags_df = flags_df.loc[keep_idx].copy()
        except NameError:
            # descriptors/flags may be computed after this point in some variants
            pass
   
    ref_doc_topic = topic_proba_df_lda.values  # N×K

    # -------- Choose topic method: LDA (default) or FCM variants ----------
    topic_method = str(config.get('topic_method', 'lda')).lower()
    topics_baseline = topics_baseline_lda
    topic_proba_df  = topic_proba_df_lda
    fcm_centers = None; fcm_pca_model = None
    gmm_bundle  = None; gmm_model = None; gmm_pca_model = None
    hdbscan_bundle = None; hdb_model = None; hdb_pca_model = None
    qda_bundle  = None; qda_model = None; qda_pca_model = None
    X_for_fcm = None; X_for_qda = None; X_for_gmm = None; X_for_hdb = None

    _allowed = {'lda', 'fcm', 'lda+fcm', 'gmm', 'lda+gmm', 'hdbscan', 'lda+hdbscan', 'lda+qda'}
    if topic_method not in _allowed:
        logging.warning("Unknown topic_method '%s'; falling back to 'lda'. Allowed: %s",
                        topic_method, sorted(_allowed))
        topic_method = 'lda'
    logging.info("Selected topic method: %s", topic_method)

    # Detect which topic builder is active for this run
    _active_method = 'lda'
    if topic_method in ('fcm', 'lda+fcm'):
        _active_method = 'fcm'
    elif topic_method in ('gmm', 'lda+gmm'):
        _active_method = 'gmm'
    elif topic_method in ('hdbscan', 'lda+hdbscan'):
        _active_method = 'hdbscan' 
    elif topic_method in ('lda+qda',):
        _active_method = 'qda' 

    if topic_method in ("lda","lda+fcm","lda+gmm","lda+hdbscan","lda+qda") and bool(config.get('topic_similarity_enable', False)):
        metrics_dir = os.path.join(config['output_dir'], "metrics")
        _ensure_dir(metrics_dir)

        # 3.1 Perplexity on DRFP doc-term (sklearn's definition)
        try:
            lda_perp = float(lda.perplexity(drfp_df.values))
            with open(os.path.join(metrics_dir, "lda_perplexity.json"), "w", encoding="utf-8") as f:
                    json.dump({"n_topics": int(config.get("n_topics", drfp_df.shape[1] if hasattr(lda, "n_components") else None)),
                            "perplexity": lda_perp}, f, indent=2)
        except Exception as e:
            logging.warning("LDA perplexity could not be computed: %s", e)

        # 3.2 Topic–topic cosine in document-topic space
        DT = topic_proba_df.values  # N x K
        from sklearn.metrics.pairwise import cosine_similarity
        cos_tt = cosine_similarity(DT.T, DT.T)
        pd.DataFrame(cos_tt, index=topic_proba_df.columns, columns=topic_proba_df.columns)\
        .to_csv(os.path.join(metrics_dir, "lda_topic_cosine.csv"))

        # 3.3 Jaccard over top-N "words"/bits (use components_)
        try:
            topn = int(config.get('topic_similarity_topn', 20))
            comps = lda.components_  # K x V
            top_sets = [set(np.argsort(row)[-topn:]) for row in comps]
            K = len(top_sets)
            J = np.zeros((K, K), float)
            for i in range(K):
                for j in range(K):
                    inter = len(top_sets[i].intersection(top_sets[j]))
                    union = max(1, len(top_sets[i]) + len(top_sets[j]) - inter)
                    J[i, j] = inter / union
            pd.DataFrame(J, index=topic_proba_df.columns, columns=topic_proba_df.columns)\
            .to_csv(os.path.join(metrics_dir, "lda_topic_jaccard_topN.csv"))
        except Exception as e:
            logging.warning("LDA topic Jaccard failed: %s", e)

        # 3.4 Optional: Gensim topic coherence over DRFP-bit tokens
        if bool(config.get('lda_coherence_enable', False)):
            try:
                compute_lda_coherence_bits(lda, drfp_df, config, metrics_dir)
            except Exception as e:
                logging.warning("LDA coherence block failed: %s", e)

    if topic_method in ('fcm', 'lda+fcm'):
        if not HAVE_FCM:
            raise ImportError("topic_method requires scikit-fuzzy. Install with: pip install scikit-fuzzy")

        fcm_input = str(config.get('fcm_input', 'lda')).lower()
        if fcm_input == 'drfp':
            # (1) DRFP → (optional PCA) → FCM
            X_for_fcm, fcm_pca_model = fit_pca_on_drfp(drfp_df, config)
            logging.info("Running fcm on DRFP(+PCA): PCA=%s", 'on' if fcm_pca_model is not None else 'off')
        else:
            # (2) LDA topic space → FCM  (recommended)
            X_for_fcm, fcm_pca_model = topic_proba_df_lda, None
            logging.info("Running fcm on LDA topic space")

        fcm_k = int(config.get('fcm_k', config.get('n_topics', 6)))
        fcm_m = float(config.get('fcm_m', 1.1))
        fcm_max_iter = int(config.get('fcm_max_iter', 300))
        fcm_error = float(config.get('fcm_error', 1e-6))

        logging.info("Running FCM: input=%s | K=%d | m=%.3f | max_iter=%d | error=%.1e | PCA=%s",
                     'DRFP' if fcm_input == 'drfp' else 'LDA-topic',
                     fcm_k, fcm_m, fcm_max_iter, fcm_error,
                     'on' if fcm_pca_model is not None else 'off')

        topics_baseline_fcm, proba_fcm_df, fcm_centers = compute_fcm_topics(
            X_for_fcm, n_clusters=fcm_k, m=fcm_m, max_iter=fcm_max_iter, error=fcm_error,
            random_state=int(config.get('random_state', 42))
        )

        # For transparency, expose both LDA and FCM baselines
        df['family_lda'] = topics_baseline_lda
        df['family_fcm'] = topics_baseline_fcm

        # Use FCM outputs as the active baseline
        topics_baseline = topics_baseline_fcm
        topic_proba_df  = proba_fcm_df

        if bool(config.get('cluster_validity_enable', False)):
            try:
                U = topic_proba_df.values   # N x K
                C = np.asarray(fcm_centers) # K x d
                X = X_for_fcm.values
                eps = 1e-12
                N, K = U.shape

                # Partition Coefficient & Entropy
                PC = float(np.sum(U**2) / N)
                PE = float(-np.sum(np.clip(U, eps, 1.0) * np.log(np.clip(U, eps, 1.0))) / N)

                # Xie–Beni index
                # numerator: sum_i sum_k u_ik^m * ||x_i - c_k||^2 (use m from config)
                m = float(config.get('fcm_m', 1.1))
                dists2 = np.sum((X[:,None,:] - C[None,:,:])**2, axis=2)     # N x K
                num = float(np.sum((U**m) * dists2))
                # denominator: N * min_{k != l} ||c_k - c_l||^2
                pair = C[:,None,:] - C[None,:,:]
                min_sep2 = float(np.min(np.sum(pair**2, axis=2) + np.eye(K)*1e9))
                XB = float(num / max(eps, (N * min_sep2)))
            
                with open(os.path.join(metrics_dir, "fcm_validity.json"), "w", encoding="utf-8") as f:
                    json.dump({'PC': PC, 'PE': PE, 'XB': XB}, f, indent=2)
            except Exception as e:
                logging.warning("FCM validity indices failed: %s", e)

    # ---- LDA+QDA path ------------------------------------------------
    elif topic_method in ('lda+qda',):
        # Choose feature space for QDA
        qda_input = str(config.get('qda_input', 'lda')).lower()
        if qda_input == 'drfp':
            X_for_qda, qda_pca_model = fit_pca_on_drfp(drfp_df, config)
            logging.info("Running QDA on DRFP(+PCA): PCA=%s", 'on' if qda_pca_model is not None else 'off')
        else:
            X_for_qda, qda_pca_model = topic_proba_df_lda, None
            logging.info("Running QDA on LDA topic space")

        K = int(config.get('n_topics', config.get('lda_n_topics', 6)))
        qda_params = {
            'n_classes': K,
            'reg_param': float(config.get('qda_reg_param', 0.0)),
            'priors': config.get('qda_priors', None),
            'store_covariance': bool(config.get('qda_store_covariance', False)),
            'tol': float(config.get('qda_tol', 1e-4)),
            'min_class_count': int(config.get('qda_min_class_count', 5)),
        }

        if bool(config.get('qda_feasibility_gate_enable', True)):
            _qda_report = _qda_feasibility_report(
                topics_baseline_lda,
                min_class_count=int(config.get('qda_min_class_count', 5)),
                n_classes=K,
            )
            if not _qda_report['feasible']:
                _msg = _format_qda_feasibility_message(_qda_report)
                if (str(config.get('family_selection_mode', 'fixed')).lower() == 'auto'
                        and bool(config.get('qda_auto_skip_infeasible', True))):
                    logging.info("[auto] Skipping QDA candidate before fit: %s", _msg)
                    raise AutoChooserSkipCandidate(_msg)
                raise ValueError(_msg)

        # Train QDA supervised on LDA families
        topics_baseline_qda, proba_qda_df, qda_model = compute_qda_topics(
            X_source=X_for_qda,
            base_labels=topics_baseline_lda,
            **qda_params
        )

        # Expose both labelings for audit
        df['family_lda'] = topics_baseline_lda
        df['family_qda'] = topics_baseline_qda

        # Make QDA the active family assignment
        topics_baseline = topics_baseline_qda
        topic_proba_df  = proba_qda_df

        if bool(config.get('qda_report_metrics', True)):
            try:
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
                y_true = np.asarray(topics_baseline)                # trained on LDA hard labels
                y_pred = np.asarray(df['family'])
                P = topic_proba_df.values  # N x K

                acc = float(accuracy_score(y_true, y_pred))
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
                try:
                    ll = float(log_loss(y_true, P, labels=sorted(np.unique(y_true))))
                except Exception:
                    ll = None

                with open(os.path.join(metrics_dir, "qda_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump({'accuracy': acc, 'precision_macro': float(prec), 'recall_macro': float(rec),
                            'f1_macro': float(f1), 'log_loss': ll}, f, indent=2)
            except Exception as e:
                logging.warning("QDA metrics failed: %s", e)        

        # ---------- Bootstrap-QDA (bag QDA on resamples) ----------
        agg_topic_proba_df = None
        stability_df = None
        if _bootstrap_method_enabled(config, 'qda'):
            B = int(config.get('bootstrap_B', 25))
            frac = float(config.get('bootstrap_sample_frac', 0.8))
            try:
                agg_topic_proba_df, stability_df = bootstrap_qda_topics(
                    X_source=X_for_qda,
                    base_labels=topics_baseline_lda,  # keep supervision fixed to base LDA labels
                    n_classes=K,
                    B=B,
                    sample_frac=frac,
                    reg_param=float(config.get('qda_reg_param', 0.0)),
                    priors=config.get('qda_priors', None),
                    store_covariance=bool(config.get('qda_store_covariance', False)),
                    random_state=int(config.get('random_state', 42)),
                    tol=float(config.get('qda_tol', 1e-4)),
                    sampling_mode=str(config.get('bootstrap_sampling_mode', 'bootstrap')),
                )
                # Save diagnostics
                out_dir = os.path.join(config['output_dir'], f"bootstrap_{_active_method}")
                _ensure_dir(out_dir)
                agg_topic_proba_df.to_csv(os.path.join(out_dir, "agg_topic_proba.csv"))
                stability_df.to_csv(os.path.join(out_dir, "stability.csv"))
                logging.info("Bootstrap-QDA: wrote aggregated probabilities and stability (%d bags, frac=%.2f).", B, frac)
            except Exception as e:
                logging.warning("Bootstrap-QDA failed; continuing without bootstrap artifacts. Err=%s", e)

        # Keep around for packager
        qda_bundle = {'model': qda_model, 'input': qda_input, 'pca': qda_pca_model}
    # ---- GMM / LDA+GMM path ---------------------------------------------------
    elif topic_method in ('gmm', 'lda+gmm'):
        # Input space
        gmm_input = str(config.get('gmm_input', 'lda')).lower()
        if gmm_input == 'drfp':
            X_for_gmm, gmm_pca_model = fit_pca_on_drfp(drfp_df, config)  # reuse PCA helper
            logging.info("Running GMM on DRFP(+PCA): PCA=%s", 'on' if gmm_pca_model is not None else 'off')
        else:
            X_for_gmm, gmm_pca_model = topic_proba_df_lda, None
            logging.info("Running GMM on LDA topic space")

        K = int(config.get('gmm_n_components', config.get('n_topics', 6)))
        cov = str(config.get('gmm_covariance_type', 'full'))
        reg = float(config.get('gmm_reg_covar', 1e-6))

        logging.info("Running GMM: input=%s | K=%d | cov=%s | reg=%.1e | PCA=%s",
                    'DRFP' if gmm_input == 'drfp' else 'LDA-topic', K, cov, reg,
                    'on' if gmm_pca_model is not None else 'off')

        topics_baseline_gmm, proba_gmm_df, gmm_model = compute_gmm_topics(X_for_gmm, config)

        if bool(config.get('gmm_report_aic_bic', True)) and isinstance(gmm_model, dict) and 'model' in gmm_model:
            # Save AIC/BIC of the fitted GMM (under metrics/)
            try:
                if hasattr(gmm_model, "aic") and hasattr(gmm_model, "bic"):
                    aic_val = float(gmm_model.aic(X_for_gmm.values))
                    bic_val = float(gmm_model.bic(X_for_gmm.values))
                    with open(os.path.join(metrics_dir, "gmm_ic.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "n_components": K,
                            "covariance_type": cov,
                            "AIC": aic_val,
                            "BIC": bic_val
                        }, f, indent=2)
                else:
                    logging.info("GMM ICs not available on this object; skipping AIC/BIC.")       
            except Exception as e:
                logging.warning("Failed to compute/write GMM AIC/BIC: %s", e)

        # expose both
        df['family_lda'] = topics_baseline_lda
        df['family_gmm'] = topics_baseline_gmm

        # set active baseline
        topics_baseline = topics_baseline_gmm
        topic_proba_df  = proba_gmm_df

        # Optional bootstrap for GMM
        agg_topic_proba_df = None; stability_df = None
        if _bootstrap_method_enabled(config, 'gmm'):
            agg_topic_proba_df, stability_df = bootstrap_gmm_topics(
                X_source=X_for_gmm,
                config=config,
                B=int(config.get('bootstrap_B', config.get('bootstrap_lda_B', 25))),
                sample_frac=float(config.get('bootstrap_sample_frac', config.get('bootstrap_lda_sample_frac', 0.8))),
                random_state=int(config.get('random_state', 42)),
                align_to_ref=bool(config.get('bootstrap_gmm_align_components', True)),
                ref_means=getattr(gmm_model, 'means_', None),
                ref_covariances=getattr(gmm_model, 'covariances_', None),
                align_cost=str(config.get('bootstrap_gmm_cost', 'mean_euclid')).lower(),
            )
            # save bootstrap artifacts (same folder as LDA/FCM for parity)
            out_dir = os.path.join(config['output_dir'], f"bootstrap_{_active_method}")
            _ensure_dir(out_dir)
            agg_topic_proba_df.to_csv(os.path.join(out_dir, "bootstrap_topics_aggregated.csv"))
            stability_df.to_csv(os.path.join(out_dir, "bootstrap_stability.csv"))
            logging.info("Bootstrap-GMM outputs saved to %s.", out_dir)

        # keep around for saving
        gmm_bundle = {'model': gmm_model, 'input': gmm_input, 'pca': gmm_pca_model}

    elif topic_method in ('hdbscan', 'lda+hdbscan'):
        if not HAVE_HDBSCAN:
            raise ImportError("topic_method requires hdbscan. Install with: pip install hdbscan")

        hdb_input = str(config.get('hdbscan_input', 'lda')).lower()
        if hdb_input == 'drfp':
            X_for_hdb, hdb_pca_model = fit_pca_on_drfp(drfp_df, config)  # reuse PCA helper
            logging.info("Running HDBSCAN on DRFP(+PCA): PCA=%s", 'on' if hdb_pca_model is not None else 'off')
        else:
            X_for_hdb, hdb_pca_model = topic_proba_df_lda, None
            logging.info("Running HDBSCAN on LDA topic space")

        params = {
            'min_cluster_size': int(config.get('hdbscan_min_cluster_size', 10)),
            'min_samples': config.get('hdbscan_min_samples', None),
            'metric': str(config.get('hdbscan_metric', 'euclidean')),
            'cluster_selection_epsilon': float(config.get('hdbscan_cluster_selection_epsilon', 0.0)),
            'prediction_data': bool(config.get('hdbscan_prediction_data', True)),
            'random_state': int(config.get('random_state', 42)),
            'core_dist_n_jobs' : int(config.get('hdbscan_core_dist_n_jobs', 1))
        }
        logging.info("Running HDBSCAN: input=%s | params=%s | PCA=%s",
                    'DRFP' if hdb_input == 'drfp' else 'LDA-topic',
                    params, 'on' if hdb_pca_model is not None else 'off')

        topics_baseline_hdb, proba_hdb_df, hdb_model = compute_hdbscan_topics(X_for_hdb, **params)

        if bool(config.get('hdbscan_stability_enable', True)):
            hdb = hdb_model
            out = {}
            try:
                out['cluster_persistence'] = {int(k): float(v) for k, v in enumerate(getattr(hdb, 'cluster_persistence_', []))}
            except Exception:
                pass
            try:
                out['outlier_scores_summary'] = {
                    'mean': float(np.mean(getattr(hdb, 'outlier_scores_', []))),
                    'median': float(np.median(getattr(hdb, 'outlier_scores_', []))),
                }
            except Exception:
                pass
            with open(os.path.join(metrics_dir, "hdbscan_stability.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)


        # Keep raw HDBSCAN labels for auditing (-1 = noise)
        df['hdbscan_label'] = topics_baseline_hdb

        # Choose how to define 'family' for pipeline training/reporting
        hdb_family_from = str(config.get('hdbscan_family_from', 'proba')).lower()

        if proba_hdb_df is not None and proba_hdb_df.shape[1] > 0 and hdb_family_from == 'proba':
            # Recommended: derive families from soft-membership argmax (guaranteed 0..K-1)
            topics_baseline = proba_hdb_df.values.argmax(axis=1).tolist()
        else:
            # Raw labels (may include -1). Optionally map noise→0 (and coerce all-noise → single family 0)
            lbl = np.asarray(topics_baseline_hdb)
            if bool(config.get('hdbscan_noise_to_family', True)):
                if np.all(lbl == -1):
                    logging.warning("HDBSCAN returned all noise; coercing to single family 0.")
                    lbl = np.zeros_like(lbl)
                else:
                    lbl = np.where(lbl == -1, 0, lbl)
            topics_baseline = lbl.tolist()

        # For reference: keep both the LDA family and the *chosen* HDBSCAN family
        if 'family_lda' not in df.columns:
            df['family_lda'] = topics_baseline_lda
        df['family_hdbscan'] = topics_baseline  # the chosen HDBSCAN-based family

        topic_proba_df = proba_hdb_df

        # Optional bootstrap for HDBSCAN
        agg_topic_proba_df = None; stability_df = None
        if _bootstrap_method_enabled(config, 'hdbscan'):
            agg_topic_proba_df, stability_df = bootstrap_hdbscan_topics(
                X_full=X_for_hdb,
                B=int(config.get('bootstrap_B', 25)),
                sample_frac=float(config.get('bootstrap_sample_frac', 0.8)),
                hdb_kwargs=params,
                base_hdb=hdb_model,
                log_each=True,
                random_state=int(config.get('random_state', 42))
            )
            out_dir = os.path.join(config['output_dir'], f"bootstrap_{_active_method}")
            _ensure_dir(out_dir)
            agg_topic_proba_df.to_csv(os.path.join(out_dir, "bootstrap_topics_aggregated.csv"))
            stability_df.to_csv(os.path.join(out_dir, "bootstrap_stability.csv"))
            
            # Persist a quick stability summary for HDBSCAN (mirrors LDA/FCM summaries)
            try:
                summary = {
                    "median_modal_vote": float(np.median(stability_df['modal_vote'])),
                    "mean_modal_vote": float(np.mean(stability_df['modal_vote'])),
                    "median_norm_entropy": float(np.median(stability_df['norm_entropy'])),
                    "mean_norm_entropy": float(np.mean(stability_df['norm_entropy'])),
                }
                with open(os.path.join(out_dir, "bootstrap_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                logging.warning("Failed to write HDBSCAN bootstrap_summary.json: %s", e)

            logging.info("Bootstrap-HDBSCAN outputs saved to %s.", out_dir)

        # keep around for saving
        hdbscan_bundle = {'model': hdb_model, 'input': hdb_input, 'pca': hdb_pca_model}

        # Persist HDBSCAN cluster persistence & outlier scores (optional)
        if bool(config.get('hdbscan_stability_enable', True)) and HAVE_HDBSCAN and (hdb_model is not None):
            try:
                pers = getattr(hdb_model, 'cluster_persistence_', None)
                outl = getattr(hdb_model, 'outlier_scores_', None)
                labels_attr = getattr(hdb_model, 'labels_', None)
                payload = {
                    "n_clusters": int(np.max(labels_attr[labels_attr >= 0]) + 1) if (isinstance(labels_attr, np.ndarray) and np.any(labels_attr >= 0)) else 0,
                    "cluster_persistence": (pers.tolist() if isinstance(pers, np.ndarray) else None),
                    "outlier_scores_head": (outl[:min(10, len(outl))].tolist() if isinstance(outl, np.ndarray) else None)
                }
                with open(os.path.join(metrics_dir, "hdbscan_stability.json"), "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                logging.info("HDBSCAN metrics written to %s", os.path.join(metrics_dir, "hdbscan_stability.json"))
            except Exception as e:
                logging.warning("Failed to write HDBSCAN persistence/outlier metrics: %s", e)

    # keep both versions of labels around the pipeline
    df['family_original'] = topics_baseline[:]   # baseline used by training unless replaced
    df['family'] = topics_baseline[:]            # reported family by default

    replaced_labels = None
    # topics used for training (may switch to replaced if retraining is enabled)
    topics_used_for_training = topics_baseline[:]  # default: train on baseline families
    
    # 4) Build X_full (Mordred + optional DRFP + flags), then impute on merged table
    parts = [desc_df]
    if config.get('include_drfp_in_features', True): parts.append(drfp_df)
    parts.append(flags_df)
    X_full = pd.concat(parts, axis=1)

    fill_policy = str(config.get('feature_fillna', 'mean')).lower()
    if fill_policy == 'zero': X_full = X_full.fillna(0.0)
    else: X_full = X_full.fillna(X_full.mean(numeric_only=True))
    X_full = X_full.dropna(axis=1, how='all')

    if config.get('apply_zero_variance_filter', False):
        X_full = zero_variance_filter(X_full)
    if config.get('prefilter_enable', False):
        X_full, pre_rep = pre_filter_descriptors(
            X_full,
            enable_nzv=bool(config.get('prefilter_nzv', True)),
            nzv_std_thresh=float(config.get('prefilter_nzv_std_thresh', 1e-12)),
            enable_corr=bool(config.get('prefilter_corr', False)),
            corr_threshold=float(config.get('prefilter_corr_threshold', 0.995)),
            corr_sample=int(config.get('prefilter_corr_sample', 5000)),
            random_state=int(config.get('random_state', 42)),
        )
        pf_dir = os.path.join(config['output_dir'], "prefilter")
        _ensure_dir(pf_dir)
        with open(os.path.join(pf_dir, "prefilter_report.json"), "w", encoding="utf-8") as f:
            json.dump(pre_rep, f, indent=2)
        pd.DataFrame({"kept_features": X_full.columns}).to_csv(
            os.path.join(pf_dir, "prefilter_kept_features.csv"), index=False
        )

    # 5) Bootstrap topics (works for either LDA or FCM)
    agg_topic_proba_df = None; stability_df = None; lda_list = None

    # Respect bootstrap_fcm_enable if explicitly set; else reuse bootstrap_lda_enable
    if topic_method in ('fcm', 'lda+fcm'):
        bootstrap_switch = _bootstrap_method_enabled(config, 'fcm')
    else:
        bootstrap_switch = _bootstrap_method_enabled(config, 'lda')

    if bootstrap_switch:
        if topic_method == 'lda':
            agg_topic_proba_df, stability_df, lda_list = bootstrap_lda_topics(
                drfp_df=drfp_df,
                n_topics=config['n_topics'],
                B=int(config.get('bootstrap_B', 25)),
                sample_frac=float(config.get('bootstrap_sample_frac', 0.8)),
                random_state=int(config.get('random_state', 42)),
                log_each=bool(config.get('bootstrap_log_each', False)),
                align_to_ref=bool(config.get('bootstrap_lda_topic_alignment', True)),
                ref_doc_topic=ref_doc_topic,
                sampling_mode=str(config.get('bootstrap_sampling_mode', 'bootstrap'))
            )
            method_tag = 'lda'
        else:
            # FCM bootstrap (with optional center alignment to reference)
            agg_topic_proba_df, stability_df = bootstrap_fcm_topics(
                X_full=X_for_fcm,
                B=int(config.get('bootstrap_B', 25)),
                sample_frac=float(config.get('bootstrap_sample_frac', 0.8)),
                k=int(config.get('fcm_k', config.get('n_topics', 6))),
                m=float(config.get('fcm_m', 1.1)),
                max_iter=int(config.get('fcm_max_iter', 300)),
                error=float(config.get('fcm_error', 1e-6)),
                align=bool(config.get('bootstrap_fcm_align_centers', True)),
                log_each=bool(config.get('bootstrap_log_each', False)),
                random_state=int(config.get('random_state', 42)),
                sampling_mode=str(config.get('bootstrap_sampling_mode', 'bootstrap')),
            )
            method_tag = 'fcm'

        out_dir = os.path.join(config['output_dir'], f"bootstrap_{_active_method}")
        _ensure_dir(out_dir)
        meta = {
            "method": method_tag,
            "B": int(config.get('bootstrap_B', 25)),
            "sample_frac": float(config.get('bootstrap_sample_frac', 0.8)),
            "n_topics_or_clusters": int(config.get('n_topics', config.get('fcm_k', 6))) if method_tag=='lda' else int(config.get('fcm_k', 6)),
            "aligned": bool(config.get('bootstrap_lda_topic_alignment', False)) if method_tag=='lda'
                       else bool(config.get('bootstrap_fcm_align_centers', True))
        }
        with open(os.path.join(out_dir, "bootstrap_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        agg_topic_proba_df.to_csv(os.path.join(out_dir, "bootstrap_topics_aggregated.csv"))
        stability_df.to_csv(os.path.join(out_dir, "bootstrap_stability.csv"))

        fam_boot = agg_topic_proba_df.values.argmax(axis=1)
        pd.DataFrame({'reaction_smiles': reac.values, f'family_bootstrap_{method_tag}': fam_boot},
                     index=reac.index).to_csv(os.path.join(out_dir, "bootstrap_assignments.csv"), index=False)
        if stability_df is not None and len(stability_df):
            summary = {
                "median_modal_vote": float(np.median(stability_df['modal_vote'])),
                "mean_modal_vote": float(np.mean(stability_df['modal_vote'])),
                "median_norm_entropy": float(np.median(stability_df['norm_entropy'])),
                "mean_norm_entropy": float(np.mean(stability_df['norm_entropy'])),
            }
            with open(os.path.join(out_dir, "bootstrap_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        logging.info("Bootstrap-%s outputs saved to %s.", method_tag, out_dir)

    # 6) Optional family replacement using Bootstrap stability  
    # Respect the method-specific bootstrap toggle (fallback to LDA toggle if missing)
    _bootstrap_on = _bootstrap_method_enabled(config, _active_method)

    if bool(config.get('bootstrap_replace_family_labels', False)) and _bootstrap_on:
        if (agg_topic_proba_df is None) or (stability_df is None):
            logging.warning(
                "Requested family replacement but bootstrap artifacts are missing for %s. Skipping replacement.", _active_method)
        else:
            metric = str(config.get('bootstrap_replace_metric', 'vote_fraction')).lower()
            thr = float(config.get('bootstrap_replace_threshold', 0.75))
            keep_unstable = bool(config.get('bootstrap_replace_keep_unstable', True))

            modal = agg_topic_proba_df.values.argmax(axis=1)
            if metric == 'entropy':
                stable_mask = stability_df['norm_entropy'].values <= thr
            else:
                stable_mask = stability_df['modal_vote'].values >= thr

            replaced = np.array(topics_baseline, dtype=int)
            if keep_unstable:
                replaced[stable_mask] = modal[stable_mask]
            else:
                replaced = modal  # replace all with modal from bootstrap

            # Save diagnostics
            out_dir = os.path.join(config['output_dir'], f"bootstrap_{_active_method}")
            _ensure_dir(out_dir)
            pd.DataFrame({
                'reaction_smiles': reac.values,
                'family_original': topics_baseline,
                'family_replaced': replaced,
                'stable': stable_mask.astype(int)
            }, index=reac.index).to_csv(os.path.join(out_dir, "family_replacement_detail.csv"), index=False)

            try:
                conf = pd.crosstab(pd.Series(topics_baseline, name="original"),
                                   pd.Series(replaced, name="replaced"))
                conf.to_csv(os.path.join(out_dir, "family_replacement_confusion.csv"))
            except Exception:
                pass

            frac_changed = float(np.mean(replaced != np.array(topics_baseline)))
            frac_stable  = float(np.mean(stable_mask))
            logging.info("Family replacement: %.1f%% stable, %.1f%% changed overall (metric=%s, threshold=%.3f).",
                         100*frac_stable, 100*frac_changed, metric, thr)

            # Expose replacements always, but DO NOT switch the active/reported label unless retraining
            df['family_replaced'] = replaced.tolist()
            replaced_labels = replaced.tolist()

            if bool(config.get('bootstrap_retrain_on_replaced', False)):
                # Make training & reporting consistent: switch both to replaced labels
                df['family'] = replaced_labels
                topics_used_for_training = replaced_labels
                logging.info("Retraining specialists on REPLACED families: reporting and training labels are consistent.")
            else:
                # Scientific fix: keep the original family labels active for training *and* reporting
                # Replacements are published side-by-side for audit in 'family_replaced'
                logging.info("No retraining: keeping ORIGINAL families for training and reporting; replacements available in 'family_replaced'.")
    
    # Ensure we persist the true training families for provenance (used by packager)
    df['_training_family'] = list(topics_used_for_training)

    # 7) Scaling for Boruta
    logging.info("Scaling features for Boruta.")
    X_scaled, _ = scale_features(X_full, use_robust=config['use_robust_scaler'])

    # 8) Boruta selection
    logging.info("Running Boruta for feature selection.")
    boruta_t0 = time.perf_counter()
    boruta_cache_dir = config.get('boruta_cache_dir', os.path.join(config['output_dir'], 'cache'))
    use_boruta_cache = bool(config.get('boruta_cache', False))
    estimator_tag = str(config.get('boruta_estimator', 'lgbm')).lower()
    include_names = bool(config.get('boruta_meta_include_names', False))

    boruta_mode_label = 'per-target' if config.get('boruta_per_target', True) else f"single:{str(config.get('boruta_single_target', target_cols[-1] if target_cols else 'target'))}"
    boruta_meta = {
        'mode': boruta_mode_label,
        'target_cols': list(target_cols),
        'estimator_requested': estimator_tag,
        'cache_used': False,
        'details': {},
    }

    if config.get('boruta_per_target', True):
        cache_mode = 'per:' + ','.join(target_cols)
        confirmed = None
        if use_boruta_cache:
            cached = try_load_boruta_cache(X_scaled.columns.tolist(), config, cache_mode, estimator_tag, boruta_cache_dir)
            if cached is not None:
                confirmed = cached
                boruta_meta['cache_used'] = True
                boruta_meta['details'] = {t: {'from_cache': True} for t in target_cols}

        if confirmed is None:
            # Fresh runs per configured target
            per_target_features: Dict[str, List[str]] = {}

            def _run_boruta_for_target(target: str):
                local_cfg = copy.deepcopy(config)

                # Avoid CPU oversubscription when Boruta targets run in parallel.
                # New canonical key: boruta_parallel_inner_n_jobs
                # Old generic key is kept only as a backward-compatible fallback.
                inner_jobs = int(
                    local_cfg.get(
                        'boruta_parallel_inner_n_jobs',
                        local_cfg.get('parallel_inner_n_jobs', 1)
                    )
                )

                if isinstance(local_cfg.get('boruta_lgbm_params', None), dict):
                    local_cfg['boruta_lgbm_params'] = dict(local_cfg['boruta_lgbm_params'])
                    local_cfg['boruta_lgbm_params']['n_jobs'] = inner_jobs

                if isinstance(local_cfg.get('boruta_rf_params', None), dict):
                    local_cfg['boruta_rf_params'] = dict(local_cfg['boruta_rf_params'])
                    local_cfg['boruta_rf_params']['n_jobs'] = inner_jobs

                feat_t, meta_t = boruta_feature_selection(X_scaled, Y[target], local_cfg)
                return target, list(feat_t), meta_t

            use_parallel_boruta = (
                bool(config.get('boruta_parallel_enable', config.get('parallel_enable', False)))
                and bool(config.get('boruta_parallel_targets_enable', config.get('parallel_boruta_targets', True)))
                and len(target_cols) > 1
            )

            if use_parallel_boruta:
                n_jobs_outer = int(config.get('boruta_parallel_n_jobs', config.get('parallel_n_jobs', -1)))
                backend = str(config.get('boruta_parallel_backend', config.get('parallel_backend', 'loky')))

                logging.info(
                    "Running per-target Boruta in parallel: n_jobs=%s, backend=%s, inner_n_jobs=%s",
                    n_jobs_outer,
                    backend,
                    int(config.get('boruta_parallel_inner_n_jobs', config.get('parallel_inner_n_jobs', 1))),
                )

                results = Parallel(n_jobs=n_jobs_outer, backend=backend)(
                    delayed(_run_boruta_for_target)(target)
                    for target in target_cols
                )
            else:
                results = [
                    _run_boruta_for_target(target)
                    for target in target_cols
                ]

            for target, feat_t, meta_t in results:
                per_target_features[target] = list(feat_t)
                boruta_meta['details'][target] = {**meta_t, 'n_confirmed': len(feat_t)}
                if include_names:
                    boruta_meta['details'][target]['confirmed_names'] = list(feat_t)

            confirmed = sorted(list(set().union(*[set(v) for v in per_target_features.values()]))) if per_target_features else []

            if use_boruta_cache:
                save_boruta_cache(confirmed, X_scaled.columns.tolist(), config, cache_mode, estimator_tag, boruta_cache_dir)

        boruta_meta['n_confirmed_union'] = len(confirmed)
        if include_names:
            boruta_meta['confirmed_union_names'] = confirmed

        _counts_msg = ', '.join(
            f"{t}=" + ('cache' if boruta_meta['cache_used'] else str(boruta_meta['details'].get(t, {}).get('n_confirmed', 'na')))
            for t in target_cols
        )
        logging.info("Boruta per-target: %s; union=%d", _counts_msg, len(confirmed))

    else:
        # Single target mode
        target = str(config.get('boruta_single_target', target_cols[-1] if target_cols else 'target'))
        if target not in target_cols:
            logging.warning("Invalid boruta_single_target '%s', defaulting to '%s'.", target, target_cols[-1] if target_cols else 'target')
            target = target_cols[-1] if target_cols else target
        cache_mode = f"single:{target}"
        confirmed = None
        if use_boruta_cache:
            cached = try_load_boruta_cache(X_scaled.columns.tolist(), config, cache_mode, estimator_tag, boruta_cache_dir)
            if cached is not None:
                confirmed = cached
                boruta_meta['cache_used'] = True

        if confirmed is None:
            feat_single, meta_single = boruta_feature_selection(X_scaled, Y[target], config)
            confirmed = feat_single
            boruta_meta['details'] = {target: {**meta_single, 'n_confirmed': len(feat_single)}}
            if include_names:
                boruta_meta['details'][target]['confirmed_names'] = feat_single
            if use_boruta_cache:
                save_boruta_cache(confirmed, X_scaled.columns.tolist(), config, cache_mode, estimator_tag, boruta_cache_dir)

        boruta_meta['n_confirmed'] = len(confirmed)
        if include_names:
            boruta_meta['confirmed_names'] = confirmed
        logging.info("Boruta single-target '%s' selected %d features%s.",
                     target, len(confirmed),
                     " (cache)" if boruta_meta['cache_used'] else "")

    if not confirmed:
        logging.warning("Boruta selected no features; falling back to all.")
        confirmed = X_full.columns.tolist()
        # reflect in meta
        boruta_meta['fallback_to_all'] = True
        boruta_meta['n_confirmed'] = len(confirmed)
        if include_names:
            boruta_meta['confirmed_names'] = confirmed

    boruta_elapsed = time.perf_counter() - boruta_t0
    boruta_meta['elapsed_seconds'] = float(boruta_elapsed)
    boruta_meta['elapsed_human'] = _format_elapsed(boruta_elapsed)

    logging.info("Boruta feature-selection workflow finished in %s.", _format_elapsed(boruta_elapsed))
    logging.info("Using %d features for model training.", len(confirmed))

    # --- OPTIONAL: Per-family Boruta (experimental; no public API change) ---
    features_by_family = None
    per_family_report = None
    
        # 1) Config takes precedence
    _cfg_flag = config.get('Experimental_Per_Family_Boruta', None)
    if isinstance(_cfg_flag, str):
        _per_family_on = _cfg_flag.strip().lower() in ('1','true','yes','on')
    elif _cfg_flag is None:
        # 2) If config key is missing/None, fall back to env var (legacy)
        _pfb_env = os.environ.get("Experimental_Per_Family_Boruta", "").strip().lower()
        _per_family_on = _pfb_env in ("1","true","yes","on")
    else:
        _per_family_on = bool(_cfg_flag)

    if _per_family_on:
        try:
            fam_labels = df['family'].to_numpy()
            y_targets = {k: df[k].to_numpy() for k in target_cols if k in df.columns}
            features_by_family, per_family_report = boruta_feature_selection_familywise(
                X_full=X_full,
                y_targets=y_targets,
                family_labels=fam_labels,
                cfg=config,
                global_confirmed=confirmed,
                logger_prefix="PerFamilyBoruta"
            )
            # record provenance in boruta_meta for the packager
            if per_family_report is not None:
                boruta_meta = boruta_meta or {}
                boruta_meta['per_family'] = {
                    'enabled': bool(per_family_report.get('enabled', False)),
                    'estimator': per_family_report.get('estimator'),
                    'targets': per_family_report.get('targets'),
                    'keep_frac': per_family_report.get('keep_frac'),
                    'min_family_n': per_family_report.get('min_family_n'),
                    'min_feat_out': per_family_report.get('min_feat_out'),
                    'restrict_to_global': per_family_report.get('restrict_to_global'),
                    'runtime_sec': per_family_report.get('runtime_sec'),
                    'families': per_family_report.get('families', {}),
                    'features_by_family': (features_by_family if features_by_family else None),
                }
        except Exception as e:
            logging.warning("Per-family Boruta failed; falling back to global features. Error: %s", e)
            features_by_family = None

    # 9) Hyperparameter recommender / consumption (optional)
    # -------------------------------------------------------------------------
    # This respects the design where the FINAL model is trained ONCE on the full
    # clustered network; CV is only used offline to recommend hyperparameters.
    xgb_params_base = copy.deepcopy(config.get('xgb_params', DEFAULT_CONFIG['xgb_params']))
    xgb_params_by_family = None
    xgb_params_by_target_global = None
    _hp_out_json_path = None

    if bool(config.get('hp_recommender_run', False)):
        try:
            reco = recommend_xgb_hyperparams(
                X_full=X_full,
                Y=Y,
                topics=list(topics_used_for_training),
                features=list(confirmed),
                cfg=config,
                features_by_family=features_by_family
            )

            _hp_out_json_path = os.path.join(
                config.get('output_dir', 'network_output'),
                str(config.get('hp_recommender_out_json', 'xgb_hyperparams_by_family.json'))
            )
            _hp_out_csv_path = None
            if config.get('hp_recommender_report_csv'):
                _hp_out_csv_path = os.path.join(
                    config.get('output_dir', 'network_output'),
                    str(config.get('hp_recommender_report_csv'))
                )

            write_hyperparams_artifacts(reco, _hp_out_json_path, _hp_out_csv_path)
            logging.info("Hyperparameter recommender wrote: %s", _hp_out_json_path)

            if bool(config.get('hp_recommender_exit_after', True)):
                logging.info("hp_recommender_exit_after=True → exiting before final training.")
                return

        except Exception as e:
            logging.exception("Hyperparameter recommender failed; proceeding without recommendations. Error: %s", e)

    if bool(config.get('hp_use_recommended_params', False)):
        hp_path = config.get('hp_recommended_json_path', None) or _hp_out_json_path
        if hp_path:
            try:
                # ---- Resolve path robustly (helpful for relative paths) ----
                hp_candidates = []
                hp_candidates.append(hp_path)

                # If a relative path was provided, also try relative to output_dir
                out_dir = config.get('output_dir', 'network_output')
                if isinstance(hp_path, str) and not os.path.isabs(hp_path):
                    hp_candidates.append(os.path.join(out_dir, hp_path))

                hp_path_found = None
                for cand in hp_candidates:
                    if cand and os.path.exists(cand):
                        hp_path_found = cand
                        break

                if hp_path_found is None:
                    raise FileNotFoundError(f"Could not find recommended JSON. Tried: {hp_candidates}")

                with open(hp_path_found, 'r', encoding='utf-8') as f:
                    reco_in = json.load(f)

                # -----------------------------------------------------------------
                # IMPORTANT DESIGN:
                #   - Keep xgb_params_base as your baseline config['xgb_params']
                #   - Apply recommender overrides ONLY for selected targets
                # -----------------------------------------------------------------
                # Baseline fallback (already set above, but ensure seed consistency)
                if isinstance(xgb_params_base, dict) and ('random_state' not in xgb_params_base):
                    xgb_params_base['random_state'] = int(config.get('random_state', 42))

                # ---- Determine target order used by the recommender JSON ----
                meta = (reco_in or {}).get('meta', {}) or {}
                target_cols = list(meta.get('target_cols', _get_target_cols(config, Y, required=False)))

                # ---- Parse which targets to actually apply from the recommender ----
                def _parse_target_list(v):
                    if v is None:
                        return None
                    if isinstance(v, (list, tuple)):
                        return [str(x).strip() for x in v if str(x).strip()]
                    if isinstance(v, str):
                        # allow "Ea" or "Ea,lnA" or "Ea lnA"
                        s = v.replace(',', ' ').strip()
                        return [t.strip() for t in s.split() if t.strip()]
                    return None

                # New config option (recommended):
                #   hp_use_recommended_targets: ["Ea"]   # only apply recommender to Ea
                use_targets = _parse_target_list(config.get('hp_use_recommended_targets', None))

                # Optional "denylist" (if you prefer it):
                #   hp_force_baseline_targets: ["lnA","n"]
                force_baseline = _parse_target_list(config.get('hp_force_baseline_targets', None)) or []

                if use_targets is None:
                    # default behavior: apply recommender to ALL targets found in JSON
                    use_targets = list(target_cols)

                # sanitize
                use_targets = [t for t in use_targets if t in target_cols]
                if force_baseline:
                    use_targets = [t for t in use_targets if t not in set(force_baseline)]

                if not use_targets:
                    logging.warning(
                        "hp_use_recommended_params=True but after applying hp_use_recommended_targets / "
                        "hp_force_baseline_targets, no targets remain. Using baseline xgb_params only."
                    )
                    xgb_params_by_family = None
                    xgb_params_by_target_global = None
                else:
                    # ---- Build name -> index mapping for MultiOutputXGBRegressor ----
                    name_to_idx = {t: i for i, t in enumerate(target_cols)}

                    # ---- Global per-target overrides (ONLY for selected targets) ----
                    g = (reco_in or {}).get('global', {})

                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    # NEW SECTION: optionally allow JSON to overwrite xgb_params_base
                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    if bool(config.get('hp_recommended_base_from_json', False)):
                        bt = (g or {}).get('by_target_index', None)
                        if isinstance(bt, dict) and bt:
                            try:
                                # Prefer base params from the first *applied* target (e.g., Ea if apply_targets=["Ea"])
                                preferred_idx = None
                                if use_targets:
                                    preferred_idx = int(name_to_idx.get(use_targets[0]))

                                # bt keys may be strings or ints
                                def _get_bt(idx):
                                    if idx is None:
                                        return None
                                    if str(idx) in bt:
                                        return bt[str(idx)]
                                    if idx in bt:
                                        return bt[idx]
                                    return None

                                chosen = _get_bt(preferred_idx)

                                if chosen is None:
                                    # fallback to first available key
                                    first_k = sorted([int(k) for k in bt.keys()])[0]
                                    chosen = _get_bt(first_k)

                                if isinstance(chosen, dict) and chosen:
                                    xgb_params_base = dict(chosen)
                                    # ensure reproducibility seed remains set
                                    if 'random_state' not in xgb_params_base:
                                        xgb_params_base['random_state'] = int(config.get('random_state', 42))
                                else:
                                    logging.warning(
                                        "hp_recommended_base_from_json=True but JSON global.by_target_index was empty/invalid; "
                                        "keeping baseline xgb_params_base."
                                    )

                            except Exception as _e:
                                logging.warning(
                                    "hp_recommended_base_from_json=True but failed to set xgb_params_base from JSON. Error: %s",
                                    _e
                                )
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                    xgb_params_by_target_global = None
                    if isinstance(g, dict) and isinstance(g.get('targets'), dict):
                        _gmap = {}
                        for tname in use_targets:
                            tinfo = (g['targets'] or {}).get(tname, {})
                            if isinstance(tinfo, dict) and isinstance(tinfo.get('params'), dict):
                                _gmap[int(name_to_idx[tname])] = dict(tinfo['params'])
                        xgb_params_by_target_global = _gmap if _gmap else None

                    # ---- Per-family per-target overrides (ONLY for selected targets) ----
                    fam_block = (reco_in or {}).get('families', {})
                    xgb_params_by_family = None
                    if isinstance(fam_block, dict):
                        _tmp = {}
                        for k, v in fam_block.items():
                            try:
                                fam_id = int(k)
                            except Exception:
                                continue
                            if not isinstance(v, dict):
                                continue

                            tblock = v.get('targets', None)
                            if not isinstance(tblock, dict):
                                continue

                            by_target = {}
                            for tname in use_targets:
                                tinfo = tblock.get(tname, {})
                                if isinstance(tinfo, dict) and isinstance(tinfo.get('params'), dict):
                                    by_target[int(name_to_idx[tname])] = dict(tinfo['params'])

                            if by_target:
                                _tmp[fam_id] = {'by_target': by_target}

                        xgb_params_by_family = _tmp if _tmp else None

                    logging.info(
                        "Loaded recommended hyperparams from %s | apply_targets=%s | "
                        "families_with_overrides=%d | baseline_targets=%s",
                        hp_path_found,
                        use_targets,
                        0 if xgb_params_by_family is None else len(xgb_params_by_family),
                        [t for t in target_cols if t not in use_targets]
                    )

            except Exception as e:
                logging.warning(
                    "Failed to load/apply recommended hyperparams from %s; using baseline config xgb_params. Error: %s",
                    hp_path, e
                )
        else:
            logging.warning(
                "hp_use_recommended_params=True but no hp_recommended_json_path is set and no recommender JSON was produced; "
                "using baseline config xgb_params."
            )

    # 10) Train specialists
    logging.info("Training per-family specialist models and a global fallback model.")
    models, scalers = train_topic_models(
        X_full, Y, topics_used_for_training, confirmed,
        random_state=config['random_state'],
        use_family_scaler=config['use_family_scaler'],
        family_min_size=config['family_min_size'],
        xgb_params=xgb_params_base,
        xgb_params_by_family=xgb_params_by_family,
        xgb_params_by_target_global=xgb_params_by_target_global,
        use_robust_scaler=bool(config.get('use_robust_scaler', True)),
        features_by_family=features_by_family,
        target_cols=target_cols
    )

    logging.info("Specialists were trained on: %s",
                 "REPLACED families" if list(topics_used_for_training) != list(topics_baseline) else "ORIGINAL/BASELINE families")
    
    # Choose the ACTIVE topic probabilities for routing.
    # Scientific safeguard: bootstrap is always allowed for diagnostics, but it is not
    # automatically injected into predictive soft routing unless explicitly requested.
    bootstrap_available = (
        'agg_topic_proba_df' in locals() and isinstance(agg_topic_proba_df, pd.DataFrame)
        and agg_topic_proba_df is not None
        and 'stability_df' in locals() and isinstance(stability_df, pd.DataFrame)
        and stability_df is not None
    )
    if bootstrap_available:
        try:
            _bs_med_vote = float(np.nanmedian(pd.to_numeric(stability_df['modal_vote'], errors='coerce')))
            _bs_med_ent = float(np.nanmedian(pd.to_numeric(stability_df['norm_entropy'], errors='coerce')))
        except Exception:
            _bs_med_vote, _bs_med_ent = np.nan, np.nan
        bootstrap_quality_ok = True
        if bool(config.get('bootstrap_quality_gate_enable', True)):
            min_vote = float(config.get('bootstrap_min_median_modal_vote', 0.65))
            max_ent = float(config.get('bootstrap_max_median_norm_entropy', 0.55))
            bootstrap_quality_ok = (np.isfinite(_bs_med_vote) and np.isfinite(_bs_med_ent)
                                    and _bs_med_vote >= min_vote and _bs_med_ent <= max_ent)
            if not bootstrap_quality_ok:
                logging.warning(
                    "Bootstrap routing quality gate failed: median_modal_vote=%.3f, median_norm_entropy=%.3f; predictive routing will fall back to baseline probabilities.",
                    _bs_med_vote, _bs_med_ent
                )
    else:
        bootstrap_quality_ok = False

    if bootstrap_available and bootstrap_quality_ok and bool(config.get('bootstrap_use_for_soft_routing', False)):
        proba_for_routing = agg_topic_proba_df
    else:
        proba_for_routing = topic_proba_df

    # Cluster validity (hard labels on soft space): Silhouette & Davies–Bouldin
    if bool(config.get('cluster_validity_enable', False)):
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            Z = proba_for_routing  # DataFrame of N x K memberships (bootstrap agg if available, else baseline)
            labels = np.asarray(df['family'])  # final hard labels already decided above

            # Need at least 2 clusters present
            if isinstance(Z, pd.DataFrame) and Z is not None and np.unique(labels).size >= 2:
                Xv = Z.values
                if config.get('cluster_validity_sample'):
                    rng = np.random.default_rng(int(config['random_state']))
                    idx = rng.choice(Xv.shape[0], size=int(config['cluster_validity_sample']), replace=False)
                    Xv, labels = Xv[idx], labels[idx]
                try:
                    sil = float(silhouette_score(Xv, labels))
                except Exception:
                    sil = None
                try:
                    dbi = float(davies_bouldin_score(Xv, labels))
                except Exception:
                    dbi = None

                with open(os.path.join(metrics_dir, "cluster_validity.json"), "w", encoding="utf-8") as f:
                    json.dump({"silhouette": sil, "davies_bouldin": dbi}, f, indent=2)
            else:
                logging.info("Cluster validity skipped: need >=2 clusters and a valid proba table.")
        except Exception as e:
            logging.warning("Cluster validity metrics failed: %s", e)

    # 11) Predictions & routing
    logging.info("Generating predictions for evaluation.")

    routing_mode_requested = ('hybrid' if bool(config.get('use_hybrid_routing', False))
                              else ('soft' if bool(config.get('use_soft_routing', True)) else 'hard'))

    metric = str(config.get('hybrid_stability_metric', 'entropy')).lower()
    thresh = float(config.get('hybrid_stability_threshold', 0.35))
    use_hybrid_now = (
        bool(config.get('use_hybrid_routing', False))
        and bootstrap_available
        and bootstrap_quality_ok
        and bool(config.get('bootstrap_use_for_hybrid_routing', True))
    )

    if use_hybrid_now:
        if metric in ('vote', 'vote_fraction', 'vf'):
            is_hard = stability_df['modal_vote'].reindex(X_full.index) >= thresh
        else:
            is_hard = stability_df['norm_entropy'].reindex(X_full.index) <= thresh
        hard_frac = float(np.nanmean(is_hard.astype(float)))
        if bool(config.get('bootstrap_quality_gate_enable', True)):
            min_hf = float(config.get('bootstrap_min_hard_fraction', 0.05))
            max_hf = float(config.get('bootstrap_max_hard_fraction', 0.95))
            if (not np.isfinite(hard_frac)) or hard_frac < min_hf or hard_frac > max_hf:
                logging.warning(
                    "Hybrid routing disabled by hard-fraction gate: hard_fraction=%.3f outside [%.3f, %.3f]. Falling back to %s routing.",
                    hard_frac, min_hf, max_hf, 'soft' if config.get('use_soft_routing', True) else 'hard'
                )
                use_hybrid_now = False

    routing_mode = 'hybrid' if use_hybrid_now else ('soft' if bool(config.get('use_soft_routing', True)) else 'hard')
    proba_for_active_routing = agg_topic_proba_df if use_hybrid_now else proba_for_routing

    if use_hybrid_now:
        logging.info("Using HYBRID routing (metric=%s, threshold=%.3f).", metric, thresh)
        preds = predict_arrhenius_hybrid(
            models, X_full, agg_topic_proba_df, stability_df, confirmed,
            scalers, config['use_family_scaler'], metric=metric, threshold=thresh,
            features_by_family=features_by_family, target_cols=target_cols,
            hard_topic_series=df['_training_family']
        )
        logging.info("Hybrid routing: %.1f%% HARD, %.1f%% SOFT.",
                     100.0 * hard_frac, 100.0 * (1.0 - hard_frac))
    else:
        if routing_mode_requested == 'hybrid':
            logging.info("Hybrid routing requested but not used; using %s routing.",
                         'soft' if config.get('use_soft_routing', True) else 'hard')
        if config.get('use_soft_routing', True):
            preds = predict_arrhenius_weighted(models, X_full, proba_for_routing, confirmed, scalers, config['use_family_scaler'],
                                               features_by_family=features_by_family, target_cols=target_cols
            )
        else:
            preds = predict_arrhenius(models, X_full, df['family'].tolist(), confirmed, scalers, config['use_family_scaler'],
                                      features_by_family=features_by_family, target_cols=target_cols
            )

    # 12) Save outputs (report both original and replaced labels when present)
    cols_for_report = ['reaction_smiles']
    if 'family_original' in df.columns:
        cols_for_report.append('family_original')
    if 'family_replaced' in df.columns:
        cols_for_report.append('family_replaced')
    cols_for_report.append('family')  # active/used for training+metrics+SHAP
    if 'family_lda' in df.columns:
        cols_for_report.append('family_lda')
    if 'family_fcm' in df.columns:
        cols_for_report.append('family_fcm')
    # Preserve original mechanistic/RMG labels in the main prediction output when present.
    _rmg_for_report = _safe_label_series(df, str(config.get('rmg_family_col', 'rmg_family')))
    if _rmg_for_report is not None:
        _rmg_col_name = _rmg_for_report.name
        if _rmg_col_name in df.columns and _rmg_col_name not in cols_for_report:
            cols_for_report.append(_rmg_col_name)

    # Original ground-truth targets
    truth_df = Y[target_cols].copy()

    # Error metrics
    eps = float(config.get('relerr_epsilon', 1e-8))
    mode = str(config.get('relerr_mode', 'both')).lower()

    # Always include absolute error (stable, unitful)
    err_df = pd.DataFrame(index=Y.index)
    for target in target_cols:
        pcol = f'{target}_pred'
        if target in truth_df.columns and pcol in preds.columns:
            err_df[f'{target}_abs_err'] = _abs_err(truth_df[target], preds[pcol])

    # Add chosen relative error columns
    if mode in ('abs_rel', 'both'):
        for target in target_cols:
            pcol = f'{target}_pred'
            if target in truth_df.columns and pcol in preds.columns:
                err_df[f'{target}_abs_rel_err'] = _abs_rel_err(truth_df[target], preds[pcol], eps)

    if mode in ('smape', 'both'):
        for target in target_cols:
            pcol = f'{target}_pred'
            if target in truth_df.columns and pcol in preds.columns:
                err_df[f'{target}_smape'] = _smape(truth_df[target], preds[pcol], eps)

    # Final output: IDs/families + original targets + predictions + errors

    # Family assignments — write a transparent view for auditing
    try:
        fam_cols = ['reaction_smiles', 'family']  # active label used by training/eval
        if 'family_original' in df.columns:
            fam_cols.append('family_original')
        if 'family_replaced' in df.columns:
            fam_cols.append('family_replaced')
        if 'hdbscan_label' in df.columns:
            fam_cols.append('hdbscan_label')   # (-1 = HDBSCAN noise)
        _rmg_fam_for_assignment = _safe_label_series(df, str(config.get('rmg_family_col', 'rmg_family')))
        if _rmg_fam_for_assignment is not None:
            _rmg_assignment_col = _rmg_fam_for_assignment.name
            if _rmg_assignment_col in df.columns and _rmg_assignment_col not in fam_cols:
                fam_cols.append(_rmg_assignment_col)

        fam_out = df[fam_cols].copy()
        fam_out.to_csv(os.path.join(config['output_dir'], "family_assignments.csv"), index=False)
        fam_out.to_csv(os.path.join(config['output_dir'], "family_classification.csv"), index=False)
    except Exception as e:
        logging.warning("Could not save family assignment files: %s", e)


    out_pred = pd.concat([df[cols_for_report], truth_df, preds, err_df], axis=1)
    out_pred.to_csv(os.path.join(config['output_dir'], "full_prediction_output.csv"), index=False)

    # Also save original LDA families (reference) for transparency
    pd.DataFrame({'reaction_smiles': reac.values, 'family_original_lda': topics_baseline_lda},
                 index=reac.index).to_csv(os.path.join(config['output_dir'], "family_assignments_lda_reference.csv"), index=False)

    # Feature analysis (legacy)
    pd.DataFrame({"Confirmed_Features": confirmed}).to_csv(
        os.path.join(config['output_dir'], "descriptor_family_analysis.csv"), index=False
    )

    # Per-family model importances
    for topic, model in models.items():
        if topic == 'global':
            continue

        coef_parts = []

        for i, out_name in enumerate(_target_cols_from_frame(Y)):
            try:
                feat_imp = model.estimators_[i].feature_importances_
            except Exception:
                feat_imp = np.zeros(len(confirmed), dtype=float)

            d = pd.DataFrame({
                'output': str(out_name),
                'feature': list(confirmed),
                'importance': np.asarray(feat_imp, dtype=float)
            })

            # Avoid pandas FutureWarning from concatenating an initially empty
            # DataFrame with non-empty/all-NA frames. Accumulate valid parts and
            # concatenate once.
            if not d.empty and not d.isna().all(axis=None):
                coef_parts.append(d)

        if coef_parts:
            coef_df = pd.concat(coef_parts, ignore_index=True)
        else:
            coef_df = pd.DataFrame(columns=['output', 'feature', 'importance'])

        coef_df.to_csv(os.path.join(config['output_dir'], f"family_model_formula_{topic}.csv"),index=False)

    # Family-wise stats
    stats_per_family = []
    for fam in sorted(set(df['family'].tolist())):
        fam_idx = df['family'] == fam
        y_true_fam = Y.loc[fam_idx]
        y_pred_fam = preds.loc[fam_idx]
        n_fam = int(fam_idx.sum())
        p_features = len(confirmed) if 'confirmed' in locals() and confirmed is not None else X_full.shape[1]

        for out_name in _target_cols_from_frame(Y):
            try:
                r2 = _safe_r2_score(y_true_fam[out_name], y_pred_fam[f"{out_name}_pred"])
                mae = _safe_mae(y_true_fam[out_name], y_pred_fam[f"{out_name}_pred"])
                rmse = _safe_rmse(y_true_fam[out_name], y_pred_fam[f"{out_name}_pred"])
                adj_r2 = _adj_r2(r2, n_fam, p_features)

                nrmse = _nrmse(y_true_fam[out_name], y_pred_fam[f"{out_name}_pred"])
                aard = _aard(y_true_fam[out_name], y_pred_fam[f"{out_name}_pred"])
                rae = _rae(y_true_fam[out_name], y_pred_fam[f"{out_name}_pred"])
            except Exception:
                r2, mae, rmse, adj_r2 = np.nan, np.nan, np.nan, np.nan
                nrmse, aard, rae = np.nan, np.nan, np.nan

            stats_per_family.append({
                "family": fam,
                "output": out_name,
                "R2": r2,
                "AdjR2": adj_r2,
                "MAE": mae,
                "RMSE": rmse,
                "nRMSE": nrmse,   
                "AARD": aard,     
                "RAE": rae,
            })

    pd.DataFrame(stats_per_family).to_csv(
        os.path.join(config['output_dir'], "family_stats.csv"),
        index=False
    )

    # Weighted per-family stats for SOFT/HYBRID routing

    try:
        if routing_mode in ('soft','hybrid') and isinstance(proba_for_active_routing, pd.DataFrame) and (not proba_for_active_routing.empty):
            soft_rows = []
            W_stats = _normalize_probability_rows(proba_for_active_routing.reindex(X_full.index).values)
            K = W_stats.shape[1]
            p_features = len(confirmed) if 'confirmed' in locals() and confirmed is not None else X_full.shape[1]

            for k in range(K):
                w = W_stats[:, k]
                mask = w > 0  # evaluate over all, weights already carry membership
                for out_name in _target_cols_from_frame(Y):
                    y  = Y[out_name].values[mask]
                    yp = preds[f"{out_name}_pred"].values[mask]
                    ww = w[mask]

                    if y.size == 0:
                        r2w = mae_w = rmse_w = adj_r2w = np.nan
                        nrmse_w = aard_w = rae_w = np.nan
                    else:
                        r2w   = _w_r2(y, yp, ww)
                        mae_w = float(np.average(np.abs(y - yp), weights=ww))
                        rmse_w = float(np.sqrt(np.average((y - yp)**2, weights=ww)))

                        nrmse_w = _w_nrmse(y, yp, ww)
                        aard_w  = _w_aard(y, yp, ww)
                        rae_w   = _w_rae(y, yp, ww)

                        # Approximate n as number of samples with non-zero weight
                        n_eff = int(y.size)
                        adj_r2w = _adj_r2(r2w, n_eff, p_features)

                    soft_rows.append({
                        "family": k,
                        "output": out_name,
                        "wR2": r2w,
                        "wAdjR2": adj_r2w,
                        "wMAE": mae_w,
                        "wRMSE": rmse_w,
                        "wnRMSE": nrmse_w,   
                        "wAARD": aard_w,    
                        "wRAE": rae_w,       
                    })

            pd.DataFrame(soft_rows).to_csv(
                os.path.join(metrics_dir, "family_stats_soft.csv"),
                index=False
            )
    except Exception as e:
        logging.warning("Could not compute weighted family stats for soft/hybrid routing: %s", e)

    # Global stats
    global_stats = []
    p_features = len(confirmed) if 'confirmed' in locals() and confirmed is not None else X_full.shape[1]

    for out_name in _target_cols_from_frame(Y):
        try:
            r2 = _safe_r2_score(Y[out_name], preds[f"{out_name}_pred"])
            mae = _safe_mae(Y[out_name], preds[f"{out_name}_pred"])
            rmse = _safe_rmse(Y[out_name], preds[f"{out_name}_pred"])
            n_global = len(Y[out_name])
            adj_r2 = _adj_r2(r2, n_global, p_features)

            nrmse = _nrmse(Y[out_name], preds[f"{out_name}_pred"])
            aard = _aard(Y[out_name], preds[f"{out_name}_pred"])
            rae = _rae(Y[out_name], preds[f"{out_name}_pred"])

        except Exception:
            r2, mae, rmse, adj_r2 = np.nan, np.nan, np.nan, np.nan
            nrmse, aard, rae = np.nan, np.nan, np.nan

        global_stats.append({
            "output": out_name,
            "R2": r2,
            "AdjR2": adj_r2,
            "MAE": mae,
            "RMSE": rmse,
            "nRMSE": nrmse,   
            "AARD": aard,     
            "RAE": rae,
        })

    pd.DataFrame(global_stats).to_csv(
        os.path.join(config['output_dir'], "global_stats.csv"),
        index=False
    )

    # 12b) Descriptive figure diagnostics
    # Same control pattern as requested for SHAP:
    #   descriptive_figures_enable=True  -> full PNG figures + related CSV data + summary CSV
    #   descriptive_figures_enable=False -> summary CSV only
    try:
        desc_dir = os.path.join(config['output_dir'], "descriptive_figures")
        if bool(config.get('descriptive_figures_enable', True)):
            logging.info("Computing descriptive reaction-family figures.")
            compute_and_save_descriptive_figures(
                df=df,
                X=X_full,
                Y=Y,
                preds=preds,
                families=df['family'].tolist(),
                features=confirmed,
                output_dir=desc_dir,
                topic_proba_df=(proba_for_routing if 'proba_for_routing' in locals() else None),
                stability_df=(stability_df if 'stability_df' in locals() else None),
                rmg_family_col=str(config.get('rmg_family_col', 'rmg_family')),
                max_features=int(config.get('descriptive_figures_max_features', 20)),
                boxplot_max_features=int(config.get('descriptive_figures_boxplot_max_features', 6)),
                pca_sample_size=config.get('descriptive_figures_pca_sample_size', None),
                random_state=int(config.get('random_state', 42)),
                dpi=int(config.get('descriptive_figures_dpi', 600)),
                font_size=int(config.get('descriptive_figures_font_size', 12)),
                heatmap_cmap=str(config.get('descriptive_figures_heatmap_cmap', 'YlGnBu')),
                descriptor_heatmap_cmap=str(config.get(
                    'descriptive_figures_descriptor_heatmap_cmap',
                    config.get('descriptive_figures_heatmap_cmap', 'RdBu_r')
                )),
                routing_heatmap_cmap=str(config.get(
                    'descriptive_figures_routing_heatmap_cmap',
                    config.get('descriptive_figures_heatmap_cmap', 'YlGnBu')
                )),
                routing_family_heatmap_cmap=str(config.get(
                    'descriptive_figures_routing_family_heatmap_cmap',
                    config.get('descriptive_figures_routing_heatmap_cmap',
                            config.get('descriptive_figures_heatmap_cmap', 'YlGnBu'))
                )),
                heatmap_annotation_auto_contrast=bool(config.get(
                    'descriptive_figures_heatmap_annotation_auto_contrast',
                    True
                )),
                figure_label_index_base=config.get('figure_label_index_base', 1),
            )
        else:
            logging.info("Descriptive figures disabled; writing descriptive_figures_summary.csv only.")
            write_descriptive_figures_summary(
                df=df,
                X=X_full,
                Y=Y,
                preds=preds,
                families=df['family'].tolist(),
                features=confirmed,
                output_dir=desc_dir,
                topic_proba_df=(proba_for_routing if 'proba_for_routing' in locals() else None),
                stability_df=(stability_df if 'stability_df' in locals() else None),
                rmg_family_col=str(config.get('rmg_family_col', 'rmg_family')),
            )
    except Exception as e:
        logging.warning("Descriptive figures skipped or failed: %s", e)

    # 12c) Diagnostics
    try:
        run_diagnostics(
            df=df,
            Y=Y,
            X_full=X_full,
            preds=preds,
            models=models,
            scalers=scalers,
            confirmed=confirmed,
            config=config,
            stability_df=(stability_df if 'stability_df' in locals() else None),
            topic_proba_df=(
                proba_for_active_routing
                if 'proba_for_active_routing' in locals()
                else None
            ),
            routing_mode_actual=(routing_mode if 'routing_mode' in locals() else None),
        )
    except Exception as e:
        logging.warning("Diagnostics skipped or failed: %s", e)

    # 13) SHAP
    # Preserve original XTQSPRKinetic behavior:
    #   - Core SHAP artifacts are controlled by shap_sample_size and model/data availability.
    #   - shap_soft_enable controls only the optional soft-probability-weighted SHAP aggregation.
    #   - If you want only shap_targets_summary.csv, set shap_sample_size=0.
    shap_dir = os.path.join(config['output_dir'], "shap")
    try:
        logging.info("Computing SHAP explanations (new + legacy outputs).")
        compute_and_save_shap(models, scalers, X_full, Y, df['family'].tolist(), confirmed,
                              output_dir=shap_dir,
                              use_family_scaler=config['use_family_scaler'],
                              shap_sample_size=config['shap_sample_size'],
                              figure_dpi=int(config.get('shap_figure_dpi', 600)),
                              font_size=int(config.get('shap_figure_font_size', 12)),
                              max_display=int(config.get('shap_figure_max_display', 20)),
                              figure_label_index_base=config.get('figure_label_index_base', 1))
    except Exception as e:
        logging.warning(f"SHAP computations skipped or failed: {e}")
    
    # Soft-aggregated SHAP (weighted by routing probabilities)
    # Produces: <output_dir>/shap/soft_mean_abs_shap.csv and PNG only when shap_soft_enable=True
    try:
        if bool(config.get('shap_soft_enable', False)):
            # Pick the same probability table used for routing
            W = None
            if 'proba_for_routing' in locals() and isinstance(proba_for_routing, pd.DataFrame):
                W = proba_for_routing
            elif 'topic_proba_df' in locals() and isinstance(topic_proba_df, pd.DataFrame):
                W = topic_proba_df

            if W is None or W.empty:
                logging.info("Soft SHAP skipped: no routing probabilities available.")
            else:
                # Reuse shap_dir from the main SHAP block.
                if not os.path.isdir(shap_dir):
                    logging.info("Soft SHAP skipped: SHAP directory not found at %s", shap_dir)
                else:
                    import glob
                    fam_cols = list(W.columns)
                    fam_masses = np.array([float(W[c].sum()) for c in fam_cols], dtype=float)
                    total_mass = float(np.sum(fam_masses))
                    if total_mass <= 0.0:
                        logging.info("Soft SHAP skipped: zero total membership mass.")
                    else:
                        fam_weights = fam_masses / total_mass  # global weight per family

                        # Collect per-family mean|SHAP| files written by compute_and_save_shap
                        # We look for CSVs that include both "mean_abs_shap" and the family id/name in the filename.
                        csvs = glob.glob(os.path.join(shap_dir, "*.csv"))
                        per_fam_maps = {}
                        for c in fam_cols:
                            c_lower = str(c).lower()
                            match = None
                            for path in csvs:
                                name = os.path.basename(path).lower()
                                if ("mean_abs_shap" in name or "mean-abs-shap" in name) and c_lower in name:
                                    match = path
                                    break
                            if match:
                                try:
                                    df_sh = pd.read_csv(match)
                                    # Normalize column names to ['feature', 'mean_abs_shap']
                                    feat_col, val_col = None, None
                                    for k in df_sh.columns:
                                        lk = k.lower()
                                        if feat_col is None and ("feature" in lk or "name" in lk):
                                            feat_col = k
                                        if val_col is None and ("mean_abs_shap" in lk or "mean|shap|" in lk or "value" in lk):
                                            val_col = k
                                    if feat_col and val_col:
                                        per_fam_maps[c] = df_sh[[feat_col, val_col]].rename(
                                            columns={feat_col: "feature", val_col: "mean_abs_shap"}
                                        )
                                except Exception as e:
                                    logging.warning("Failed to read SHAP summary for family %s: %s", c, e)

                        if not per_fam_maps:
                            logging.info("Soft SHAP skipped: no per-family mean_abs_shap CSVs found in %s", shap_dir)
                        else:
                            # Union of features from all families
                            features = sorted(set().union(*[set(dfm['feature']) for dfm in per_fam_maps.values()]))
                            agg = pd.DataFrame({'feature': features})
                            vals = np.zeros(len(features), dtype=float)

                            # Weighted blend: soft_mean_abs_shap(f) = sum_k w_k * mean_abs_shap_k(f)
                            # where w_k = total membership mass assigned to family k.
                            for i, f in enumerate(features):
                                s = 0.0
                                for j, c in enumerate(fam_cols):
                                    dfm = per_fam_maps.get(c)
                                    if dfm is None:
                                        continue
                                    row = dfm[dfm['feature'] == f]
                                    if not row.empty:
                                        s += fam_weights[j] * float(row['mean_abs_shap'].values[0])
                                vals[i] = s

                            agg['soft_mean_abs_shap'] = vals
                            agg = agg.sort_values('soft_mean_abs_shap', ascending=False)

                            out_csv = os.path.join(shap_dir, "soft_mean_abs_shap.csv")
                            agg.to_csv(out_csv, index=False)

                            # Paper-ready soft-aggregated SHAP bar plot.
                            try:
                                topn = int(config.get('shap_figure_max_display', 20))
                                top = agg.head(max(1, topn)).iloc[::-1]
                                with plt.rc_context(_publication_ready_rcparams(int(config.get('shap_figure_font_size', 12)))):
                                    plt.figure(figsize=(8.0, max(4.5, 0.34 * len(top) + 1.8)))
                                    ax = plt.gca()
                                    ax.barh(top['feature'].astype(str), top['soft_mean_abs_shap'].astype(float))
                                    ax.set_xlabel('Soft mean |SHAP|')
                                    ax.set_ylabel('Feature')
                                    ax.set_title('Soft-routing weighted mean |SHAP|', fontweight='bold')
                                    _save_current_figure_publication(
                                        os.path.join(shap_dir, "soft_mean_abs_shap.png"),
                                        dpi=int(config.get('shap_figure_dpi', 600)),
                                        font_size=int(config.get('shap_figure_font_size', 12)),
                                    )
                            except Exception as e:
                                logging.warning("Soft SHAP plot failed: %s", e)

                            logging.info("Soft SHAP summary written: %s", out_csv)
    except Exception as e:
        logging.warning("Soft SHAP aggregation failed: %s", e)

    # 14) Save pipeline components

    topic_proba_active_df = (
        proba_for_active_routing if 'proba_for_active_routing' in locals()
        else (topic_proba_df if 'topic_proba_df' in locals() else None)
    )
    
    logging.info("Saving trained pipeline to 'pipeline.pkl'.")
    save_pipeline_components_from_config(
    version=_pipeline_version_from(config),
    config=config,
    models=models,
    scalers=scalers,
    confirmed_features=confirmed,
    X_full=X_full,
    df=df,  # must have final 'family'
    topic_method_actual=str(config.get('topic_method', 'lda')),
    routing_mode_actual=(routing_mode if 'routing_mode' in locals() else None),
    lda_model=(lda if 'lda' in locals() else None),
    drfp_df=(drfp_df if 'drfp_df' in locals() else None),
    flags_df=(flags_df if 'flags_df' in locals() else None),
    fcm_centers=(fcm_centers if 'fcm_centers' in locals() else None),
    prefilter_report=(pre_rep if 'pre_rep' in locals() else None),
    boruta_meta=boruta_meta,  
    topic_proba_active_df=topic_proba_active_df,
    bootstrap_agg_topic_proba=(agg_topic_proba_df if 'agg_topic_proba_df' in locals() else None),
    bootstrap_stability=(stability_df if 'stability_df' in locals() else None),
    global_stats_df=(pd.DataFrame(global_stats) if 'global_stats' in locals() else None),
    family_stats_df=(pd.DataFrame(stats_per_family) if 'stats_per_family' in locals() else None),
    output_path=os.path.join(config['output_dir'], 'pipeline.pkl'),
    fcm_pca_model=fcm_pca_model if 'fcm_pca_model' in locals() else None,
    gmm_model=gmm_bundle if 'gmm_bundle' in locals() else None,
    hdbscan_model=hdbscan_bundle if 'hdbscan_bundle' in locals() else None,
    qda_model=qda_bundle if 'qda_bundle' in locals() else None,
)

    logging.info("Pipeline completed successfully. All outputs are in '%s'.", config['output_dir'])
