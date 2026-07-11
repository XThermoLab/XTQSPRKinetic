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
from .utils import _sha1, _ensure_dir, timeit_logger, _get_target_cols
from .config import *

# --------------------------- FEATURE SELECTION -------------------------------

def _boruta_cache_key(X_cols: List[str], cfg: dict, mode: str, estimator: str) -> str:
    payload = "|".join([
        f"mode={mode}", f"est={estimator}",
        f"seed={cfg.get('random_state', 42)}",
        f"max_iter={cfg.get('boruta_max_iter', 100)}",
        f"ncols={len(X_cols)}", ",".join(X_cols),
    ])
    return _sha1(payload)[:16]

def try_load_boruta_cache(X_cols: List[str], cfg: dict, mode: str, estimator: str, cache_dir: str):
    """
    Load Boruta 'confirmed' features from cache if present and matching metadata.
    """
    key = _boruta_cache_key(X_cols, cfg, mode, estimator)
    path = os.path.join(cache_dir, f"boruta_{key}.json")
    if not os.path.exists(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
        if data.get("columns", []) == list(X_cols) and data.get("mode")==mode and data.get("estimator")==estimator:
            logging.info("Loaded Boruta selection from cache: %s", path)
            return data.get("confirmed", [])
    except Exception as e:
        logging.warning("Failed to load Boruta cache (%s): %s", path, e)
    return None

def save_boruta_cache(confirmed: List[str], X_cols: List[str], cfg: dict, mode: str, estimator: str, cache_dir: str):
    _ensure_dir(cache_dir)
    key = _boruta_cache_key(X_cols, cfg, mode, estimator)
    path = os.path.join(cache_dir, f"boruta_{key}.json")
    data = {
        "mode": mode, "estimator": estimator, "columns": list(X_cols),
        "confirmed": list(confirmed), "random_state": cfg.get('random_state', 42),
        "boruta_max_iter": cfg.get('boruta_max_iter', 100),
        "created": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    }
    try:
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
        logging.info("Saved Boruta selection to cache: %s", path)
    except Exception as e:
        logging.warning("Could not save Boruta cache (%s): %s", path, e)

def pre_filter_descriptors(df: pd.DataFrame,
                           enable_nzv: bool = True,
                           nzv_std_thresh: float = 1e-12,
                           enable_corr: bool = False,
                           corr_threshold: float = 0.995,
                           corr_sample: int = 5000,
                           random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Optional, safe descriptor prefilter prior to Boruta.

    Rationale:
        - Zero/Near-zero variance (NZV) columns carry no predictive signal and can harm stability.
        - Extremely high pairwise correlations (|r| > threshold) inflate redundancy; dropping one side
          reduces runtime and stabilizes importances without losing information.

    Args:
        df: Feature matrix [N × D]
        enable_nzv: remove variance <= nzv_std_thresh
        nzv_std_thresh: threshold on std() for NZV removal
        enable_corr: if True, remove one side of very high |r|
        corr_threshold: absolute Pearson r threshold (e.g., 0.995)
        corr_sample: sample size for correlation estimation when N is huge
        random_state: RNG seed for sampling

    Returns:
        (df_filtered, report_dict), where report includes counts & dropped names.
    """
    report = {"n_cols_in": int(df.shape[1]), "dropped_nzv": [], "dropped_corr": [],
              "corr_threshold": corr_threshold if enable_corr else None}
    X = df.copy()
    if enable_nzv:
        std = X.std(axis=0, ddof=0).fillna(0.0)
        to_drop = std[std <= nzv_std_thresh].index.tolist()
        if to_drop:
            X = X.drop(columns=to_drop); report["dropped_nzv"] = to_drop
            logging.info("PreFilter: dropped %d NZV cols.", len(to_drop))
    if enable_corr and X.shape[1] > 1:
        n = len(X); Xs = X.sample(n=min(n, corr_sample), random_state=random_state) if n > corr_sample else X
        corr = Xs.corr(method='pearson').abs()
        upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
        cols = np.array(corr.columns); drop_mask = np.zeros(len(cols), dtype=bool)
        cv = corr.values
        for i in range(len(cols)-1):
            if drop_mask[i]: continue
            high = np.where((upper[i, :] == True) & (cv[i, :] > corr_threshold))[0]
            if len(high) > 0: drop_mask[high] = True
        to_drop = cols[drop_mask].tolist()
        if to_drop:
            X = X.drop(columns=to_drop); report["dropped_corr"] = to_drop
            logging.info("PreFilter: dropped %d highly correlated cols (|r|>%.3f).", len(to_drop), corr_threshold)
    report["n_cols_out"] = int(X.shape[1])
    return X, report

@timeit_logger("Boruta feature selection")
def boruta_feature_selection(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    config: Dict[str, Any]
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Run Boruta feature selection with LGBM (default) or RF, with optional RF rescue.

    IMPORTANT: Mirrors original XTQSPRKinetic behavior by dropping any row that has
    a missing value in X (any column) or y before computing early-exit stats
    and fitting the Boruta estimator.

    Returns
    -------
    confirmed : list[str]
        Confirmed feature names (columns of X). On degenerate cases, returns all.
    meta : dict
        Provenance and run settings (estimator, iterations, rescue path, counts, etc.).
    """
    # ---- Coerce y to a Series aligned to X.index, then build the NA mask ----
    if isinstance(y, pd.Series):
        y_series = y.reindex(X.index)
    else:
        y_arr = np.asarray(y).ravel()
        if len(y_arr) != len(X):
            raise ValueError(f"boruta_feature_selection: length mismatch len(y)={len(y_arr)} vs len(X)={len(X)}")
        y_series = pd.Series(y_arr, index=X.index)

    # Mask: drop any row with NA in X (any col) OR NA in y
    mask_valid = (~X.isna().any(axis=1)).values & (~y_series.isna().values)

    Xf = X.loc[mask_valid]
    yf = y_series.loc[mask_valid].values.ravel()

    n_rows_all, n_cols = X.shape
    n_rows = Xf.shape[0]
    dropped = n_rows_all - n_rows

    # ---------------- Early-exit safety checks (on filtered data) ----------------
    y_std = float(np.nanstd(yf)) if n_rows > 0 else 0.0
    n_unique = len(np.unique(yf[~np.isnan(yf)])) if n_rows > 0 else 0

    if n_rows < 2 or y_std == 0.0 or n_unique < 2:
        logging.warning(
            "Boruta: insufficient data or zero-variance target after filtering "
            "(rows=%d of %d kept, dropped=%d, y_std=%.6g, unique_y=%d); keeping all %d features.",
            n_rows, n_rows_all, dropped, y_std, n_unique, n_cols
        )
        confirmed_all = X.columns.tolist()
        meta = {
            'estimator': 'skip',
            'max_iter': int(config.get('boruta_max_iter', 50)),
            'perc': int(config.get('boruta_perc', 100)),
            'alpha': float(config.get('boruta_alpha', 0.05)),
            'two_step': bool(config.get('boruta_two_step', True)),
            'rescue_with_rf': bool(config.get('boruta_rescue_with_rf', True)),
            'rescue_perc': int(config.get('boruta_rescue_perc', 85)),
            'n_confirmed': len(confirmed_all),
            'early_exit_reason': 'insufficient_samples_or_constant_target',
            'rows_kept': int(n_rows),
            'rows_total': int(n_rows_all),
            'rows_dropped': int(dropped),
        }
        
        return confirmed_all, meta
    # ---------------------------------------------------------------------------

    # ---- Choose base estimator (LGBM preferred; RF fallback) -------------------
    est_choice = str(config.get('boruta_estimator', 'lgbm')).lower()
    rescue_with_rf = bool(config.get('boruta_rescue_with_rf', True))
    rescue_perc = int(config.get('boruta_rescue_perc', 85))
    
    def _make_lgbm():
        params = dict(DEFAULT_CONFIG.get('boruta_lgbm_params', {}))
        params.update(config.get('boruta_lgbm_params', {}) or {})
        return LGBMRegressor(**params)

    def _make_rf():
        params = dict(DEFAULT_CONFIG.get('boruta_rf_params', {}))
        params.update(config.get('boruta_rf_params', {}) or {})
        return RandomForestRegressor(**params)

    def _run(  
    X: pd.DataFrame,  
    y: pd.Series,  
    estimator,  
    **boruta_kwargs,  
    ) -> Tuple[List[str], object]:  
        # Ensure y is 1D array  
        y_arr = np.asarray(y).ravel()  
        logging.info("Boruta initialized with: estimator=%s", estimator)
        # Initialize and fit Boruta  
        selector = BorutaPy(estimator=estimator, **boruta_kwargs)  
        selector.fit(X.values, y_arr)  

        # Extract support mask and confirmed features  
        mask_support = getattr(selector, "support_", None)  
        confirmed = X.columns[mask_support].tolist() if mask_support is not None else []  

        return confirmed, selector 

    if est_choice == 'lgbm' and not HAVE_LGBM:  
        logging.warning("Boruta with LightGBM failed. Falling back to RandomForest.")  
        est_choice = 'rf'
        boruta_rounds = config.get('boruta_rf_params', {}).get('n_estimators', 100)
    else:
        boruta_rounds = config.get('boruta_lgbm_params', {}).get('n_estimators', 100)
        
    if boruta_rounds <= 0: boruta_rounds = 100

    use_lgbm = (est_choice == 'lgbm' and 'LGBMRegressor' in globals())
    base_estimator = _make_lgbm() if use_lgbm else _make_rf()
    est_used = 'lgbm' if use_lgbm else 'rf'

    # ---- Boruta settings -------------------------------------------------------
    boruta_kwargs = dict(
        n_estimators=boruta_rounds,
        perc=int(config.get('boruta_perc', 100)),
        alpha=float(config.get('boruta_alpha', 0.05)),
        two_step=bool(config.get('boruta_two_step', True)),
        max_iter=int(config.get('boruta_max_iter', 50)),
        random_state=int(config.get('random_state', 42)),
        verbose=0,
    )

    logging.info(
        "Boruta start: estimator=%s, n_estimator=%d, perc=%d, alpha=%.3f, two_step=%s; smax_iter=%d, random_state=%d, rows_kept=%d/%d (dropped=%d)",
        est_used, boruta_kwargs['n_estimators'], boruta_kwargs['perc'], boruta_kwargs['alpha'], 
        boruta_kwargs['two_step'], boruta_kwargs['max_iter'], boruta_kwargs['random_state'], n_rows, n_rows_all, dropped
    )

    confirmed, selector =_run(Xf, yf, estimator=base_estimator,**boruta_kwargs)
    kwargs_change=False
    # ---- Rescue / fallback logic ----------------------------------------------
    if len(confirmed) == 0 and est_used == 'lgbm':
        logging.warning("Boruta(LGBM) selected 0 features; retrying with perc=%d and max_iter=%d.",  rescue_perc, boruta_kwargs['max_iter'] + 100)
        boruta_kwargs['max_iter']=int(config.get('boruta_max_iter', 100))+100
        boruta_kwargs['perc']=rescue_perc
        confirmed, selector =_run(Xf, yf, estimator=base_estimator,**boruta_kwargs)
        boruta_kwargs['max_iter']=int(config.get('boruta_max_iter', 100))
        boruta_kwargs['perc']=int(config.get('boruta_perc', 100))
        kwargs_change=True

    if len(confirmed) == 0 and est_used == 'lgbm' and rescue_with_rf:
        logging.info("Boruta returned 0 features with LGBM; retrying with RF fallback.")
        confirmed, selector_rf =_run(Xf, yf, estimator=_make_rf(),**boruta_kwargs)
        if len(confirmed) == 0 and hasattr(selector_rf.estimator_, 'feature_importances_'):
            imps = np.asarray(selector_rf.estimator_.feature_importances_)
            thr = np.nanpercentile(imps, rescue_perc)
            confirmed = X.columns[(imps >= thr)].tolist()
            logging.info("RF rescue kept %d features at >= %dth percentile.", len(confirmed), rescue_perc)

    elif len(confirmed) == 0 and hasattr(selector.estimator_, 'feature_importances_'):
        imps = np.asarray(selector.estimator_.feature_importances_)
        thr = np.nanpercentile(imps, rescue_perc)
        confirmed = X.columns[(imps >= thr)].tolist()
        logging.info("%s rescue kept %d features at >= %dth percentile.",
                     est_used.upper(), len(confirmed), rescue_perc)

    if not confirmed:
        logging.warning("Boruta selected no features; falling back to all.")
        confirmed = X.columns.tolist()

    # ---- OPTIONAL: cap by sample-size heuristic (no API change) ---------------
    # Supports: 'sqrtn' (default suggestion) and 'log2n' (your rule).
    try:
        rule = str(config.get('boruta_feature_cap_rule', 'none')).lower()
        if rule != 'none':
            n = max(1, int(n_rows))
            f = float(config.get('boruta_feature_cap_factor', 1.0))
            cap = None
            if rule in ('sqrtn', 'sqrtn_rule', 'sqrtN'.lower()):
                cap = int(max(1, round(f * math.sqrt(n))))
            elif rule in ('log2n', 'log2n_rule', 'log2N'.lower()):
                cap = int(max(1, round(f * math.log2(n))))
            if cap is not None and len(confirmed) > cap:
                confirmed = confirmed[:cap]
                logging.info("Boruta feature cap applied (rule=%s): kept %d features (n=%d, factor=%.3f).",
                            rule, cap, n, f)
    except Exception as e:
        logging.warning("Boruta feature cap skipped: %s", e)

    meta = {
        'estimator': est_used,
        'max_iter': (int(config.get('boruta_max_iter', 100))+100) if kwargs_change else boruta_kwargs['max_iter'],
        'perc': rescue_perc if kwargs_change else boruta_kwargs['perc'],
        'alpha': boruta_kwargs['alpha'],
        'two_step': boruta_kwargs['two_step'],
        'rescue_with_rf': rescue_with_rf,
        'rescue_perc': rescue_perc,
        'n_confirmed': len(confirmed),
        'rows_kept': int(n_rows),
        'rows_total': int(n_rows_all),
        'rows_dropped': int(dropped),
    }
    return confirmed, meta

def _make_boruta_base_estimator(cfg: Dict[str, Any]):
    est_choice = str(cfg.get('boruta_estimator', 'lgbm')).lower()
    rs = int(cfg.get('random_state', 42))
    if est_choice == 'lgbm' and HAVE_LGBM:
        params = copy.deepcopy(cfg.get('boruta_lgbm_params', {}))
        params.setdefault('random_state', rs)
        try:
            return LGBMRegressor(**params), 'lgbm'
        except Exception as e:
            logging.warning("Boruta: LGBM init failed (%s); falling back to RF.", e)
    params = copy.deepcopy(cfg.get('boruta_rf_params', {}))
    params.setdefault('random_state', rs)
    return RandomForestRegressor(**params), 'rf'

@timeit_logger("Boruta feature selection familywise")
def boruta_feature_selection_familywise(
    X_full: pd.DataFrame,
    y_targets: Dict[str, Union[pd.Series, np.ndarray]],
    family_labels: Union[pd.Series, np.ndarray, List[int]],
    cfg: Dict[str, Any],
    *,
    global_confirmed: List[str],
    logger_prefix: str = "PerFamilyBoruta",
) -> Tuple[Optional[Dict[int, List[str]]], Dict[str, Any]]:
    t0 = time.time()
    fam = np.asarray(family_labels).astype(int)
    uniq, counts = np.unique(fam, return_counts=True)
    fam_sizes = dict(zip(uniq.tolist(), counts.tolist()))

    # internal knobs (no public API change)
    min_family_n = int(max(int(cfg.get('family_min_size', 15)), 80))
    n_folds_cfg  = int(max(3, int(cfg.get('boruta_family_n_folds', 5))))
    keep_frac    = 0.6
    min_feat_out = max(20, min(100, len(global_confirmed)))
    
    # expose overrides for per-family Boruta thresholds
    keep_frac = float(cfg.get('boruta_family_keep_frac', keep_frac))
    min_feat_out = int(cfg.get('boruta_family_min_features', min_feat_out))

    restrict_to_global = True
    max_iter_fast = max(20, int(round(cfg.get('boruta_max_iter', 100) / 3)))

    cfg_targets = _get_target_cols(cfg, required=False)
    targets = cfg_targets if bool(cfg.get('boruta_per_target', True)) \
              else [str(cfg.get('boruta_single_target', cfg_targets[0] if cfg_targets else 'Ea'))]
    targets = [t for t in targets if t in y_targets]
    if not targets:
        return None, {'enabled': False, 'reason': 'no_targets'}
    
    from sklearn.base import clone

    base_est, est_tag = _make_boruta_base_estimator(cfg)
    cols_pool = list(global_confirmed) if (restrict_to_global and global_confirmed) else list(X_full.columns)
    rs = int(cfg.get('random_state', 42))

    features_by_family: Dict[int, List[str]] = {}
    fam_reports = {}

    for f_id, n_f in sorted(fam_sizes.items()):
        if n_f < min_family_n:
            fam_reports[f_id] = {'size': n_f, 'status': 'fallback_global_small'}
            continue

        idx_all = np.where(fam == f_id)[0]
        n_folds = min(n_folds_cfg, max(3, min(10, n_f // 5)))
        if n_folds < 3:
            fam_reports[f_id] = {'size': n_f, 'status': 'fallback_global_fewfolds'}
            continue

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rs + f_id)

        # choose a base estimator once per family; clone it inside folds
        # Optional: use RF for very small families; falls back safely if LightGBM unavailable.
        th = int(cfg.get('boruta_family_estimator_small_threshold', 50))
        cfg_local = copy.deepcopy(cfg)
        if n_f <= th:
            cfg_local['boruta_estimator'] = 'rf'
        est_base, est_tag_family = _make_boruta_base_estimator(cfg_local)


        union_family: Set[str] = set()

        for tgt in targets:
            counts_sel = pd.Series(0, index=cols_pool, dtype=int)
            y_full = np.asarray(y_targets[tgt]).astype(float)
            folds_used = 0
            for k, (tr_rel, _te_rel) in enumerate(kf.split(idx_all), 1):
                tr_idx = idx_all[tr_rel]
                if len(tr_idx) < 20:  # protect tiny folds
                    continue
                X_tr = X_full.iloc[tr_idx][cols_pool].values
                y_tr = y_full[tr_idx]
                est_fold = clone(est_base)
                selector = BorutaPy(
                    estimator=est_fold,
                    n_estimators='auto',
                    max_iter=max_iter_fast,
                    perc=int(cfg.get('boruta_perc', 100)),
                    alpha=float(cfg.get('boruta_alpha', 0.05)),
                    two_step=bool(cfg.get('boruta_two_step', True)),
                    random_state=rs + 1000*f_id + k,
                    verbose=0
                )
                try:
                    selector.fit(X_tr, y_tr)
                    mask = (selector.support_ | selector.support_weak_)
                    if mask.sum() > 0:
                        sel_cols = [c for c, m in zip(cols_pool, mask) if m]
                        counts_sel.loc[sel_cols] += 1
                        folds_used += 1
                except Exception as e:
                    logging.warning("%s: family %d, target %s, fold %d failed: %s",
                                    logger_prefix, f_id, tgt, k, e)
                    continue

            if folds_used == 0:
                continue
            thresh = int(math.ceil(keep_frac * folds_used))
            stable_cols = counts_sel[counts_sel >= thresh].index.tolist()
            union_family.update(stable_cols)

        if len(union_family) < min_feat_out:
            need = min_feat_out - len(union_family)
            union_family.update(list(global_confirmed)[:max(0, need)])

        if union_family:
            features_by_family[f_id] = sorted(union_family)
            fam_reports[f_id] = {'size': n_f, 'status': 'ok', 'n_features': len(features_by_family[f_id])}
        else:
            fam_reports[f_id] = {'size': n_f, 'status': 'fallback_global_empty'}

    report = {
        'enabled': bool(features_by_family),
        'estimator': est_tag,
        'targets': targets,
        'restrict_to_global': restrict_to_global,
        'keep_frac': keep_frac,
        'min_family_n': min_family_n,
        'min_feat_out': min_feat_out,
        'max_iter_fast': max_iter_fast,
        'runtime_sec': round(time.time() - t0, 2),
        'families': fam_reports,
    }
    if not features_by_family:
        report['reason'] = 'no_family_selected'
        return None, report

    logging.info("Per-family Boruta: built sets for %d families.", len(features_by_family))
    return features_by_family, report
