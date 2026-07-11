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

from .config import *
from .utils import _to_int, _to_float, _sha1, _ensure_dir, timeit_logger, _target_cols_from_frame, _format_elapsed
try:
    from .mordred_xtqspr_patch import apply_xtqspr_mordred_patch
except Exception:
    apply_xtqspr_mordred_patch = None
from .xtqspr_models import MultiOutputXGBRegressor

# -------------------------- DRFP & TOPIC MODELS (LDA) -----------------------

@timeit_logger("Generat drfp topics")
def generate_drfp_topics(reactions: pd.Series, config: Dict[str, Any]):
    """
    Build DRFP features and fit a *reference* LDA (used by default and for topic alignment).

    Hydrogen-radical patch:
    For [H].[H]>>[H][H], DRFP can become all-zero because this reaction
    contains only explicit hydrogens and no heavy-atom environments.
    If enabled, this function repairs only that exact all-zero special case
    by activating deterministic sentinel bits before zero-row dropping.

    Args:
        reactions: Series[str] of reaction SMILES "R>>P"
        n_bits: DRFP folded length
        radius: DRFP radius
        n_topics: number of LDA topics (families)
        random_state: RNG seed

    Returns:
        drfp_df: DataFrame [N × n_bits] DRFP
        topic_assignments: List[int] argmax topics from reference LDA
        lda: fitted LatentDirichletAllocation object
        topic_proba_df: DataFrame [N × n_topics] per-reaction topic probabilities
    """
       
    n_bits=config.get('n_bits', 2048) 
    radius=config.get('radius', 3) 
    
    # n_components
    n_comp = config.get('lda_n_topics', None)
    if n_comp is None:
        n_comp = config.get('n_topics', 6)
    n_comp = _to_int(n_comp, 6, 2)

    # learning method
    lm = str(config.get('lda_learning_method', 'batch')).lower()
    if lm not in ('batch', 'online'):
        lm = 'batch'

    # priors (scikit-learn expects float or None; if user passed "auto", coerce to None)
    alpha = config.get('lda_doc_topic_prior', None)
    beta  = config.get('lda_topic_word_prior', None)
    if isinstance(alpha, str) and alpha.lower() == 'auto':
        logging.warning("LDA: 'auto' prior not supported by sklearn; using None.")
        alpha = None
    if isinstance(beta, str) and beta.lower() == 'auto':
        logging.warning("LDA: 'auto' prior not supported by sklearn; using None.")
        beta = None
    alpha = None if alpha in (None, 'None', '') else _to_float(alpha, None, None)
    beta  = None if beta  in (None, 'None', '') else _to_float(beta, None, None)

    max_iter = _to_int(config.get('lda_max_iter', 50), 50, 50)
    rs = _to_int(config.get('random_state', 42), 42, 42)

        
    encoder = DrfpEncoder()
    vectors = []
    for rxn in reactions:
        try: v = encoder.encode(rxn, n_folded_length=n_bits, radius=radius)
        except TypeError: v = encoder.encode(rxn)
        v = np.asarray(v); v = v.flatten() if v.ndim > 1 else v

        if bool(config.get('hydrogen_radical_special_case_enable', True)):
            v = _repair_hydrogen_radical_zero_drfp(v, rxn)

        vectors.append(v)
    arr = np.stack(vectors)  # (N, n_bits)
    drfp_df = pd.DataFrame(arr, index=reactions.index,
                           columns=[f"drfp_{i}" for i in range(arr.shape[1])])
    
    # Optional: drop reactions that produced an all-zero DRFP (invalid/no signal)
    if bool(config.get('drfp_drop_zero_rows', False)):
        zero_mask = (drfp_df.sum(axis=1) == 0)
        if bool(np.any(zero_mask)):
            n_drop = int(zero_mask.sum())
            logging.warning("Dropping %d reactions with all-zero DRFP.", n_drop)
            drfp_df = drfp_df.loc[~zero_mask]
            reactions = reactions.loc[~zero_mask]
            if len(drfp_df) == 0:
                raise ValueError("All reactions dropped due to zero DRFP; cannot fit LDA.")
            arr = drfp_df.values  # (N_filtered, n_bits)
                
    lda = LatentDirichletAllocation(
        n_components=n_comp,
        learning_method=lm,
        doc_topic_prior=alpha,      # None → sklearn default (1/n_components)
        topic_word_prior=beta,      # None → sklearn default (1/n_features)
        max_iter=max_iter,
        random_state=rs
    )

    doc_topic_dist = lda.fit_transform(arr)  # (N, K)

    topic_assignments = doc_topic_dist.argmax(axis=1).tolist()
    topic_proba_df = pd.DataFrame(doc_topic_dist, index=reactions.index,
                                  columns=[f"topic_{i}" for i in range(doc_topic_dist.shape[1])])
    return drfp_df, topic_assignments, lda, topic_proba_df

def _cosine_sim_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between columns of A (N×K) and B (N×K). Returns K×K."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    eps = 1e-15
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + eps)
    Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + eps)
    return An.T @ Bn  # (K,K)

def _align_topics_to_reference(ref_doc_topic: np.ndarray, boot_doc_topic: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Align bootstrap LDA topics to reference LDA via Hungarian assignment on cosine similarity.

    Args:
        ref_doc_topic: (N × K) reference LDA doc-topic matrix
        boot_doc_topic: (N × K) bootstrap LDA doc-topic matrix

    Returns:
        boot_aligned: (N × K) aligned bootstrap doc-topic matrix
        mean_matched_similarity: float, average cosine similarity across matched columns
    """
    K = ref_doc_topic.shape[1]
    S = _cosine_sim_columns(ref_doc_topic, boot_doc_topic)  # KxK
    if HAVE_SCIPY:
        row_ind, col_ind = linear_sum_assignment(-S)  # maximize S
        mapping = {r: c for r, c in zip(row_ind, col_ind)}
    else:
        # Greedy fallback
        mapping = {}
        used_r, used_c = set(), set()
        S_local = S.copy()
        for _ in range(K):
            i, j = np.unravel_index(np.argmax(S_local, axis=None), S_local.shape)
            while i in used_r or j in used_c:
                S_local[i, j] = -np.inf
                i, j = np.unravel_index(np.argmax(S_local, axis=None), S_local.shape)
            mapping[i] = j
            used_r.add(i); used_c.add(j); S_local[i, j] = -np.inf

    boot_aligned = np.zeros_like(boot_doc_topic)
    sims = []
    for ref_i in range(K):
        boot_j = mapping[ref_i]
        boot_aligned[:, ref_i] = boot_doc_topic[:, boot_j]
        sims.append(float(_cosine_sim_columns(ref_doc_topic[:, [ref_i]], boot_doc_topic[:, [boot_j]])[0, 0]))
    return boot_aligned, float(np.mean(sims))

def _bootstrap_sample_with_replacement(sampling_mode: Any = 'bootstrap') -> bool:
    """Return True for classical bootstrap sampling and False for subsampling stability."""
    mode = str(sampling_mode or 'bootstrap').strip().lower()
    return mode not in ('subsample', 'subsampling', 'without_replacement', 'no_replace', 'false')

def _bootstrap_method_enabled(config: Dict[str, Any], method: str) -> bool:
    """Honor method-specific bootstrap switches; None means inherit bootstrap_lda_enable."""
    key = f"bootstrap_{str(method).lower()}_enable"
    val = config.get(key, None)
    if val is None or (isinstance(val, str) and val.strip().lower() == 'none'):
        return bool(config.get('bootstrap_lda_enable', False))
    return bool(val)

def _normalize_probability_rows(P: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Finite, non-negative row-normalization for routing probability matrices."""
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P[P < 0.0] = 0.0
    if P.shape[1] == 0:
        P = np.ones((P.shape[0], 1), dtype=float)
    rs = P.sum(axis=1, keepdims=True)
    zero = (rs <= eps)
    if np.any(zero):
        P[zero.ravel(), :] = 1.0 / float(P.shape[1])
        rs = P.sum(axis=1, keepdims=True)
    return P / rs

def bootstrap_lda_topics(drfp_df: pd.DataFrame,
                         n_topics: int,
                         B: int,
                         sample_frac: float,
                         random_state: int,
                         log_each: bool = False,
                         align_to_ref: bool = False,
                         ref_doc_topic: Optional[np.ndarray] = None,
                         sampling_mode: str = 'bootstrap'
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, List[LatentDirichletAllocation]]:
    """
    Bootstrap LDA to estimate topic stability and aggregated probabilities.

    Rationale:
        Provides modal topic (vote fraction) and normalized entropy as confidence measures.
        Alignment to reference handles label switching across runs.

    Args:
        drfp_df: DataFrame, DRFP features [N × D]
        n_topics: int, number of topics
        B: int, bootstrap rounds
        sample_frac: float, fraction of samples per bootstrap fit
        random_state: int, RNG seed
        log_each: bool, verbose logging per round
        align_to_ref: bool, align each LDA to ref_doc_topic
        ref_doc_topic: np.ndarray (N×K), reference LDA doc-topic matrix

    Returns:
        agg_proba_df: DataFrame [N × K], mean topic probabilities across bootstrap rounds
        stability_df: DataFrame [N × 4], modal_topic, modal_vote, entropy, norm_entropy
        lda_list: List[LatentDirichletAllocation], fitted bootstrap models
    """
    rng = np.random.RandomState(random_state)
    X = drfp_df.values; N = X.shape[0]
    B = int(max(1, B)); sample_n = max(1, int(round(sample_frac * N)))
    sample_n = min(sample_n, N) if not _bootstrap_sample_with_replacement(sampling_mode) else sample_n
    sample_replace = _bootstrap_sample_with_replacement(sampling_mode)
    lda_list = []; proba_stack = []; match_sims = []

    t0 = time.time()
    logging.info("Bootstrap-LDA: B=%d, sample_frac=%.3f (n=%d), n_topics=%d, align_to_ref=%s, sampling_mode=%s",
                 B, sample_frac, sample_n, n_topics, align_to_ref, sampling_mode)

    for b in range(B):
        idx_boot = rng.choice(N, size=sample_n, replace=sample_replace)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=(random_state + b))
        lda.fit(X[idx_boot, :])
        proba = _normalize_probability_rows(lda.transform(X))  # N×K
        if align_to_ref and ref_doc_topic is not None:
            proba, sim_mean = _align_topics_to_reference(ref_doc_topic, proba)
            match_sims.append(sim_mean)
        proba_stack.append(proba); lda_list.append(lda)
        if log_each:
            logging.info("Bootstrap-LDA round %d/%d done.", b + 1, B)

    if match_sims:
        logging.info("Bootstrap-LDA alignment: mean matched cosine similarity = %.4f", float(np.mean(match_sims)))

    proba_stack = np.stack(proba_stack, axis=0)  # (B,N,K)
    agg_proba = proba_stack.mean(axis=0)         # (N,K)

    # modal vote across runs
    modal_topics = proba_stack.argmax(axis=2)    # (B,N)
    modal_topic = []; modal_vote = []
    for j in range(N):
        vals, counts = np.unique(modal_topics[:, j], return_counts=True)
        idx = counts.argmax(); modal_topic.append(int(vals[idx]))
        modal_vote.append(float(counts[idx] / B))
    modal_topic = np.array(modal_topic, dtype=int)
    modal_vote = np.array(modal_vote, dtype=float)

    # entropy & normalized entropy
    eps = 1e-15
    p = np.clip(agg_proba, eps, 1.0)
    ent = -(p * np.log(p)).sum(axis=1)
    norm_ent = ent / (math.log(n_topics) if n_topics > 1 else 1.0)

    agg_proba_df = pd.DataFrame(agg_proba, index=drfp_df.index, columns=[f"topic_{i}" for i in range(n_topics)])
    stability_df = pd.DataFrame({'modal_topic': modal_topic, 'modal_vote': modal_vote,
                                 'entropy': ent, 'norm_entropy': norm_ent}, index=drfp_df.index)

    logging.info("Bootstrap-LDA finished in %.2fs. Median modal_vote=%.3f; median norm_entropy=%.3f",
                 time.time() - t0, float(np.median(modal_vote)), float(np.median(norm_ent)))
    return agg_proba_df, stability_df, lda_list

# ---------------------------- GMM SUPPORT ------------------------------------

def compute_gmm_topics(X_source: pd.DataFrame, config: Dict[str, Any]) -> Tuple[List[int], pd.DataFrame, GaussianMixture]:
    """
    GMM soft clustering (probabilistic). Returns hard labels, proba matrix, and fitted GMM.
    """
    n_comp = _to_int(config.get('gmm_n_components', config.get('n_topics', 6)), 6, 3)

    cov = str(config.get('gmm_covariance_type', 'full')).lower()
    if cov not in ('full', 'tied', 'diag', 'spherical'):
        cov = 'full'

    initp = str(config.get('gmm_init_params', 'kmeans')).lower().strip()
    if '++' in initp:  # sklearn doesn’t support 'kmeans++' here
        logging.warning("GMM: 'kmeans++' init not supported; using 'kmeans'.")
        initp = 'kmeans'
    if initp not in ('kmeans', 'random'):
        logging.warning("GMM: unknown init '%s'; using 'kmeans'.", initp)
        initp = 'kmeans'

    reg      = _to_float(config.get('gmm_reg_covar', 1e-6), 1e-6, 1e-3)
    n_init   = _to_int(  config.get('gmm_n_init', 1), 1, 1)
    tol      = _to_float(config.get('gmm_tol', 1e-3), 1e-3, 1e-2)
    max_iter = _to_int(  config.get('gmm_max_iter', 100), 100, 50)
    rs = config.get('gmm_random_state', None)
    randst = _to_int(rs, _to_int(config.get('random_state', 42), 42, 42), 42) if rs is not None else _to_int(config.get('random_state', 42), 42, 42)

    gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type=cov,
        reg_covar=reg,
        init_params=initp,
        n_init=n_init,
        tol=tol,
        max_iter=max_iter,
        random_state=randst
    )
    
    gmm.fit(X_source.values)
    proba = gmm.predict_proba(X_source.values)  # (N,K)
    hard = proba.argmax(axis=1).tolist()
    proba_df = pd.DataFrame(proba, index=X_source.index, columns=[f"topic_{i}" for i in range(n_comp)])
    
    try:
        gmm._aic = float(gmm.aic(X_source.values))
        gmm._bic = float(gmm.bic(X_source.values))
    except Exception:
        gmm._aic = None; gmm._bic = None
    
    return hard, proba_df, gmm

def bootstrap_gmm_topics(
    X_source: pd.DataFrame,
    config: Dict[str, Any],
    B: int,
    sample_frac: float,
    random_state: int,
    align_to_ref: bool = True,
    ref_means: Optional[np.ndarray] = None,
    align_cost: str = None,
    ref_covariances: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap GMM to obtain sample-wise stability (modal vote, entropy) and
    aggregated responsibilities (mean posteriors across bags), with optional
    component alignment to a fixed reference ordering.

    Parameters
    ----------
    X_source : pd.DataFrame
        Feature space for GMM (LDA-topic or DRFP(+PCA); rows = reactions).
    config : Dict[str, Any]
        Pipeline configuration. GMM hyperparameters are read just like in
        `compute_gmm_topics` (n_components, covariance_type, reg_covar, ...).
    B : int
        Number of bootstrap resamples.
    sample_frac : float
        Fraction of rows per bootstrap replicate (with replacement).
    random_state : int
        RNG seed.
    align_to_ref : bool, default True
        If True, align each bag’s component order to a reference before
        accumulating responsibilities.
    ref_means : np.ndarray, optional
        Array of shape (K, d) giving the reference component means.
        If None and align_to_ref=True, a reference GMM is fit first to get these.
    align_cost : str, optional
        {'mean_euclid','mean_mahal','gaussian_kl'}; default taken from
        config['bootstrap_gmm_cost'] if present, else 'mean_euclid'.
    ref_covariances : np.ndarray, optional
        Reference covariances (K, d, d). Needed for 'mean_mahal' and 'gaussian_kl';
        will fall back to 'mean_euclid' if not supplied.
    eps : float, default 1e-6
        Ridge added to covariance matrices for numerical stability.

    Returns
    -------
    agg_topic_proba_df : pd.DataFrame, shape (N, K)
        Mean responsibilities over B bags with columns ['topic_0', ..., 'topic_{K-1}'].
    stability_df : pd.DataFrame, shape (N, 4)
        Columns: ['modal_topic','modal_vote','entropy','norm_entropy'].

    Notes
    -----
    * Uses Hungarian matching (scipy.optimize.linear_sum_assignment) when available;
      falls back to a greedy matcher otherwise.
    * If K differs across fits (should not, given fixed config), we coerce to K_ref.
    """

    # ------------------------- Prep & hyperparams -------------------------
    X = X_source.values
    index = X_source.index
    N, d = X.shape

    # Pull GMM hyperparams exactly like compute_gmm_topics
    K = int(config.get('gmm_n_components', config.get('n_topics', 6)))
    cov_type = str(config.get('gmm_covariance_type', 'full')).lower()
    if cov_type not in ('full', 'tied', 'diag', 'spherical'):
        cov_type = 'full'
    initp = str(config.get('gmm_init_params', 'kmeans')).lower()
    if '++' in initp:  # sklearn does not support 'kmeans++' here
        logging.warning("GMM bootstrap: 'kmeans++' not supported; using 'kmeans'.")
        initp = 'kmeans'
    if initp not in ('kmeans', 'random'):
        logging.warning("GMM bootstrap: unknown init '%s'; using 'kmeans'.", initp)
        initp = 'kmeans'
    reg      = float(config.get('gmm_reg_covar', 1e-6))
    n_init   = int(config.get('gmm_n_init', 1))
    tol      = float(config.get('gmm_tol', 1e-3))
    max_iter = int(config.get('gmm_max_iter', 100))
    rs_cfg   = config.get('gmm_random_state', None)
    gmm_seed = int(rs_cfg if rs_cfg is not None else config.get('random_state', random_state))

    # Alignment cost
    if align_cost is None:
        align_cost = str(config.get('bootstrap_gmm_cost', 'mean_euclid')).lower()
    if align_cost not in ('mean_euclid', 'mean_mahal', 'gaussian_kl'):
        logging.warning("GMM bootstrap: unknown align_cost '%s'; using 'mean_euclid'.", align_cost)
        align_cost = 'mean_euclid'
    need_covs = (align_cost in ('mean_mahal', 'gaussian_kl'))

    rng = np.random.RandomState(int(random_state))
    m = max(1, int(np.ceil(float(sample_frac) * N)))
    sample_replace = _bootstrap_sample_with_replacement(config.get('bootstrap_sampling_mode', 'bootstrap'))
    if not sample_replace:
        m = min(m, N)

    # --------------------- Reference (means / covariances) ---------------------
    # If align_to_ref is requested but no ref_means provided, fit a ref GMM once.
    if align_to_ref and ref_means is None:
        gmm_ref = GaussianMixture(
            n_components=K,
            covariance_type=cov_type,
            reg_covar=reg,
            init_params=initp,
            n_init=n_init,
            tol=tol,
            max_iter=max_iter,
            random_state=gmm_seed
        ).fit(X)
        ref_means = gmm_ref.means_
        # Covariances (as (K,d,d)) if needed
        if need_covs:
            ref_covariances = _gmm_covariances_as_full(gmm_ref, d, eps)

    if align_to_ref and need_covs and ref_covariances is None:
        logging.warning("GMM bootstrap: align_cost='%s' requires ref_covariances; "
                        "falling back to 'mean_euclid'.", align_cost)
        align_cost = 'mean_euclid'

    # Final K (from ref if available)
    if align_to_ref and ref_means is not None:
        K = int(ref_means.shape[0])

    # --------------------- Accumulators across B bags ---------------------
    prob_sum = np.zeros((N, K), dtype=float)
    vote_cnt = np.zeros((N, K), dtype=int)

    log_each = bool(config.get('bootstrap_log_each', config.get('bootstrap_lda_log_each', False)))  # re-use verbosity knob

    # --------------------- Helpers: covs & cost matrix ---------------------
    def _gmm_covariances_as_full(gmm_obj: GaussianMixture, dim: int, ridge: float) -> np.ndarray:
        """Return covariances as (K, d, d) full matrices regardless of covariance_type."""
        ct = gmm_obj.covariance_type
        covs = gmm_obj.covariances_
        if ct == 'full':
            C = covs
        elif ct == 'tied':
            C = np.stack([covs] * gmm_obj.n_components, axis=0)
        elif ct == 'diag':
            C = np.stack([np.diag(covs[k]) for k in range(gmm_obj.n_components)], axis=0)
        elif ct == 'spherical':
            C = np.stack([np.eye(dim) * float(covs[k]) for k in range(gmm_obj.n_components)], axis=0)
        else:
            C = np.stack([covs] * gmm_obj.n_components, axis=0)
        # ridge for numerical stability
        C = C + ridge * np.eye(dim)[None, :, :]
        return C

    def _cost_matrix(means_boot: np.ndarray,
                     covs_boot: Optional[np.ndarray],
                     means_ref: np.ndarray,
                     covs_ref: Optional[np.ndarray],
                     cost_mode: str) -> np.ndarray:
        """Build KxK cost matrix: rows=ref components, cols=boot components."""
        K_ = means_ref.shape[0]
        C = np.zeros((K_, K_), dtype=float)

        if cost_mode == 'mean_euclid':
            # ||mu_r - mu_b||_2
            for i in range(K_):
                diff = means_boot - means_ref[i]
                C[i, :] = np.sqrt(np.sum(diff * diff, axis=1))
            return C

        if covs_ref is None or covs_boot is None:
            # Should not happen if need_covs was checked; just fallback.
            for i in range(K_):
                diff = means_boot - means_ref[i]
                C[i, :] = np.sqrt(np.sum(diff * diff, axis=1))
            return C

        if cost_mode == 'mean_mahal':
            # sqrt((mu_b - mu_r)^T Σ_r^{-1} (mu_b - mu_r))
            for i in range(K_):
                try:
                    Sr_inv = np.linalg.inv(covs_ref[i])
                except np.linalg.LinAlgError:
                    Sr_inv = np.linalg.pinv(covs_ref[i])
                diffs = means_boot - means_ref[i]
                C[i, :] = np.sqrt(np.sum(diffs @ Sr_inv * diffs, axis=1))
            return C

        # 'gaussian_kl' : symmetric KL between Gaussians
        d_ = means_ref.shape[1]
        for i in range(K_):
            mu_r = means_ref[i]
            Sr = covs_ref[i]
            try:
                Sr_inv = np.linalg.inv(Sr)
                sign_r, logdet_r = np.linalg.slogdet(Sr)
            except np.linalg.LinAlgError:
                Sr_inv = np.linalg.pinv(Sr)
                sign_r, logdet_r = (1.0, np.log(np.maximum(np.linalg.det(Sr + eps * np.eye(d_)), eps)))

            for j in range(K_):
                mu_b = means_boot[j]
                Sb = covs_boot[j]
                try:
                    Sb_inv = np.linalg.inv(Sb)
                    sign_b, logdet_b = np.linalg.slogdet(Sb)
                except np.linalg.LinAlgError:
                    Sb_inv = np.linalg.pinv(Sb)
                    sign_b, logdet_b = (1.0, np.log(np.maximum(np.linalg.det(Sb + eps * np.eye(d_)), eps)))

                diff_rb = mu_b - mu_r
                # KL(N_r || N_b)
                kl_rb = 0.5 * (np.trace(Sb_inv @ Sr) + diff_rb.T @ Sb_inv @ diff_rb - d_ + (logdet_b - logdet_r))
                # KL(N_b || N_r)
                diff_br = -diff_rb
                kl_br = 0.5 * (np.trace(Sr_inv @ Sb) + diff_br.T @ Sr_inv @ diff_br - d_ + (logdet_r - logdet_b))
                C[i, j] = max(0.0, float(kl_rb + kl_br))
        return C

    # --------------------- Bootstrap loop ---------------------
    for b in range(int(max(1, B))):
        idx = rng.choice(N, size=m, replace=sample_replace)
        X_b = X[idx, :]

        gmm_b = GaussianMixture(
            n_components=K,
            covariance_type=cov_type,
            reg_covar=reg,
            init_params=initp,
            n_init=n_init,
            tol=tol,
            max_iter=max_iter,
            random_state=gmm_seed + b  # decorrelate slightly across bags
        ).fit(X_b)

        # Predict responsibilities on FULL X (N x K_run)
        R_b = _normalize_probability_rows(gmm_b.predict_proba(X))

        # Align to reference order (permute columns) if requested
        if align_to_ref and (ref_means is not None):
            means_b = gmm_b.means_
            covs_b = _gmm_covariances_as_full(gmm_b, d, eps) if need_covs else None

            costM = _cost_matrix(means_b, covs_b, ref_means, ref_covariances, align_cost)

            if HAVE_SCIPY and (linear_sum_assignment is not None):
                # rows = ref, cols = boot; get assignment from ref->boot with minimal cost
                row_ind, col_ind = linear_sum_assignment(costM)
                # We need a permutation of boot columns that maps to ref order [0..K-1]
                # row_ind is (0..K-1) already; col_ind[row_ind==i] gives the boot col matched to ref i
                # rows=reference components, cols=bootstrap components.
                # To express R_b in reference order, select the bootstrap column matched to each reference row.
                perm = np.empty(K, dtype=int)
                for r, c in zip(row_ind, col_ind):
                    if 0 <= r < K and 0 <= c < R_b.shape[1]:
                        perm[int(r)] = int(c)
            else:
                # Greedy fallback: for each ref i, pick nearest unmatched boot j
                perm = np.empty(K, dtype=int)
                unused = set(range(R_b.shape[1]))
                for i in range(K):
                    j_best = min(unused, key=lambda j: costM[i, j])
                    perm[i] = j_best
                    unused.remove(j_best)
            # Reorder columns to match reference ordering
            R_b = R_b[:, perm]

        # If some bag produced fewer/more columns (shouldn't happen if K fixed), coerce width
        K_run = R_b.shape[1]
        if K_run < K:
            R_b = np.hstack([R_b, np.zeros((N, K - K_run), dtype=float)])
        elif K_run > K:
            R_b = R_b[:, :K]

        prob_sum += R_b
        hard_b = np.argmax(R_b, axis=1)
        vote_cnt[np.arange(N), hard_b] += 1

        if log_each:
            logging.info("Bootstrap-GMM [%d/%d] done.", b + 1, B)

    # --------------------- Aggregate + stability ---------------------
    proba_mean = _normalize_probability_rows(prob_sum / float(B))
    modal_topic = vote_cnt.argmax(axis=1)
    modal_vote = vote_cnt.max(axis=1) / float(B)

    # Entropy & normalized entropy
    P = np.clip(proba_mean, 1e-15, 1.0)
    ent = -np.sum(P * np.log(P), axis=1)
    ent_max = np.log(K) if K > 1 else 1.0
    norm_ent = ent / ent_max

    agg_topic_proba_df = pd.DataFrame(proba_mean, index=index, columns=[f"topic_{i}" for i in range(K)])
    stability_df = pd.DataFrame(
        {'modal_topic': modal_topic.astype(int),
         'modal_vote': modal_vote.astype(float),
         'entropy': ent.astype(float),
         'norm_entropy': norm_ent.astype(float)},
        index=index
    )

    return agg_topic_proba_df, stability_df

# ---------------------------- HDBSCAN SUPPORT --------------------------------

def _hdbscan_memberships(hdb: "hdbscan.HDBSCAN",
                         X: Union[np.ndarray, pd.DataFrame]
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (labels, U) where U is (N, K>=1) soft-membership matrix.

    - Uses hdbscan.prediction.all_points_membership_vectors when available.
    - Guarantees U has at least 1 column, is finite, and row-normalized.
    """
    if isinstance(X, pd.DataFrame):
        Xv = X.values
    else:
        Xv = np.asarray(X)

    try:
        import hdbscan.prediction as hdb_pred
        U = hdb_pred.all_points_membership_vectors(hdb, Xv)  # (N, K?) can be empty/1-D
        labels = hdb.labels_
    except Exception:
        labels = getattr(hdb, 'labels_', np.full(Xv.shape[0], -1, dtype=int))
        U = np.zeros((Xv.shape[0], 0), dtype=float)

    U = np.asarray(U)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if U.size == 0 or U.shape[1] == 0:
        U = np.ones((Xv.shape[0], 1), dtype=float)

    U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    rs = U.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    U = U / rs
 
    # warn if HDBSCAN produced only one set of probabilities
    try:
        k = int(U.shape[1])
        uniq = np.unique(labels)
        n_nonnoise = int(np.sum(uniq >= 0))  # clusters excluding noise (-1)
        if (k <= 1) or (n_nonnoise <= 1):
            logging.warning(
                "HDBSCAN produced a single probability column (K=%d; non-noise clusters=%d). "
                "This indicates the clustering did not separate the data (e.g., all noise or one dense group). "
                "Downstream will treat this as a single family. Consider tuning min_cluster_size/min_samples/"
                "metric or applying preprocessing (e.g., PCA).",
                k, n_nonnoise
            )
    except Exception:
        # be fail-safe: never block return path
        pass
    
    return labels, U

def compute_hdbscan_topics(X_source: pd.DataFrame,
                           **params
                           ) -> Tuple[List[int], pd.DataFrame, "hdbscan.HDBSCAN"]:
    """
    HDBSCAN clustering with soft membership extraction on training data.
    Returns hard labels (may include -1 for noise) and membership matrix U (N×K).

    Notes
    -----
    - Ensures U is always 2-D and has at least 1 column (fallback column of ones)
      so downstream soft-routing never sees an empty probability table.
    - Replaces NaNs with 0 and row-normalizes; rows with zero sum become uniform.
    - Aligns returned labels with proba_df columns (no negatives or out-of-range).
    """
    if not HAVE_HDBSCAN:
        raise ImportError("HDBSCAN is required (pip install hdbscan).")

    # ------- override from **params (e.g., full config) when provided ----------
    # NOTE: keep explicit args as defaults; **params takes precedence.
    def _coerce_none(x):
        return None if (x is None or (isinstance(x, str) and x.strip().lower() == 'none')) else x

    min_cluster_size = int(params.get('min_cluster_size', 10))
    min_samples_in   = params.get('min_samples', None)
    min_samples      = _coerce_none(min_samples_in)
    
    if min_samples is not None:
        min_samples = int(min_samples)

    metric = str(params.get('metric', 'euclidean'))
    cluster_selection_epsilon = float(params.get('cluster_selection_epsilon', 0.0))
    prediction_data = bool(params.get('prediction_data', True))
    random_state = int(params.get('random_state', 42))
    core_dist_n_jobs = int(params.get('core_dist_n_jobs', 1))
    # --------------------------------------------------------------------------

    # Seed RNG for any stochastic parts (parity)
    try:
        np.random.seed(random_state)
    except Exception:
        pass

    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=prediction_data,
        core_dist_n_jobs=core_dist_n_jobs
    ).fit(X_source.values)

    labels, U = _hdbscan_memberships(hdb, X_source.values)   # (N, K>=1)
    K = U.shape[1]
    proba_df = pd.DataFrame(U, index=X_source.index, columns=[f"topic_{i}" for i in range(K)])
    return labels.tolist(), proba_df, hdb

def bootstrap_hdbscan_topics(X_full: Union[np.ndarray, pd.DataFrame],
                             B: int,
                             sample_frac: float,
                             hdb_kwargs: Dict[str, Any],
                             base_hdb: Optional["hdbscan.HDBSCAN"] = None,
                             log_each: bool = False,
                             random_state: int = 42
                             ) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """
    Bootstrap HDBSCAN on X_full. Returns:
      (agg_topic_proba_df | None, stability_df)
    where stability_df has ['modal_topic','modal_vote','entropy','norm_entropy'].

    Behavior:
      • Uses/creates a reference HDBSCAN (prediction_data=True) on full data.
      • Always builds hard-label votes as (B, N) and tallies per sample (fixes index errors).
      • If hdb_kwargs['return_agg'] is True, aligns each bootstrap run to the reference
        cluster space (Hungarian if SciPy is available; greedy otherwise) and averages
        one-hot votes into an N×K_ref probability table.
    """
    if not HAVE_HDBSCAN:
        raise ImportError("HDBSCAN is required (pip install hdbscan).")

    rng = np.random.default_rng(int(random_state))
    is_df = isinstance(X_full, pd.DataFrame)
    X_arr = X_full.values if is_df else np.asarray(X_full)
    index = X_full.index if is_df else pd.RangeIndex(X_arr.shape[0])
    N = X_arr.shape[0]

    # Extract kwargs with fallbacks
    min_cluster_size      = int(hdb_kwargs.get('min_cluster_size', 10))
    min_samples_in        = hdb_kwargs.get('min_samples', None)
    min_samples           = None if (min_samples_in in (None, 'None')) else int(min_samples_in)
    metric                = str(hdb_kwargs.get('metric', 'euclidean'))
    eps                   = float(hdb_kwargs.get('cluster_selection_epsilon', 0.0))
    prediction_data       = bool(hdb_kwargs.get('prediction_data', True))
    core_dist_n_jobs      = int(hdb_kwargs.get('core_dist_n_jobs', 1))
    return_agg            = bool(hdb_kwargs.get('return_agg', False))


    # Reference model (ensure prediction_data=True)
    hdb_ref = None
    if base_hdb is not None and bool(getattr(base_hdb, 'prediction_data', False)):
        hdb_ref = base_hdb
    else:
        hdb_ref = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=eps,
            prediction_data=True,
            core_dist_n_jobs=core_dist_n_jobs
        ).fit(X_arr)

    # Reference memberships (N x K_ref >= 1)
    _, U_ref = _hdbscan_memberships(hdb_ref, X_arr)
    K_ref = U_ref.shape[1]
    

    sample_n = max(1, int(round(float(sample_frac) * N)))
    sample_replace = _bootstrap_sample_with_replacement(hdb_kwargs.get('bootstrap_sampling_mode', 'bootstrap'))
    if not sample_replace:
        sample_n = min(sample_n, N)
    prob_sum = np.zeros((N, K_ref), dtype=float)
    vote_cnt = np.zeros((N, K_ref), dtype=int)
    w_cnt    = np.zeros((N,), dtype=int)
    ent_sum  = np.zeros((N,), dtype=float)

    
    for b in range(int(max(1, B))):
        idx = rng.choice(N, size=sample_n, replace=sample_replace)   # ROW positions
        X_b = X_arr[idx, :]

        hdb_b = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=eps,
            prediction_data=prediction_data,
            core_dist_n_jobs=core_dist_n_jobs
        ).fit(X_b)

        # Prefer out-of-bag/full-dataset membership prediction so each bootstrap bag contributes
        # to every reaction, matching the LDA/GMM/QDA/FCM stability semantics. Fall back to
        # in-bag memberships if HDBSCAN approximate prediction is unavailable for a run.
        try:
            import hdbscan.prediction as _hdb_pred
            U_b = _hdb_pred.membership_vector(hdb_b, X_arr)
            U_b = _normalize_probability_rows(U_b)
            align_positions = np.arange(N)
        except Exception:
            _, U_b = _hdbscan_memberships(hdb_b, X_b)  # (sample_n, K_run>=1)
            U_b = _normalize_probability_rows(U_b)
            align_positions = idx
        # align current run's columns to the reference via Hungarian assignment on cosine similarity
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from scipy.optimize import linear_sum_assignment

            K_run = U_b.shape[1]
            # reference memberships on the same sampled rows
            U_ref_bag = U_ref[align_positions, :]  # rows used for alignment

            # similarity between columns (topics): K_run x K_ref
            S = cosine_similarity(U_b.T, U_ref_bag.T)  # columns-as-vectors similarity
            # maximize similarity -> minimize cost
            cost = 1.0 - np.clip(S, 0.0, 1.0)

            # Solve assignment (handles rectangular cost)
            if cost.shape[0] <= cost.shape[1]:
                r_ind, c_ind = linear_sum_assignment(cost)
            else:
                r_ind, c_ind = linear_sum_assignment(cost.T)
                # swap back to (run, ref) indexing
                r_ind, c_ind = c_ind, r_ind

            # build aligned matrix with exactly K_ref columns
            U_b_aligned = np.zeros((U_b.shape[0], K_ref), dtype=float)
            for j_run, j_ref in zip(r_ind, c_ind):
                if j_run < K_run and j_ref < K_ref:
                    U_b_aligned[:, j_ref] = U_b[:, j_run]
            U_b = U_b_aligned
        except Exception as _e:
            # fallback: previous pad/truncate behavior
            K_run = U_b.shape[1]
            if K_run < K_ref:
                U_b = np.hstack([U_b, np.zeros((U_b.shape[0], K_ref - K_run), dtype=float)])
            elif K_run > K_ref:
                U_b = U_b[:, :K_ref]

        row_ent = -(np.clip(U_b, 1e-12, 1.0) * np.log(np.clip(U_b, 1e-12, 1.0))).sum(axis=1)

        hard_b = np.argmax(U_b, axis=1)
        for j, pos in enumerate(align_positions):
            vote_cnt[pos, hard_b[j]] += 1
            prob_sum[pos, :] += U_b[j, :]
            w_cnt[pos] += 1
            ent_sum[pos] += row_ent[j]

        if log_each:
            logging.info("Bootstrap-HDBSCAN [%d/%d]: K_run=%d", b + 1, B, K_run)

    with np.errstate(divide='ignore', invalid='ignore'):
        prob_avg = np.divide(prob_sum, w_cnt[:, None], out=np.zeros_like(prob_sum), where=w_cnt[:, None] > 0)

    modal_topic = vote_cnt.argmax(axis=1)
    modal_vote  = np.max(vote_cnt, axis=1) / np.clip(w_cnt.astype(float), 1.0, None)

    eps = 1e-12
    P = np.clip(prob_avg, eps, 1.0)
    ent = -(P * np.log(P)).sum(axis=1)
    norm_ent = ent / np.log(max(K_ref, 2))

    with np.errstate(divide='ignore', invalid='ignore'):
        ent_mean = np.divide(ent_sum, w_cnt, out=np.zeros_like(ent_sum), where=w_cnt > 0)
    ent_mean_norm = ent_mean / np.log(max(K_ref, 2))

    agg_df = pd.DataFrame(_normalize_probability_rows(prob_avg), index=index, columns=[f"topic_{i}" for i in range(K_ref)])
    stab_df = pd.DataFrame(
        {
            'modal_topic': modal_topic,
            'modal_vote':  modal_vote,
            'entropy':     ent,
            'norm_entropy': norm_ent,
            'entropy_mean': ent_mean,
            'norm_entropy_mean': ent_mean_norm,
        },
        index=index
    )

    logging.info("Bootstrap-HDBSCAN finished. Median modal_vote=%.3f; median norm_entropy=%.3f",
                 float(np.nanmedian(modal_vote)), float(np.nanmedian(norm_ent)))
    return agg_df, stab_df

# ---------------------------- QDA SUPPORT ------------------------------------

class AutoChooserSkipCandidate(RuntimeError):
    """Raised when an Auto-Chooser candidate is scientifically/algorithmically infeasible.

    This is not a pipeline crash; Auto-Chooser should record the candidate as SKIPPED
    and continue evaluating the remaining candidates.
    """
    pass

def _qda_feasibility_report(base_labels: Union[List[int], np.ndarray],
                            *,
                            min_class_count: int = 2,
                            n_classes: Optional[int] = None) -> Dict[str, Any]:
    """Return a data-dependent feasibility report for fitting QDA on base family labels.

    QDA must estimate one covariance matrix per present class. Any present class with
    fewer than two samples is mathematically invalid for covariance estimation; in
    practice, a larger threshold (e.g. 5 or 10) is more stable for Auto-Chooser.
    Missing classes among 0..K-1 are tolerated because predict_proba is expanded later.
    """
    y = np.asarray(base_labels).astype(int)
    vals, cnts = np.unique(y, return_counts=True)
    counts = {int(v): int(c) for v, c in zip(vals, cnts)}
    min_count = int(max(2, min_class_count))
    min_present = int(np.min(cnts)) if cnts.size else 0
    n_present = int(vals.size)
    missing = []
    if n_classes is not None:
        present_set = set(int(v) for v in vals.tolist())
        missing = [int(k) for k in range(int(n_classes)) if int(k) not in present_set]
    feasible = (n_present >= 2) and (min_present >= min_count)
    return {
        'feasible': bool(feasible),
        'n_present_classes': n_present,
        'min_present_class_count': min_present,
        'required_min_class_count': min_count,
        'class_counts': counts,
        'missing_classes': missing,
    }

def _format_qda_feasibility_message(report: Dict[str, Any]) -> str:
    return (
        "QDA infeasible before fit: "
        f"n_present_classes={report.get('n_present_classes')}, "
        f"min_present_class_count={report.get('min_present_class_count')}, "
        f"required_min_class_count={report.get('required_min_class_count')}, "
        f"class_counts={report.get('class_counts')}, "
        f"missing_classes={report.get('missing_classes')}. "
        "Reduce n_topics, merge/avoid rare LDA families, or remove this QDA candidate."
    )

def compute_qda_topics(
    X_source: Union[pd.DataFrame, np.ndarray],
    base_labels: Union[List[int], np.ndarray],
    *,
    n_classes: int,
    reg_param: float = 0.0,
    priors: Optional[List[float]] = None,
    store_covariance: bool = False,
    tol: float = 1e-4,
    min_class_count: int = 2,
) -> Tuple[List[int], pd.DataFrame, QuadraticDiscriminantAnalysis]:
    """
    Train QDA as a supervised family assigner and return hard labels + soft probabilities.

    Inputs
    ------
    X_source : features to feed QDA (N×D) — either LDA doc-topic probs or DRFP(+PCA)
    base_labels : list/array of length N — integer family labels used to train QDA
    n_classes : target number of families (K), usually config['n_topics']
    reg_param : shrinkage in [0,1] to stabilize covariance inversion
    priors : length-K class priors or None for empirical
    store_covariance : keep per-class covariance matrices in the model

    Returns
    -------
    topics_qda : list[int] length N
    proba_df   : DataFrame N×K with class probability columns 0..K-1
    qda        : fitted QuadraticDiscriminantAnalysis model
    """
    if isinstance(X_source, pd.DataFrame):
        X = X_source.values
    else:
        X = np.asarray(X_source)

    y = np.asarray(base_labels).astype(int)
    qda_report = _qda_feasibility_report(y, min_class_count=int(min_class_count), n_classes=int(n_classes))
    if not qda_report['feasible']:
        raise ValueError(_format_qda_feasibility_message(qda_report))

    # If some classes 0..K-1 are missing in y, QDA will have fewer classes_;
    # We'll expand predict_proba output to full K with zeros in missing columns.
    qda = QuadraticDiscriminantAnalysis(
        reg_param=float(reg_param),
        priors=(None if priors in (None, 'None') else np.asarray(priors, dtype=float)),
        store_covariance=bool(store_covariance), tol=float(tol)
    )
    qda.fit(X, y)

    # Probabilities for the classes that QDA actually saw:
    proba_seen = qda.predict_proba(X)                 # N×|classes_|
    classes_seen = qda.classes_.astype(int).tolist()  # e.g., [0,2,3,...]

    # Expand to N×K aligned by class index
    K = int(n_classes)
    proba_full = np.zeros((X.shape[0], K), dtype=float)
    for col_idx, cls in enumerate(classes_seen):
        if 0 <= cls < K:
            proba_full[:, cls] = proba_seen[:, col_idx]

    topics_qda = proba_full.argmax(axis=1).astype(int).tolist()
    proba_df = pd.DataFrame(proba_full, columns=list(range(K)))
    return topics_qda, proba_df, qda

def bootstrap_qda_topics(
    X_source: Union[pd.DataFrame, np.ndarray],
    base_labels: Union[List[int], np.ndarray],
    *,
    n_classes: int,
    B: int = 25,
    sample_frac: float = 0.8,
    reg_param: float = 0.0,
    priors: Optional[List[float]] = None,
    store_covariance: bool = False,
    random_state: int = 42,
    tol: float=1e-4,
    sampling_mode: str = 'bootstrap',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bag QDA on resamples of (X_source, base_labels), aggregate mean probabilities,
    and compute stability metrics (modal vote, entropy, normalized entropy).

    Returns
    -------
    agg_topic_proba_df : DataFrame (N×K) of mean predicted probabilities over B bags
    stability_df       : DataFrame with columns: ['modal_topic','modal_vote','entropy','norm_entropy']
    """
    # Prepare arrays
    if isinstance(X_source, pd.DataFrame):
        X = X_source.values
        index = list(X_source.index)
    else:
        X = np.asarray(X_source)
        index = list(range(X.shape[0]))
    y = np.asarray(base_labels).astype(int)
    N = X.shape[0]
    K = int(n_classes)

    rng = np.random.RandomState(int(random_state))
    proba_sum = np.zeros((N, K), dtype=float)
    votes = np.zeros((N, K), dtype=int)

    # Helper: fit QDA for a bootstrap sample and return proba on full X (expanded to K classes)
    def _fit_predict_qda(ix_boot: np.ndarray) -> np.ndarray:
        qda = QuadraticDiscriminantAnalysis(
            reg_param=float(reg_param),
            priors=(None if priors in (None, 'None') else np.asarray(priors, dtype=float)),
            store_covariance=bool(store_covariance), tol=float(tol)
        )
        qda.fit(X[ix_boot], y[ix_boot])

        proba_seen = qda.predict_proba(X)                 # N × |classes_|
        classes_seen = qda.classes_.astype(int)

        # Expand to N×K aligned by class index
        proba_full = np.zeros((N, K), dtype=float)
        for c_idx, cls in enumerate(classes_seen):
            if 0 <= cls < K:
                proba_full[:, cls] = proba_seen[:, c_idx]
        return proba_full

    m = max(1, int(math.ceil(sample_frac * N)))
    sample_replace = _bootstrap_sample_with_replacement(sampling_mode)
    if not sample_replace:
        m = min(m, N)

    B_req = int(max(1, B))
    B_eff = 0
    for b in range(B_req):
        ix_boot = None
        # QDA requires at least two classes and numerically benefits from >=2 samples/class.
        for _attempt in range(100):
            cand = rng.choice(N, size=m, replace=sample_replace)
            vals, cnts = np.unique(y[cand], return_counts=True)
            if vals.size >= 2 and np.min(cnts) >= 2:
                ix_boot = cand
                break
        if ix_boot is None:
            ix_boot = np.arange(N)
        try:
            proba_b = _normalize_probability_rows(_fit_predict_qda(ix_boot))
        except Exception as e:
            logging.warning("Bootstrap-QDA bag %d/%d failed and was skipped: %s", b + 1, B_req, e)
            continue
        proba_sum += proba_b
        argb = np.argmax(proba_b, axis=1)
        votes[np.arange(N), argb] += 1
        B_eff += 1

    if B_eff <= 0:
        logging.warning("Bootstrap-QDA produced no valid bags; falling back to deterministic QDA probabilities.")
        _, fallback_df, _ = compute_qda_topics(X_source, base_labels, n_classes=K, reg_param=reg_param,
                                               priors=priors, store_covariance=store_covariance, tol=tol,
                                               min_class_count=2)
        proba_sum = fallback_df.values
        votes[np.arange(N), np.argmax(proba_sum, axis=1)] = 1
        B_eff = 1

    proba_mean = _normalize_probability_rows(proba_sum / float(B_eff))
    # Stability metrics
    modal_topic = votes.argmax(axis=1)
    modal_vote = votes.max(axis=1) / float(B_eff)

    # Entropy (natural log); normalized by log(K)
    with np.errstate(divide='ignore', invalid='ignore'):
        # clip to avoid log(0)
        P = np.clip(proba_mean, 1e-15, 1.0)
        ent = -np.sum(P * np.log(P), axis=1)
    ent_max = math.log(K) if K > 1 else 1.0
    norm_entropy = (ent / ent_max) if ent_max > 0 else np.zeros_like(ent)

    agg_topic_proba_df = pd.DataFrame(proba_mean, index=index, columns=list(range(K)))
    stability_df = pd.DataFrame({
        'modal_topic': modal_topic.astype(int),
        'modal_vote': modal_vote.astype(float),
        'entropy': ent.astype(float),
        'norm_entropy': norm_entropy.astype(float),
    }, index=index)

    return agg_topic_proba_df, stability_df

# ---------------------------- FCM SUPPORT ------------------------------------

def fit_pca_on_drfp(drfp_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[PCA]]:
    """
    Optional PCA projection for DRFP.

    Uses either a fixed number of components or the explained-variance target.

    Returns:
        (X_pca_df, pca_model or None)
    """
    if not bool(config.get('pca_enable', False)):
        return drfp_df, None

    n_comp = config.get('pca_n_components', None)
    var_goal = float(config.get('pca_var', 0.95))
    whiten = bool(config.get('pca_whiten', False))
    pca_rs = config.get('pca_random_state', config.get('random_state', 42))

    if n_comp is None or int(n_comp) <= 0:
        pca = PCA(n_components=var_goal, whiten=whiten, random_state=pca_rs)
    else:
        pca = PCA(n_components=int(n_comp), whiten=whiten, random_state=pca_rs)

    X = drfp_df.values if isinstance(drfp_df, pd.DataFrame) else np.asarray(drfp_df)

    #Xp = pca.fit_transform(drfp_df.values)
    Xp = pca.fit_transform(X)
    cols = [f"drfp_pca_{i}" for i in range(Xp.shape[1])]

    #X_pca_df = pd.DataFrame(Xp, index=drfp_df.index, columns=cols)
    X_pca_df = pd.DataFrame(Xp, index=(drfp_df.index if isinstance(drfp_df, pd.DataFrame) else None), columns=cols)
    
    logging.info("PCA: %s -> %s dims | explained_var=%.3f",
                 drfp_df.shape[1], X_pca_df.shape[1],
                 float(np.sum(pca.explained_variance_ratio_)))
    return X_pca_df, pca

def compute_fcm_topics(X_source: pd.DataFrame,
                       n_clusters: int,
                       m: float = 1.5,
                       max_iter: int = 300,
                       error: float = 1e-6,
                       random_state: int = 42) -> Tuple[List[int], pd.DataFrame, np.ndarray]:
    """
    Fuzzy C-Means soft clustering.

    Args:
        X_source: DataFrame [N × D], features to cluster (DRFP or LDA topic space)
        n_clusters: number of clusters (families)
        m: fuzzifier (>1); lower ~ crisper (k-means-like), higher ~ softer memberships
        max_iter: max iterations for cmeans
        error: tolerance for convergence
        random_state: RNG seed (controls initialization)

    Returns:
        hard_labels: List[int], argmax memberships per sample
        proba_df: DataFrame [N × K], membership matrix per sample (rows sum to 1)
        centers: np.ndarray [K × D], cluster centers in feature space
    """
    if not HAVE_FCM:
        raise ImportError("scikit-fuzzy is required for topic_method='fcm' (pip install scikit-fuzzy).")

    X = np.asarray(X_source.values, dtype=float).T  # skfuzzy expects (D, N)
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X, c=int(n_clusters), m=float(m), error=float(error), maxiter=int(max_iter),
        init=None, seed=int(random_state)
    )
    U = u.T  # (N, K)
    hard = U.argmax(axis=1).tolist()
    proba_df = pd.DataFrame(U, index=X_source.index, columns=[f"topic_{i}" for i in range(n_clusters)])
    return hard, proba_df, cntr

def bootstrap_fcm_topics(X_full: Union[np.ndarray, pd.DataFrame],
                         B: int,
                         sample_frac: float,
                         k: int = 6,
                         m: float = 1.1,
                         max_iter: int = 300,
                         error: float = 1e-6,
                         align: bool = True,
                         log_each: bool = False,
                         random_state: int = 42,
                         sampling_mode: str = 'bootstrap'
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap FCM on X_full and return:
      1) agg_topic_proba_df: (N x K) averaged soft-membership across B bootstraps
      2) stability_df: DataFrame with ['modal_topic','modal_vote','entropy','norm_entropy']

    Notes
    -----
    - Fixes scikit-fuzzy cmeans_predict unpack: it returns SIX values.
    - Robust to pandas.DataFrame inputs (row sampling uses .iloc and algorithms get NumPy).
    - Ensures memberships are always 2-D (N x K), rows sum to 1; zero-sum rows become uniform.
    - Optional alignment of per-bootstrap columns to reference using center matching
      (Hungarian if SciPy available, otherwise a greedy fallback).
    """
    if not HAVE_FCM:
        raise ImportError("scikit-fuzzy is required for FCM bootstrap (pip install scikit-fuzzy).")

    rng = np.random.default_rng(int(random_state))

    # Robust handling for DataFrame vs ndarray inputs
    is_df = isinstance(X_full, pd.DataFrame)
    X_arr = X_full.to_numpy() if is_df else np.asarray(X_full)
    df_index = X_full.index if is_df else None

    n = int(X_arr.shape[0])
    if n == 0:
        return (pd.DataFrame(index=df_index), pd.DataFrame(index=df_index))

    # --- Guards for B and sample size
    B = max(0, int(B))
    sf = float(sample_frac)
    if not np.isfinite(sf):
        sf = 0.8
    sf = min(1.0, max(1e-12, sf))
    msize = int(round(sf * n))
    sample_replace = _bootstrap_sample_with_replacement(sampling_mode)
    if sample_replace:
        msize = max(1, msize)
    else:
        msize = max(1, min(n, msize))

    # --- Helper: normalize memberships to (N x K), K>=1, rows sum to 1
    def _normalize_U(U: np.ndarray) -> np.ndarray:
        U = np.asarray(U)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        if U.size == 0 or U.shape[1] == 0:
            U = np.ones((n, 1), dtype=float)
        U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        rs = U.sum(axis=1, keepdims=True)
        zero = (rs == 0.0)
        if np.any(zero):
            U[zero.ravel(), :] = 1.0 / U.shape[1]
            rs = U.sum(axis=1, keepdims=True)
        return U / rs

    # --- Reference FCM on full data
    # cmeans returns: cntr, u (K x N), u0, d, jm, p, fpc
    cntr_ref, U_ref_T, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_arr.T, c=int(k), m=float(m), error=float(error), maxiter=int(max_iter),
        init=None, seed=int(random_state)
    )
    U_ref = _normalize_U(U_ref_T.T)  # (N, K)
    topics_ref = U_ref.argmax(axis=1).astype(int)

    # Accumulators
    U_sum = np.zeros_like(U_ref, dtype=float)
    label_votes: List[np.ndarray] = []

    # Alignment preparation
    try:
        from scipy.optimize import linear_sum_assignment as hungarian
        have_hungarian = True
    except Exception:
        have_hungarian = False

    def _align_columns_by_centers(cntr_run: np.ndarray, cntr_ref_: np.ndarray) -> List[int]:
        """Map run columns to reference columns using center distances (Hungarian if available)."""
        cntr_run = np.asarray(cntr_run)
        cntr_ref_ = np.asarray(cntr_ref_)
        D = np.linalg.norm(cntr_run[:, None, :] - cntr_ref_[None, :, :], axis=2)  # (K_run x K_ref)
        K_run, K_ref = D.shape
        if K_run == 0 or K_ref == 0:
            return list(range(K_run))
        if have_hungarian:
            r_idx, c_idx = hungarian(D)
            perm = [-1] * K_run
            for r, c in zip(r_idx, c_idx):
                if 0 <= r < K_run and 0 <= c < K_ref:
                    perm[r] = int(c)
            used = set([c for c in perm if c >= 0])
            for r in range(K_run):
                if perm[r] < 0:
                    order = np.argsort(D[r, :])
                    for c in order:
                        if int(c) not in used:
                            perm[r] = int(c); used.add(int(c)); break
            return perm
        # Greedy fallback
        perm = [-1] * K_run
        used = set()
        for r in range(K_run):
            order = np.argsort(D[r, :])
            pick = None
            for c in order:
                if int(c) not in used:
                    pick = int(c); break
            if pick is None:
                pick = int(order[0])
            perm[r] = pick
            used.add(pick)
        return perm

    # --- Bootstrap loop
    for b in range(B):
        idx = rng.choice(n, size=msize, replace=sample_replace)
        X_b = X_arr[idx]  # NumPy slice (row sampling)

        # Fit FCM on bootstrap sample
        # returns: cntr_b, u_b (K x m), u0, d, jm, p, fpc
        cntr_b, U_b_T, u0_b, d_b, jm_b, p_b, fpc_b = fuzz.cluster.cmeans(
            X_b.T, c=int(k), m=float(m), error=float(error), maxiter=int(max_iter),
            init=None, seed=int(random_state + b + 1)
        )

        # Predict memberships for ALL rows against bootstrap centers
        # cmeans_predict returns SIX values: u_pred (K x N), u0, d, jm, p, fpc
        u_pred, u0p, dp, jmp, pp, fpcp = fuzz.cluster.cmeans_predict(
            X_arr.T, cntr_b, m=float(m), error=float(error), maxiter=int(max_iter), init=None
        )
        U_run = _normalize_U(u_pred.T)  # (N, K)  <-- fixed unpack + transpose

        # Optional alignment to reference
        if align:
            perm = _align_columns_by_centers(np.asarray(cntr_b), np.asarray(cntr_ref))
            if len(perm) == U_run.shape[1]:
                U_aligned = np.zeros_like(U_run)
                for r_col, ref_col in enumerate(perm):
                    if 0 <= ref_col < U_run.shape[1]:
                        U_aligned[:, ref_col] = U_run[:, r_col]
                U_run = _normalize_U(U_aligned)

        U_sum += U_run
        label_votes.append(U_run.argmax(axis=1).astype(int))

        if log_each:
            logging.info("Bootstrap-FCM [%d/%d]: K=%d (aligned=%s)", b + 1, B, U_run.shape[1], str(bool(align)))

    # --- Aggregate memberships
    U_avg = _normalize_probability_rows((U_sum / float(B)) if B > 0 else U_ref.copy())

    # --- Stability from hard labels across bootstraps
    if len(label_votes) == 0:
        LV = topics_ref[:, None]
    else:
        LV = np.stack(label_votes, axis=1)  # (N, B)

    modal_topic = []
    modal_vote_frac = []
    entropy = []
    norm_entropy = []
    for i in range(n):
        vals, counts = np.unique(LV[i, :], return_counts=True)
        p = counts / counts.sum()
        j = int(vals[np.argmax(counts)])
        modal_topic.append(j)
        modal_vote_frac.append(float(p.max()))
        ent = -float((p * np.log(p + 1e-12)).sum())
        K_eff = max(1, len(vals))
        ent_norm = 0.0 if K_eff <= 1 else float(ent / np.log(K_eff))
        entropy.append(ent)
        norm_entropy.append(ent_norm)

    stability_df = pd.DataFrame({
        'modal_topic':  modal_topic,
        'modal_vote':   modal_vote_frac,   # fraction in [0,1]
        'entropy':      entropy,
        'norm_entropy': norm_entropy
    }, index=df_index)

    # --- Summary log (match original message)
    modal_vote = modal_vote_frac
    norm_ent   = norm_entropy
    try:
        logging.info(
            "Bootstrap-FCM finished. Median modal_vote=%.3f; median norm_entropy=%.3f",
            float(np.median(modal_vote)),
            float(np.median(norm_ent))
        )
    except Exception:
        pass

    agg_topic_proba_df = pd.DataFrame(
        U_avg, columns=[f"topic_{i}" for i in range(U_avg.shape[1])], index=df_index
    )
    return agg_topic_proba_df, stability_df

# ------------------------------ DESCRIPTORS ----------------------------------

def _mordred_cache_key(
    reac_series: pd.Series,
    agg: str,
    nan_policy: str,
    use3d: Optional[bool] = None,
    xtqspr_explicit_h_patch: bool = False,
) -> str:
    """
    Construct a stable Mordred descriptor cache key.

    The key depends on:
      - reaction SMILES values,
      - reaction index/order,
      - Mordred aggregation mode,
      - missing-value policy,
      - 2D/3D Mordred mode,
      - XTQSPRKinetic explicit-H Mordred runtime patch mode.

    Parallel Mordred settings are intentionally not included because they should
    change speed only, not descriptor values.
    """
    if use3d is None:
        use3d = os.environ.get("XT_MORDRED_3D", "").strip().lower() in (
            "1", "true", "yes", "on"
        )

    payload = {
        "reactions": [str(x) for x in reac_series.tolist()],
        "index": [str(x) for x in reac_series.index.tolist()],
        "count": int(len(reac_series)),
        "agg": str(agg or "sum").lower(),
        "nan_policy": str(nan_policy or "mean").lower(),
        "mordred_3d": bool(use3d),
        "mordred_xtqspr_explicit_h_patch": bool(xtqspr_explicit_h_patch),
    }

    text = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return _sha1(text)[:16]

def get_or_compute_mordred(
    reac_series: pd.Series,
    desc_nan_policy: str,
    agg: str,
    use_cache: bool,
    cache_dir: str,
    mordred_parallel_enable: bool = False,
    mordred_parallel_n_jobs: int = 1,
    mordred_parallel_backend: str = 'loky',
    mordred_parallel_batch_size: Union[int, str] = 'auto',
    mordred_xtqspr_explicit_h_patch_enable: bool = False,
) -> pd.DataFrame:
    """
    Compute Mordred descriptors with optional on-disk caching.

    Parameters
    ----------
    reac_series:
        Series of reaction SMILES.
    desc_nan_policy:
        Missing-value policy for Mordred descriptors.
    agg:
        Mordred aggregation mode: {'sum','concat','delta'}.
    use_cache:
        If True, load/write descriptor cache.
    cache_dir:
        Directory for descriptor cache.
    mordred_parallel_enable:
        If True, compute reaction-level Mordred rows in parallel.
    mordred_parallel_n_jobs:
        Number of parallel jobs. -1 uses all available CPUs.
    mordred_parallel_backend:
        Joblib backend: {'loky','multiprocessing','threading'}.
    mordred_parallel_batch_size:
        Joblib batch size. Usually 'auto'.

    Returns
    -------
    pd.DataFrame
        Mordred descriptor matrix.
    """
    if not use_cache:
        logging.info("Descriptor cache disabled; computing Mordred from scratch.")
        df = calculate_descriptors_mordred(
            reac_series,
            desc_nan_policy,
            agg=agg,
            parallel_enable=bool(mordred_parallel_enable),
            n_jobs=int(mordred_parallel_n_jobs),
            backend=str(mordred_parallel_backend),
            batch_size=mordred_parallel_batch_size,
            xtqspr_explicit_h_patch_enable=bool(mordred_xtqspr_explicit_h_patch_enable),
        )
        return df

    _ensure_dir(cache_dir)
    key = _mordred_cache_key(
    reac_series,
    agg,
    desc_nan_policy,
    xtqspr_explicit_h_patch=bool(mordred_xtqspr_explicit_h_patch_enable))
    cache_path = os.path.join(cache_dir, f"mordred_{key}.parquet")
    meta_path = os.path.join(cache_dir, f"mordred_{key}.json")

    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            logging.info("Loaded Mordred cache: %s", cache_path)
            return df
        except Exception as e:
            logging.warning("Failed to load Mordred cache (%s); recomputing. Error: %s", cache_path, e)

    t0 = time.time()
    df = calculate_descriptors_mordred(
        reac_series,
        desc_nan_policy,
        agg=agg,
        parallel_enable=bool(mordred_parallel_enable),
        n_jobs=int(mordred_parallel_n_jobs),
        backend=str(mordred_parallel_backend),
        batch_size=mordred_parallel_batch_size,
        xtqspr_explicit_h_patch_enable=bool(mordred_xtqspr_explicit_h_patch_enable),
    )

    try:
        df.to_parquet(cache_path, index=True)
        meta = {
            "created": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "agg": agg,
            "nan_policy": desc_nan_policy,
            "parallel_enable": bool(mordred_parallel_enable),
            "parallel_n_jobs": int(mordred_parallel_n_jobs),
            "parallel_backend": str(mordred_parallel_backend),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "mordred_xtqspr_explicit_h_patch_enable": bool(mordred_xtqspr_explicit_h_patch_enable),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logging.info("Saved Mordred cache: %s (%.2fs).", cache_path, time.time() - t0)
    except Exception as e:
        logging.warning("Could not save Mordred cache (%s): %s", cache_path, e)

    return df

# ---------------- HYDROGEN-RADICAL SPECIAL-CASE DESCRIPTORS ------------------

def _split_reaction_sides_for_hydrogen_patch(rxn_smiles: str) -> Tuple[str, str]:
    """
    Return (reactant_side, product_side) for both R>>P and R>agents>P forms.
    Used only for the hydrogen-radical special-case patch.
    """
    text = str(rxn_smiles).strip()
    parts = text.split(">")

    if len(parts) == 3:
        return parts[0].strip(), parts[2].strip()

    if ">>" in text:
        left, right = text.split(">>", 1)
        return left.strip(), right.strip()

    return text, ""


def _mol_from_smiles_keep_explicit_h(smi: str):
    """
    Parse a species SMILES while preserving explicit hydrogens.

    Important for:
        [H]      hydrogen radical
        [H][H]   molecular hydrogen
    """
    if not HAVE_RDKIT:
        return None

    smi = str(smi).strip()
    if not smi:
        return None

    try:
        params = Chem.SmilesParserParams()
        params.removeHs = False
        return Chem.MolFromSmiles(smi, params)
    except Exception:
        try:
            return Chem.MolFromSmiles(smi)
        except Exception:
            return None

def _hydrogen_side_stats(mols: List[Any]) -> np.ndarray:
    """
    Return hydrogen-specific statistics for one reaction side.

    Output order:
        [H atoms,
         H radical electrons,
         H-H bonds,
         isolated H radical atoms]
    """
    n_h_atoms = 0.0
    n_h_radical_electrons = 0.0
    n_hh_bonds = 0.0
    n_isolated_h_radicals = 0.0

    for mol in mols:
        if mol is None:
            continue

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                n_h_atoms += 1.0

                rad_e = float(atom.GetNumRadicalElectrons())
                n_h_radical_electrons += rad_e

                if atom.GetDegree() == 0 and rad_e > 0:
                    n_isolated_h_radicals += 1.0

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            if a1.GetAtomicNum() == 1 and a2.GetAtomicNum() == 1:
                n_hh_bonds += 1.0

    return np.array(
        [
            n_h_atoms,
            n_h_radical_electrons,
            n_hh_bonds,
            n_isolated_h_radicals,
        ],
        dtype=float,
    )

def hydrogen_radical_reaction_features(rxn_smiles: str) -> Dict[str, float]:
    """
    Minimal special descriptors for hydrogen radical recombination:

        [H].[H]>>[H][H]

    Normal Mordred/DRFP may become zero or non-informative because atom counts
    are unchanged and hydrogen-only reactions have no heavy-atom environments.

    These features encode the real reaction change:
        - radical electrons consumed
        - H-H bond formed
    """
    base = {
        "rxn_H_atom_delta": 0.0,
        "rxn_H_radical_electron_delta": 0.0,
        "rxn_HH_bond_delta": 0.0,
        "rxn_isolated_H_radical_atom_delta": 0.0,
        "rxn_is_2H_radicals_to_H2": 0.0,
    }

    if not HAVE_RDKIT:
        return base

    try:
        reactant_side, product_side = _split_reaction_sides_for_hydrogen_patch(rxn_smiles)

        reactant_mols = [
            _mol_from_smiles_keep_explicit_h(s)
            for s in reactant_side.split(".")
            if str(s).strip()
        ]

        product_mols = [
            _mol_from_smiles_keep_explicit_h(s)
            for s in product_side.split(".")
            if str(s).strip()
        ]

        R = _hydrogen_side_stats(reactant_mols)
        P = _hydrogen_side_stats(product_mols)
        d = P - R

        is_2h_to_h2 = (
            R[0] == 2.0 and
            R[1] == 2.0 and
            R[2] == 0.0 and
            R[3] == 2.0 and
            P[0] == 2.0 and
            P[1] == 0.0 and
            P[2] == 1.0 and
            P[3] == 0.0
        )

        base.update({
            "rxn_H_atom_delta": float(d[0]),
            "rxn_H_radical_electron_delta": float(d[1]),
            "rxn_HH_bond_delta": float(d[2]),
            "rxn_isolated_H_radical_atom_delta": float(d[3]),
            "rxn_is_2H_radicals_to_H2": float(bool(is_2h_to_h2)),
        })

        return base

    except Exception as e:
        logging.debug(
            "Hydrogen radical special-case descriptor failed for '%s': %s",
            rxn_smiles,
            e,
        )
        return base

def build_hydrogen_radical_feature_df(reac_series: pd.Series) -> pd.DataFrame:
    """
    Build hydrogen-radical special-case features aligned to the reaction index.
    """
    rows = [
        hydrogen_radical_reaction_features(rxn)
        for rxn in reac_series
    ]

    return pd.DataFrame(
        rows,
        index=reac_series.index,
        dtype=float,
    )

def _is_hydrogen_radical_recombination(rxn_smiles: str) -> bool:
    """
    True only for the explicit special case:

        [H].[H]>>[H][H]
    """
    return bool(
        hydrogen_radical_reaction_features(rxn_smiles)
        .get("rxn_is_2H_radicals_to_H2", 0.0) > 0.5
    )

def _repair_hydrogen_radical_zero_drfp(vector: np.ndarray, rxn_smiles: str) -> np.ndarray:
    """
    If DRFP gives an all-zero vector for [H].[H]>>[H][H], activate a few stable
    sentinel bits so this reaction is not dropped or treated as featureless.

    This changes only all-zero DRFP rows for the exact hydrogen recombination case.
    """
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

    if is_zero and _is_hydrogen_radical_recombination(rxn_smiles):
        for tag in (
            "XT_HRAD_2H_TO_H2",
            "XT_HRAD_RADICAL_ELECTRONS_CONSUMED",
            "XT_HRAD_HH_BOND_FORMED",
        ):
            j = int(hashlib.sha1(tag.encode("utf-8")).hexdigest(), 16) % n
            v[j] = 1

    return v

# ---------------------- PARALLEL MORDRED WORKER HELPERS ----------------------

_MORDRED_WORKER_STATE: Dict[Tuple[bool, bool], Tuple[Any, List[str], int]] = {}


def _get_mordred_worker_state(
    use3d: bool,
    xtqspr_explicit_h_patch_enable: bool = False,
) -> Tuple[Any, List[str], int]:
    """
    Lazily construct a Mordred Calculator inside each process/thread.

    The XTQSPRKinetic explicit-H patch must be applied before Calculator(...)
    is constructed. This is especially important for joblib/loky workers.
    """
    key = (bool(use3d), bool(xtqspr_explicit_h_patch_enable))

    if key not in _MORDRED_WORKER_STATE:
        if bool(xtqspr_explicit_h_patch_enable):
            if apply_xtqspr_mordred_patch is None:
                raise RuntimeError(
                    "mordred_xtqspr_explicit_h_patch_enable=True, but "
                    "mordred_xtqspr_patch.py could not be imported."
                )
            apply_xtqspr_mordred_patch()

        calc = Calculator(descriptors, ignore_3D=not bool(use3d))

        # Do NOT call calc.config(explicit_hydrogens=True) here.
        # User validation showed that Mordred did not reproduce the patched
        # descriptor behavior through Calculator.config(...). The runtime patch
        # above is the actual controlled descriptor modification.

        desc_objs = list(calc.descriptors)
        desc_names = [str(d) for d in desc_objs]
        _MORDRED_WORKER_STATE[key] = (calc, desc_names, len(desc_objs))

    return _MORDRED_WORKER_STATE[key]


def _mordred_descriptor_columns(
    agg: str,
    use3d: bool,
    xtqspr_explicit_h_patch_enable: bool = False,
) -> List[str]:
    """Return output column names for the selected Mordred aggregation mode."""
    _, desc_names, _ = _get_mordred_worker_state(
        bool(use3d),
        bool(xtqspr_explicit_h_patch_enable),
    )

    agg = str(agg or 'sum').lower()
    if agg == 'concat':
        return [f"R_{n}" for n in desc_names] + [f"P_{n}" for n in desc_names]
    if agg == 'delta':
        return [f"{n}_delta" for n in desc_names]
    return [f"{n}_sum" for n in desc_names]


def _mordred_safe_values(
    mol: Any,
    use3d: bool,
    xtqspr_explicit_h_patch_enable: bool = False,
) -> List[float]:
    """Compute Mordred descriptor values for one molecule with robust failure handling."""
    calc, desc_names, ndesc = _get_mordred_worker_state(
        bool(use3d),
        bool(xtqspr_explicit_h_patch_enable),
    )

    if mol is None:
        return [np.nan] * ndesc

    try:
        if bool(use3d) and mol is not None:
            try:
                from rdkit.Chem import AllChem
                m3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(m3d, randomSeed=0xF00D)
                AllChem.UFFOptimizeMolecule(m3d, maxIters=200)
                mol = m3d
            except Exception:
                pass

        dd = calc(mol).asdict()

    except Exception:
        return [np.nan] * ndesc

    out: List[float] = []
    for name in desc_names:
        v = dd.get(name, np.nan)
        try:
            v = float(v)
        except Exception:
            v = np.nan
        out.append(v)

    return out


def _mordred_reaction_descriptor_row(
    idx: Any,
    rxn: Any,
    agg: str,
    use3d: bool,
    xtqspr_explicit_h_patch_enable: bool = False,
) -> Tuple[Any, List[float]]:
    """
    Worker-safe Mordred descriptor aggregation for one reaction SMILES.

    Returns
    -------
    tuple
        (original_index, descriptor_row_values)
    """
    _, _, ndesc = _get_mordred_worker_state(
        bool(use3d),
        bool(xtqspr_explicit_h_patch_enable),
    )

    rxn = str(rxn)

    try:
        reactant_smiles, product_smiles = rxn.split(">>")
    except ValueError:
        reactant_smiles, product_smiles = rxn, ""

    try:
        reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles.split('.') if str(s).strip()]
    except Exception:
        reactant_mols = []

    try:
        product_mols = [Chem.MolFromSmiles(s) for s in product_smiles.split('.') if str(s).strip()]
    except Exception:
        product_mols = []

    R_mat = [
        _mordred_safe_values(m, use3d, xtqspr_explicit_h_patch_enable)
        for m in reactant_mols
    ] if reactant_mols else []

    P_mat = [
        _mordred_safe_values(m, use3d, xtqspr_explicit_h_patch_enable)
        for m in product_mols
    ] if product_mols else []

    R = np.nansum(R_mat, axis=0) if len(R_mat) > 0 else np.zeros(ndesc)
    P = np.nansum(P_mat, axis=0) if len(P_mat) > 0 else np.zeros(ndesc)

    agg = str(agg or 'sum').lower()
    if agg == 'concat':
        data = np.concatenate([R, P])
    elif agg == 'delta':
        data = P - R
    else:
        data = R + P

    return idx, list(np.asarray(data, dtype=float))

@timeit_logger("Calculate descriptors mordred")
def calculate_descriptors_mordred(
    reac_series: pd.Series,
    nan_policy: str = 'mean',
    agg: str = 'sum',
    parallel_enable: bool = False,
    n_jobs: int = 1,
    backend: str = 'loky',
    batch_size: Union[int, str] = 'auto',
    xtqspr_explicit_h_patch_enable: bool = False,
) -> pd.DataFrame:
    """
    Compute Mordred descriptors aggregated across reactants/products per reaction.

    Parameters
    ----------
    reac_series:
        Series[str] of reaction SMILES in "R>>P" format.
    nan_policy:
        How to fill missing descriptor values: {'mean','median','zero','drop'}.
    agg:
        Aggregation over reactants/products: {'sum','concat','delta'}.
    parallel_enable:
        If True, compute independent reaction rows using joblib.Parallel.
    n_jobs:
        Number of outer jobs. Use -1 for all available CPUs.
    backend:
        Joblib backend. Recommended: 'loky' for CPU/process parallelism.
    batch_size:
        Joblib batch size. 'auto' is usually safe; an int such as 16 or 32 can
        reduce scheduling overhead for very large reaction sets.

    Returns
    -------
    pd.DataFrame
        Numeric Mordred descriptor matrix [N × D].
    """
    use3d = os.environ.get("XT_MORDRED_3D", "").strip().lower() in ("1", "true", "yes", "on")
    agg = str(agg or 'sum').lower()
    cols = _mordred_descriptor_columns(agg, use3d,
                                        xtqspr_explicit_h_patch_enable=bool(xtqspr_explicit_h_patch_enable))
    items = list(reac_series.items())

    try:
        n_jobs = int(n_jobs)
    except Exception:
        n_jobs = 1

    if n_jobs == 0:
        n_jobs = 1

    backend = str(backend or 'loky').lower()
    if backend not in {'loky', 'multiprocessing', 'threading'}:
        logging.warning("Invalid mordred_parallel_backend=%r; using 'loky'.", backend)
        backend = 'loky'

    use_parallel = bool(parallel_enable) and len(items) > 1 and n_jobs != 1

    if use_parallel:
        logging.info(
            "Mordred descriptor calculation: parallel_enable=True, n_jobs=%s, backend=%s, batch_size=%s, reactions=%d",
            n_jobs,
            backend,
            batch_size,
            len(items),
        )
        results = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
            delayed(_mordred_reaction_descriptor_row)(
                idx,
                rxn,
                agg,
                use3d,
                bool(xtqspr_explicit_h_patch_enable),
            )
            for idx, rxn in items
        )
    else:
        if bool(parallel_enable) and n_jobs == 1:
            logging.info(
            "Mordred descriptor calculation: parallel_enable=%s, n_jobs=%s, backend=%s, "
            "batch_size=%s, reactions=%d, xtqspr_explicit_h_patch=%s",
            bool(parallel_enable),
            n_jobs,
            backend,
            batch_size,
            len(items),
            bool(xtqspr_explicit_h_patch_enable))
        results = [
            _mordred_reaction_descriptor_row(
                idx,
                rxn,
                agg,
                use3d,
                bool(xtqspr_explicit_h_patch_enable),
            )
            for idx, rxn in items
        ]

    index = [idx for idx, _row in results]
    rows = [row for _idx, row in results]

    desc_df = (
        pd.DataFrame(rows, index=index, columns=cols, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
    )

    if nan_policy == 'mean':
        desc_df = desc_df.fillna(desc_df.mean(numeric_only=True))
    elif nan_policy == 'median':
        desc_df = desc_df.fillna(desc_df.median(numeric_only=True))
    elif nan_policy == 'zero':
        desc_df = desc_df.fillna(0.0)
    elif nan_policy == 'drop':
        desc_df = desc_df.dropna(axis=1)
    else:
        logging.warning("Unknown nan_policy='%s', defaulting to 'mean'", nan_policy)
        desc_df = desc_df.fillna(desc_df.mean(numeric_only=True))

    return desc_df.select_dtypes(include=[np.number]).copy()

# --------------------------- MODEL TRAINING ----------------------------------

def scale_features(X: pd.DataFrame, use_robust: bool = True) -> Tuple[pd.DataFrame, Any]:
    """
    Fit a scaler and transform X.

    Args:
        X: DataFrame [N × D], numeric features
        use_robust: bool, RobustScaler if True else StandardScaler

    Returns:
        (X_scaled DataFrame, fitted scaler)
    """
    scaler = RobustScaler() if use_robust else StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X.values), columns=X.columns, index=X.index)
    return Xs, scaler

@timeit_logger("Train per-family XGB specialists and a global fallback model")
def train_topic_models(
    X_full: pd.DataFrame,
    Y: pd.DataFrame,
    topics: List[int],
    features: List[str],
    random_state: int = 42,
    use_family_scaler: bool = True,
    family_min_size: int = 15,
    xgb_params: Optional[Dict[str, Any]] = None,
    xgb_params_by_family: Optional[Dict[int, Dict[str, Any]]] = None,
    xgb_params_by_target_global: Optional[Dict[int, Dict[str, Any]]] = None,
    use_robust_scaler: bool = None,
    features_by_family: Optional[Dict[int, List[str]]] = None,
    target_cols: Optional[Sequence[str]] = None,
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """
    Train per-family XGB specialists and a global fallback model for arbitrary
    configured continuous target columns.

    Args:
        X_full: DataFrame [N × D], final feature matrix.
        Y: DataFrame [N × T], target matrix. Columns are target names.
        topics: List[int], family label per sample (training labels).
        features: columns to train on.
        target_cols: target columns to fit; defaults to Y.columns.

    Returns:
        models: Dict[int or 'global' -> fitted MultiOutputXGBRegressor]
        scalers: Dict[int or 'global' -> fitted scaler]
    """
    target_cols = _target_cols_from_frame(Y, target_cols)
    missing_targets = [c for c in target_cols if c not in Y.columns]
    if missing_targets:
        raise KeyError(f"train_topic_models missing target column(s): {missing_targets}; available: {list(Y.columns)}")

    # --- Resolve defaults / safety --------------------------------------------
    if use_robust_scaler is None:
        use_robust_scaler = bool(DEFAULT_CONFIG.get('use_robust_scaler', True))

    if xgb_params is None:
        xgb_params = dict(DEFAULT_CONFIG['xgb_params'])
    else:
        xgb_params = dict(xgb_params)  # shallow copy so we can mutate

    # Ensure estimator reproducibility if user didn't set it explicitly
    xgb_params.setdefault('random_state', int(random_state))
    xgb_params.setdefault('n_jobs', -1)

    # normalize per-family param keys to int where possible
    if isinstance(xgb_params_by_family, dict):
        _tmp = {}
        for k, v in xgb_params_by_family.items():
            try:
                kk = int(k)
            except Exception:
                kk = k
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault('random_state', int(random_state))
                _tmp[kk] = vv
        xgb_params_by_family = _tmp

    t0_all = time.time()

    logging.info("=== Training specialist (per-family) models + global fallback ===")
    logging.info("Targets: %s", target_cols)
    logging.info("Features: %d | use_family_scaler: %s | family_min_size: %d | scaler: %s",
                 len(features), use_family_scaler, family_min_size,
                 "RobustScaler" if use_robust_scaler else "StandardScaler")
    logging.info("XGB params: %s", {k: xgb_params[k] for k in sorted(xgb_params)})

    models: Dict[Any, Any] = {}
    scalers: Dict[Any, Any] = {}

    # Global scaler
    X_all = X_full[features]
    t0 = time.perf_counter()
    X_all_scaled, global_scaler = scale_features(X_all, use_robust=use_robust_scaler)
    scalers['global'] = global_scaler
    logging.info(
        "Fitted global scaler on %d samples and %d features in %s.", len(X_all_scaled), 
        len(features), _format_elapsed(time.perf_counter() - t0))

    # Per-topic specialists
    topics_series = pd.Series(topics, index=X_full.index)
    fam_ids = sorted(topics_series.unique())
    logging.info("Found %d candidate families: %s", len(fam_ids), fam_ids)

    trained, skipped = 0, 0
    for topic in fam_ids:
        idx = topics_series == topic
        n = int(idx.sum())
        if n < family_min_size:
            logging.warning("Skipping topic %s due to insufficient data (%d < %d).", topic, n, family_min_size)
            skipped += 1
            continue

        # choose columns for this family (fallback to global feature list)
        cols_this = (features_by_family.get(topic, features) if features_by_family else features)
        X_fam = X_full.loc[idx, cols_this]
        y_fam = Y.loc[idx, target_cols]

        # if per-family feature dims differ, always use a family scaler
        force_family_scaler = features_by_family is not None
        scaler_used = "family" if (use_family_scaler or force_family_scaler) else "global"

        t0 = time.time()
        if use_family_scaler or force_family_scaler:
            X_fam_scaled, scaler_fam = scale_features(X_fam, use_robust=use_robust_scaler)
            scalers[topic] = scaler_fam
        else:
            X_fam_scaled = global_scaler.transform(X_fam)
        dt_scale = time.time() - t0

        t1 = time.time()
        # Base params for this family (fallback)
        params_base_this = dict(xgb_params) if isinstance(xgb_params, dict) else {}
        fam_entry = None
        if isinstance(xgb_params_by_family, dict) and int(topic) in xgb_params_by_family:
            fam_entry = xgb_params_by_family[int(topic)]
            # Backward compat: allow a plain dict of XGB params (shared across targets)
            if isinstance(fam_entry, dict) and 'by_target' not in fam_entry and 0 not in fam_entry and '0' not in fam_entry:
                params_base_this.update(dict(fam_entry))
                fam_entry = None

        # Per-target overrides: family-specific else global
        def _coerce_by_target(obj):
            if obj is None:
                return None
            if isinstance(obj, dict) and 'by_target' in obj and isinstance(obj['by_target'], dict):
                obj = obj['by_target']
            if not isinstance(obj, dict):
                return None
            out = {}
            for k, v in obj.items():
                try:
                    kk = int(k)
                except Exception:
                    continue
                if isinstance(v, dict):
                    out[kk] = dict(v)
            return out if out else None

        by_target_this = _coerce_by_target(fam_entry)
        if by_target_this is None:
            by_target_this = _coerce_by_target(xgb_params_by_target_global)

        model = MultiOutputXGBRegressor(base_params=params_base_this, params_by_target=by_target_this)
        X_fit = pd.DataFrame(X_fam_scaled, columns=cols_this, index=X_fam.index)
        y_fit = y_fam[target_cols].values
        model.fit(X_fit, y_fit)
        try:
            model.target_cols_ = list(target_cols)
        except Exception:
            pass
        dt_fit = time.time() - t1

        models[topic] = model
        trained += 1
        logging.info("Trained topic %s | n=%d | scaler=%s | scale=%.2fs | fit=%.2fs",
                     topic, n, scaler_used, dt_scale, dt_fit)

    # Global model
    t0 = time.time()
    # Use global per-target overrides if provided
    g_by_target = None
    try:
        if isinstance(xgb_params_by_target_global, dict):
            g_by_target = {int(k): dict(v) for k, v in xgb_params_by_target_global.items() if isinstance(v, dict)}
    except Exception:
        g_by_target = None
    g_model = MultiOutputXGBRegressor(base_params=dict(xgb_params) if isinstance(xgb_params, dict) else {}, params_by_target=g_by_target)
    Xg_fit = pd.DataFrame(X_all_scaled, columns=features, index=X_all.index)
    yg_fit = Y[target_cols].values
    g_model.fit(Xg_fit, yg_fit)
    try:
        g_model.target_cols_ = list(target_cols)
    except Exception:
        pass
    models['global'] = g_model
    logging.info("Trained GLOBAL model on %d samples in %.2fs.", len(X_all_scaled), time.time() - t0)

    logging.info("=== Training summary: trained=%d | skipped=%d | total_families=%d | elapsed=%.2fs ===",
                 trained, skipped, len(fam_ids), time.time() - t0_all)
    return models, scalers
