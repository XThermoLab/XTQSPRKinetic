# XTQSPRKinetic v2.5.1 — What's New 

1) Auto-Chooser for Family Schemes:
   - Try multiple candidate configurations:
     -DRFP → LDA → topic probabilities (soft routing), 
   - Fast evaluation (reduced trees / fewer Boruta iters) to score candidates via global CV metrics,
   - Select the best scheme and re-run the full pipeline once with the winner.

2) Scientific comments & professional docstrings across config and functions.

# XTQSPRKinetic v2.5.2 — What's New 

1) crossval_score_family upgraded:
   - Reports both R² (can be negative) and RMSE (non-negative) per family & target.
   - Saves a bar chart of CV R² at 600 dpi with y-axis spanning the data range.

2) Build Boruta provenance for pipeline.pkl

3) Build and dump a comprehensive pipeline.pkl using ONLY config-driven options.

# XTQSPRKinetic v2.5.3 — What's New 

1) DRFP→PCA→FCM and Bootstrap-FCM artifacts
    -LDA — Latent Dirichlet Allocation (topic model)
    -PCA — Principal Component Analysis (linear projection)
    -FCM — Fuzzy C-Means (fuzzy clustering)

2) Update the Auto-Chooser helpers to include FCM-on-DRFP:
   - Try multiple candidate configurations:
     -DRFP → LDA → topic probabilities (soft routing), 
     -DRFP → LDA → FCM on topic space (soft memberships from FCM; sometimes crisper boundaries), 
      with/without Bootstrap & hybrid routing),
     -DRFP → PCA → FCM (when you want clustering directly on fingerprints, but with PCA to fix scale/geometry),
   -Bootstrap variants: Refit LDA or FCM on resamples to get stability (vote fraction, entropy) for hybrid routing or label replacement.
   - Fast evaluation (reduced trees / fewer Boruta iters) to score candidates via global CV metrics,
   - Select the best scheme and re-run the full pipeline once with the winner.

# XTQSPRKinetic v2.5.4 — What's New 

1) Add LDA+GMM, GMM, LDA+HDBSCAN, and HDBSCAN

2) Update the Auto-Chooser helpers

# XTQSPRKinetic v2.5.5 — What's New 

1) Minor bug fiexed

# XTQSPRKinetic v2.5.6 — What's New 

1) Minor bug fiexed

# XTQSPRKinetic v2.5.7 — What's New 

1) Minor bug fiexed

# XTQSPRKinetic v2.5.8 — What's New 

1) Pipeline lightweight (default): save summaries only (keeps pickle small)

2) Pipeline full metrics: embed the full DataFrames.

3) Add optional pipline zip configuration

4) Minor bug fix

# XTQSPRKinetic v2.5.9 — What's New 

1) Replace parse_args and load_config

2) Extend _materialize_candidate_defaults(...) with bootstrap GMM/HDBSCAN

3) Add lightweight timing utilities

4) (a) Build a fast per-candidate config (_make_fast_eval_cfg) and 
   (b) Compute a single scalar score from each candidate’s quick run (_compute_eval_score).

# XTQSPRKinetic v2.5.10 — What's New 

1) Changed default configuration structure for Brouta

2) Replace improve load_config

3) Replace boruta_feature_selection

4) Replace run_pipeline

# XTQSPRKinetic v2.5.11 — What's New 

1) Changed default configuration structure for PCA, GMM, FCM

2) Revised following functions:
        _compute_eval_score
        _materialize_candidate_defaults
        _hdbscan_memberships
        resolve_none_config
        load_config
        fit_pca_on_drfp
        bootstrap_fcm_topics
        generate_drfp_topics
        compute_gmm_topics
        compute_hdbscan_topics
        compute_fcm_topics
        bootstrap_lda_topics
        bootstrap_hdbscan_topics
        auto_choose_family_scheme
        run_pipeline

# XTQSPRKinetic v2.5.12 — What's New 

1) Run the four run_pipeline edits:
        -Replacement of the bootstrap replacement block (Section 1).
        -Insert the _training_family assignment right after it (Section 2).
        -Replacement of the reporting columns block (Section 3).
        -Replacement of the family assignments writer block (Section 4).
         (Do this for both copies of run_pipeline present in your file.)

2) Replace the build_pipeline_components function with the new version (Section 5).

3) For FCM, GMM, and HDBSCAN runs, replacement will only occur if:
        -bootstrap_replace_family_labels=True, and
        -The relevant method’s bootstrap is enabled (e.g., bootstrap_fcm_enable=True), and the code actually produced 
         agg_topic_proba_df and stability_df (already created earlier in run_pipeline for each method).

# XTQSPRKinetic v2.5.13 — What's New 

1) New topic path: LDA + QDA (supervised boundary over LDA families)
   - topic_method: 'lda+qda' (added to allowed set).
   - QDA trains on LDA hard families and runs in either space:
       qda_input: 'lda' | 'drfp'          # 'lda' recommended to start
       qda_reg_param: float               # small shrinkage (e.g., 0.05) stabilizes covariances
       qda_priors: list|None              # class priors; None => empirical
       qda_store_covariance: bool
       qda_tol: float                     # (fixed param name; used by compute_qda_topics)
   - Packaged under pkg['topic_assigners']['qda'] = {'model', 'input', 'pca'}.

2) Bootstrap-QDA (bagging over QDA)
    - Mirrors bootstrap behavior of other topic builders; produces:
        agg_topic_proba_df (mean class probs) and stability_df (modal_vote, norm_entropy).
    - Config (mirrors LDA’s toggles; default mirrors bootstrap_lda_enable):
        bootstrap_qda_enable: None|bool
    - Plays with existing replacement/hybrid routing:
        bootstrap_replace_family_labels: bool
        bootstrap_replace_metric: 'vote_fraction'|'entropy'
        bootstrap_replace_threshold: float
        bootstrap_retrain_on_replaced: bool

3) Per-family Boruta (optional, API-stable)
   - OFF by default; enable via environment variable (no CLI change):
       QRPS_EXPERIMENTAL_PER_FAMILY_BORUTA=1
   - Runs Boruta inside each eligible family on TRAIN folds only, aggregates ≥60% fold stability,
     unions across targets (if per-target mode), and falls back to global for small families.
   - Training & prediction now accept per-family feature sets transparently:
       train_topic_models(..., features_by_family=...)
       predict_arrhenius(..., features_by_family=...)
       predict_arrhenius_weighted(..., ..., features_by_family=...)
       predict_arrhenius_hybrid(..., ..., features_by_family=...)
   - Provenance: pkg['boruta_meta']['per_family'] includes report + features_by_family map.

4) Bootstrap-GMM: component alignment cost (configurable)
   - New knobs (defaults preserve old behavior):
       bootstrap_gmm_align_components: true
       bootstrap_gmm_cost: 'mean_euclid' | 'mean_mahal' | 'gaussian_kl'
   - Alignment uses Hungarian (if SciPy) on chosen cost; otherwise greedy fallback.

5) Scientific / logic fixes
   - GMM & HDBSCAN comments corrected to “unsupervised” (were mislabeled previously).
   - Bootstrap replacement outputs now go to method-aware folder:
       output_dir/bootstrap_{method}/  # e.g., bootstrap_qda, bootstrap_gmm, ...
   - GMM default max_iter restored to 100 (was 1 → premature convergence).
   - QDA param name fixed: compute_qda_topics expects 'tol' (not 'qda_tol').
   - Run method set includes 'lda+qda'; ensure only one canonical run_pipeline definition.

6) Backward compatibility
   - All new features default to prior behavior (no surprises):
       • QDA path opt-in via topic_method: 'lda+qda'
       • Bootstrap-QDA mirrors bootstrap_lda_enable when None
       • Per-family Boruta is OFF unless env var or config var is set
       • GMM alignment defaults match previous implicit Euclidean-mean matching
   - Existing artifacts unchanged; new metadata stored additively in pipeline.pkl.

7) Practical starters
   - QDA: topic_method: lda+qda; qda_input: lda; qda_reg_param: 0.05
   - Bootstrap + replacement: bootstrap_*_enable: true;
     bootstrap_replace_family_labels: true; metric: vote_fraction; threshold: 0.75; retrain_on_replaced: true
   - Per-family Boruta (if data-rich): export env var above; inspect pkg['boruta_meta']['per_family'].

# XTQSPRKinetic v2.5.14 — What's New 

1) Expanded reporting & diagnostics toggles in DEFAULT_CONFIG
   - New switches now exposed for cluster validity and topic builders:
   - cluster_validity_enable, cluster_validity_sample, gmm_report_aic_bic, qda_report_metrics, 
      hdbscan_stability_enable, plus shap_soft_enable for probability-weighted SHAP aggregation. 

2) Richer bootstrap outputs (LDA/FCM) written by run_pipeline
   - In the LDA/FCM paths, the pipeline now saves a small set of standardized artifacts per run:
   - bootstrap_meta.json, bootstrap_topics_aggregated.csv, bootstrap_stability.csv, bootstrap_assignments.csv, 
     and a compact bootstrap_summary.json. This block is new compared to 2.5.13. 

3) HDBSCAN persistence & outlier reporting
   - When enabled (hdbscan_stability_enable=True), the HDBSCAN branch writes a dedicated metrics file with cluster persistence 
     and soft outlier scores: metrics/hdbscan_stability.json. 

4) Soft-CV auditing and soft-SHAP aggregation
   - The pipeline includes a soft cross-validation report (family_crossval_soft_r2.csv) that weights per-family predictions 
     by routing probabilities, and an optional probability-weighted SHAP aggregation (shap/soft_mean_abs_shap.csv) controlled 
     by run_family_cv_soft and shap_soft_enable. 

5) Family-wise CV & global metrics block retained and extended
   - Family and global R2/MAE/RMSE summaries are still produced, and the soft-weighted family stats writer (family_stats_soft.csv) 
     is present alongside the classic metrics.

6) The new bootstrap artifacts make the pipeline more reproducible and auditable without inflating the pickle size—good practice for 
   model provenance. (See the write block that serializes meta, stability, assignments, and the compact summary.) 

7) The HDBSCAN metrics capture recognized stability notions (persistence) and soft outlier scores, aligning with common HDBSCAN diagnostics. 
   The code cleanly guards the write with a try/except and a config flag. 

8) Soft-CV and soft-SHAP are methodologically sound for specialist routing systems: 
   they weight contributions by the topic probabilities used at inference, giving a more faithful performance/attribution picture under soft routing.

# XTQSPRKinetic v2.5.15 — What's New 

1) Coherence options finalized

    -Token sources now include brics_smarts in addition to drfp_bits and drfp_ngrams.
    -Config additions:
        -brics_to_smarts (toggle SMARTS vs fragment SMILES)
        -brics_min_fragment_size (integer threshold)
        -ring (singular; used for DRFP n-gram mapping via DrfpEncoder)

2) Optional imports expanded & normalized: HAVE_RDKIT, HAVE_GENSIM present and used as guards.

3) Placement & flow: the coherence computation is properly hooked in the LDA diagnostics section right after the Jaccard write/except.

# XTQSPRKinetic v2.5.16 — What's New 

1) Adjusted R² metrics added alongside classic R²

   - Introduced a small helper _adj_r2(...) to compute adjusted R² from (R², n samples, p predictors).
   - Global metrics (global_stats.csv) now include an AdjR2 column per target next to R2/MAE/RMSE.
   - Family-wise metrics (family_stats.csv) now report AdjR2 per family/target in addition to R2, MAE, RMSE.
   - Soft-weighted family stats (family_stats_soft.csv) now include wAdjR2 beside wR2, wMAE, wRMSE.
   - Cross-validation diagnostics
       • family_crossval_r2.csv: new cv_adjr2 column next to cv_r2 and cv_rmse.
       • family_crossval_soft_r2.csv: new cv_adjr2_soft column next to cv_r2_soft and cv_rmse_soft.

2) Methodology preserved; Auto-Chooser semantics unchanged

   - auto_eval_metric: r2_mean continues to use the original R2 columns only; adjusted R² is purely
     additive reporting and does not affect candidate ranking or training logic.
   - No changes to the pipeline.pkl schema or downstream APIs; all new adjusted-R² values live in
     CSV metrics only.

3) Numerically robust adjusted-R² implementation

   - _adj_r2(...) guards against degenerate regimes (e.g., n ≤ p+1, non-finite R²) by returning NaN
     instead of misleading scores.
   - This makes it straightforward to report both R² and Adjusted R² in manuscripts and benchmarking
     tables with no external post-processing.

4) The Kinetics_Performance.py v2.1 script has been added to the package to evaluate pipeline.pkl.

# XTQSPRKinetic v2.5.17 — What's New 

1) Removed redundant SHAP outputs (shap_topic_{topic}importance.csv, shap_topic{topic}summary.png, 
   shap_topic{topic}_force.html) that duplicated existing files.

2) Added proper CSV storage for raw SHAP values (shap_values_family_{topic}_{out_name}.csv) 
   to support the point plot.

3) Fixed the Global section CSV data for SHAP values.

4) Updated evaluation metrics to include dimensionless indicators (nRMSE, AARD, RAE) 
   alongside R² and adjusted-R² for a more robust and scale-independent model assessment (Auto-chooser).

5) Modified the scoring logic to calculate the mean of values across all rows 
   (for all outputs like lnA, n, Ea) instead of relying solely on the last row.
   This ensures the score reflects the model's overall performance across all parameters (Auto-chooser).

6) Implemented negative scoring for error-based metrics (nRMSE, AARD, RAE).
   Since the optimization process aims to maximize the score, error metrics 
   are inverted (multiplied by -1) so that lower errors result in higher scores (Auto-chooser).

7) Expanded evaluation metrics of cross validation to include both dimensional (RMSE, MAE) 
   and dimensionless (nRMSE, AARD, RAE) indicators for a comprehensive assessment.

8) Replaced 'cross_val_score' with 'cross_val_predict' to generate out-of-fold 
   predictions. This allows for the calculation of complex metrics (AARD, RAE) 
   and avoids redundant model training, improving computational efficiency.

9) Updated 'crossval_score_family' to report metrics per family and per target 
   (lnA, n, Ea) using the full set of 7 metrics.

10) Updated 'crossval_score_family_soft' to calculate the same 7 metrics 
    based on the soft mixture of specialist predictions, ensuring consistency 
    with the hard routing evaluation.

11) The Kinetics_Performance.py v2.2 script has been added to the package to evaluate pipeline.pkl.

12) [Refactoring] Function Renaming & Strategy Support:
    - Renamed CV functions to explicitly support three distinct strategies:
        1. crossval_score_family_kfold (Standard K-Fold)
        2. crossval_score_family_groupkfold (Group K-Fold)
        3. crossval_score_family_nested (Nested Group K-Fold)
    - Corresponding soft-routing functions renamed similarly (e.g., 
        crossval_score_family_soft_nested).
    - This resolves function signature conflicts and allows dynamic strategy 
        selection via configuration.

13) [Feature] Dynamic Configuration Integration:
    - Functions now accept a 'cv_strategy' parameter from the main config 
        to switch between KFold, GroupKFold, and Nested approaches without 
        code modification.
    - Output filenames (CSV/PNG) now automatically include the strategy suffix 
        (e.g., 'family_crossval_groupkfold.csv') to prevent overwriting results.

14) [Feature] Advanced Nested CV Configuration:
    - Added 'use_grid_search' parameter to enable/disable hyperparameter tuning 
        within the inner loop of Nested CV. Disabling this allows Nested CV to run 
        with default params for faster benchmarking.
    - Added 'scoring' parameter to allow customization of the optimization metric 
        for GridSearchCV (e.g., 'neg_mean_squared_error', 'r2').
    - Added 'param_grid' parameter to inject custom hyperparameter search spaces 
        from the configuration file, removing the need to hardcode tuning grids.

15) [Robustness] Inner Loop Logic:
    - Improved handling of small sample sizes in the inner loop of Nested CV.
    - Automatically adjusts the number of inner splits (Inner CV) if the number 
        of groups or samples is insufficient for the requested fold count, 
        preventing crashes during GridSearchCV.

16) [Logging] Enhanced Feedback:
    - Added detailed logging for Nested CV steps, including explicit messages 
        when Grid Search is skipped vs. executed, and reporting of best parameters 
        found in each fold.
    - Added detailed logging for utils fuctions.

# XTQSPRKinetic v2.5.18 — What's New 

1) Cross-validation philosophy refactor: CV is no longer treated as an in-run
    "final model comparator". The previous run_pipeline() block that split the
    same reaction network into train/test folds to *report CV performance*
    (run_family_cv / run_family_cv_soft with cv_strategy = kfold/groupkfold/
    nested_groupkfold) has been removed from the main training flow. Instead,
    v2.5.18 aligns CV with the intended methodology: use CV ONLY as OFFLINE
    model-selection (hyperparameter recommendation), then train ONCE on the full
    clustered network and evaluate externally on a different reaction network.
    (See the new HYPERPARAMETER RECOMMENDER config section and run_pipeline()
    recommender/consumer block.)  

2) New “Hyperparameter Recommender (offline CV)” module:
    - recommend_xgb_hyperparams(...) performs CV-based tuning per target
      (lnA, n, Ea) and writes a structured JSON containing recommended XGBoost
      params per family or per size-regime pool (depending on hp_recommender_mode).
    - The function explicitly documents the scientific intent (NOT a final score).
    - New config controls: hp_recommender_run, hp_recommender_exit_after,
      hp_recommender_mode, hp_recommender_param_grid, hp_recommender_n_folds,
      hp_recommender_min_fold_size, hp_recommender_min_family_size,
      hp_recommender_scoring (currently neg_mean_rmse), report paths, etc. 

3) Two-phase workflow added inside run_pipeline():
    (A) Recommendation run: hp_recommender_run=True → write JSON (and optional CSV),
        optionally exit early via hp_recommender_exit_after=True.
    (B) Final training run: hp_use_recommended_params=True and hp_recommended_json_path
        → load JSON and apply recommended params during the single full-network fit. 

4) Training now supports per-family/per-target hyperparameters cleanly:
    train_topic_models(...) was extended to accept:
       - xgb_params_by_family (family-specific overrides),
       - xgb_params_by_target_global (global per-target defaults),
    and uses MultiOutputXGBRegressor so each target can receive different params
    while still behaving as one “specialist” object per family. 

5) Per-family Boruta stability control: a dedicated fold-count knob
    boruta_family_n_folds was added to DEFAULT_CONFIG to make within-family
    Boruta stability aggregation explicitly controllable (typical 3–10; reduce
    if families are small). 

6) SHAP artifact behavior: v2.5.17 removed some “legacy” SHAP outputs as redundant,
    but v2.5.18’s SHAP routine includes explicit “legacy per-target artifacts”
    generation (e.g., force HTML) in addition to the raw SHAP CSV pathway. 

7) Packaging / environment robustness: optional dependency flags expanded and
    lightweight package-version lookup (importlib.metadata/importlib-metadata)
    was added for better provenance and diagnostics. 

# XTQSPRKinetic v2.5.19-reviewer-diagnostics — What's New

1) Reviewer-response diagnostic layer added
   - Added a dedicated network-preserving diagnostic module controlled by:
       reviewer_diagnostics_enable
   - The new diagnostics are designed to address reviewer concerns without
     reintroducing random reaction-level K-fold validation as the final
     performance estimator.
   - The diagnostic layer is called after full-network fitting and global metric
     generation, preserving the v2.5.18 methodology in which the final model is
     trained once on the full clustered reaction network.

2) Dataset and target-summary reports
   - Added reviewer_dataset_summary_enable.
   - New outputs are written to:
       reviewer_diagnostics/dataset_target_summary.csv
       reviewer_diagnostics/learned_family_counts.csv
       reviewer_diagnostics/rmg_family_counts.csv
       reviewer_diagnostics/dataset_summary_meta.json
   - These reports document the number of reactions, Arrhenius target
     distributions, learned-family sizes, and original RMG/mechanistic family
     counts when available.

3) External mechanistic alignment between learned families and RMG labels
   - Added reviewer_alignment_enable and rmg_family_col.
   - Learned XTQSPRKinetic family labels are compared post hoc with the original
     RMG/mechanistic reaction-family labels.
   - The RMG labels are not used during family discovery, Boruta feature
     selection, specialist-model training, or routing.
   - New outputs include:
       reviewer_diagnostics/family_rmg_contingency_matrix.csv
       reviewer_diagnostics/family_rmg_alignment_metrics.csv
       reviewer_diagnostics/family_rmg_alignment_metrics.json
       reviewer_diagnostics/family_rmg_enrichment_by_family.csv
   - Metrics include cluster purity, adjusted Rand index, normalized mutual
     information, adjusted mutual information, homogeneity, completeness,
     V-measure, dominant RMG family, dominant-class count, within-family purity,
     and RMG-label entropy.

4) Bootstrap stability summaries for selected learned-family assignments
   - Added reviewer_stability_summary_enable.
   - Aggregates bootstrap assignment diagnostics when stability_df is available.
   - New family-resolved stability output:
       reviewer_diagnostics/bootstrap_stability_by_learned_family.csv
   - Reports mean/median/min/max modal vote fraction and normalized entropy
     by learned family when the corresponding bootstrap columns are available.

5) Residual and failure-mode audit outputs
   - Added reviewer_residual_enable and reviewer_residual_top_n.
   - Residuals are analyzed by learned family, RMG/mechanistic family label,
     target parameter, target magnitude, and routing/stability diagnostics when
     available.
   - New outputs include:
       reviewer_diagnostics/residuals_all_reactions.csv
       reviewer_diagnostics/top_residuals_lnA.csv
       reviewer_diagnostics/top_residuals_n.csv
       reviewer_diagnostics/top_residuals_Ea.csv
       reviewer_diagnostics/residual_summary_by_learned_family.csv
       reviewer_diagnostics/residual_summary_by_rmg_family.csv
   - These files support reaction-level error auditing and mechanistic
     interpretation of high-residual cases.

6) Size-matched and optional empirical random-cluster negative controls
   - Added reviewer_random_controls_mode with options:
       off, size_matched, empirical_label_sampling.
   - reviewer_random_controls_enable and reviewer_random_controls_preserve_sizes
     are retained as deprecated compatibility keys.
   - The manuscript/reviewer-response setting is:
       reviewer_random_controls_mode = size_matched.
   - In size_matched mode, random controls train specialist models after
     randomly permuting learned-family labels, preserving the exact family-size
     distribution while destroying the chemical relationship between reactions
     and learned families.
   - New outputs include:
       reviewer_diagnostics/random_cluster_controls_all_runs.csv
       reviewer_diagnostics/random_cluster_controls_summary.csv
       reviewer_diagnostics/random_cluster_controls_meta.json.

7) Component-wise ablation diagnostics
   - Added reviewer_ablation_enable.
   - Added reviewer_ablation_include_rmg_label_experts to compare learned-family
     specialists with specialists trained using original RMG/mechanistic labels.
   - Added reviewer_ablation_use_all_features to optionally compare the
     Boruta-confirmed descriptor set against the full prefiltered descriptor
     matrix.
   - New output:
       reviewer_diagnostics/component_ablation_metrics.csv
   - The ablation table compares the proposed MoE workflow, global XGBoost
     baseline, RMG-label expert baseline, and optional all-feature variants.

8) Optional k-sensitivity diagnostic driver
   - Added reviewer_k_sensitivity_enable, reviewer_k_values, and
     reviewer_k_sensitivity_shap_sample_size.
   - This optional driver supports separate k-sensitivity runs for candidate
     family resolutions such as k = 3, 5, 7, 9, and 11.
   - It is disabled by default because it can launch multiple full pipeline jobs
     and is intended for final reviewer/manuscript diagnostics rather than
     routine model training.

9) Unified reviewer diagnostics dispatcher and runtime report
   - Added run_reviewer_diagnostics(...), a central dispatcher for the reviewer
     diagnostic layer.
   - The dispatcher coordinates dataset summaries, RMG-family alignment,
     bootstrap stability summaries, residual audits, component ablation,
     random-cluster controls, and optional k-sensitivity diagnostics.
   - New runtime report:
       reviewer_diagnostics/reviewer_diagnostics_runtime.json
   - The runtime report records total reviewer-diagnostics wall-clock time in
     seconds and minutes.

10) Run-pipeline integration after global metrics
    - The reviewer diagnostic dispatcher is called after global_stats.csv is
      written and before SHAP computation.
    - This placement allows diagnostics to use the final full-network predictions,
      learned families, trained specialist/global models, confirmed Boruta
      features, and bootstrap stability table if available.
    - Failures in the diagnostic layer are caught and logged so that the main
      trained pipeline and SHAP workflow remain recoverable.

11) Metric-consistent diagnostic evaluation helper
    - Added _evaluate_prediction_table(...) for diagnostic model comparisons.
    - The helper reports the same performance quantities used in the revised
      manuscript tables:
        R2, adjusted R2, MAE, RMSE, nRMSE, AARD, RAE, number of samples,
        number of features, and status.
    - This ensures consistency among ablation, random-control, and global
      reconstruction diagnostic outputs.

12) Reviewer-specific configuration block added to DEFAULT_CONFIG
    - Added the REVIEWER-RESPONSE DIAGNOSTICS section to DEFAULT_CONFIG.
    - The block exposes reviewer diagnostic switches without changing the core
      model-training interface.
    - The original v2.5.18 full-network training and offline hyperparameter
      recommender workflow remains unchanged.

# XTQSPRKinetic v2.6.0 — What's New

1) Generalized the v2.5.19 reviewer-response diagnostic layer into a public
   Diagnostic Audit module.

2) Renamed reviewer_* configuration keys to diagnostics_* keys.

3) Renamed run_reviewer_diagnostics(...) to run_diagnostics(...).

4) Renamed reviewer_diagnostics/ output folder to diagnostics/.

5) Renamed reviewer_diagnostics_runtime.json to diagnostics_runtime.json.

6) Added/retained network-preserving diagnostic outputs for:
    - dataset and target summary,
    - learned-family/RMG-family alignment,
    - bootstrap stability summaries,
    - k-sensitivity analysis,
    - size-matched random-cluster controls,
    - component-wise ablation,
    - descriptor-sensitivity analysis,
    - residual/error audit.

7) Preserved the v2.5.18 full-network training philosophy:
   cross-validation remains an offline hyperparameter-recommendation procedure,
   not the final estimator of external transferability.

8) Added hydrogen-radical recombination special-case support
    - Added a targeted chemistry patch for the reaction:
        [H].[H]>>[H][H]
    - This prevents the reaction from being treated as chemically empty when
      normal DRFP/Mordred descriptors become zero or non-informative.

9) Added explicit hydrogen/radical reaction descriptors
    - New reaction-level descriptors are appended through the existing flags_df
      feature pathway:
        rxn_H_atom_delta
        rxn_H_radical_electron_delta
        rxn_HH_bond_delta
        rxn_isolated_H_radical_atom_delta
        rxn_is_2H_radicals_to_H2
    - For [H].[H]>>[H][H], these encode:
        H atom delta                  =  0
        radical electron delta        = -2
        H-H bond delta                = +1
        isolated H radical atom delta = -2
        special-case flag             =  1

10) Added zero-DRFP repair for hydrogen-only radical recombination
    - When DRFP returns an all-zero vector for [H].[H]>>[H][H], deterministic
      sentinel bits are activated.
    - This prevents the reaction from being dropped by drfp_drop_zero_rows=True
      or routed as an all-zero/featureless DRFP sample.
    - The repair is applied only to the exact hydrogen recombination case and
      only when the DRFP vector is all-zero.

11) Added configuration switch
    - New DEFAULT_CONFIG key:
        hydrogen_radical_special_case_enable: True
    - Setting this to False restores the previous behavior.

12) Preserved existing pipeline structure
    - No changes were made to the main Mordred workflow.
    - No changes were made to the model training, Boruta, topic routing,
      diagnostics, SHAP, or pipeline.pkl structure.
    - The patch uses the existing feature assembly path:
        Mordred descriptors + optional DRFP + flags_df

13) Improved chemical interpretability for hydrogen-only reactions
    - The model now receives the chemically meaningful transformation:
        2 H· radicals → H2
    - This explicitly captures radical-electron consumption and H-H bond
      formation, which are not reliably represented by standard molecular
      descriptors alone.

14) Backward compatibility
    - Existing datasets and normal reaction SMILES are unaffected.
    - The special DRFP repair activates only for:
        [H].[H]>>[H][H]
    - Additional hydrogen-radical descriptors are zero-valued for reactions that
      do not contain this special transformation.

15) Added manuscript-ready descriptive figure module
    - Added an optional post-hoc descriptive-figure workflow controlled by:
        descriptive_figures_enable
    - The module complements SHAP rather than replacing it:
        • SHAP explains feature contributions to model predictions.
        • Descriptive figures audit the learned reaction-family manifold,
          descriptor signatures, routing uncertainty, and residual behavior.
    - New outputs are written to:
        output_dir/descriptive_figures/

16) Added new descriptive-figure configuration keys
    - New DEFAULT_CONFIG keys:
        descriptive_figures_enable: False
        descriptive_figures_dpi: 600
        descriptive_figures_font_size: 12
        descriptive_figures_max_features: 20
        descriptive_figures_boxplot_max_features: 6
        descriptive_figures_pca_sample_size: None
    - Default behavior is intentionally tuning-safe:
        descriptive_figures_enable=False writes only a compact summary CSV.
        descriptive_figures_enable=True writes full PNG figures and related CSV data.

17) Added lightweight descriptive summary output
    - When descriptive_figures_enable=False, the pipeline writes:
        descriptive_figures/descriptive_figures_summary.csv
    - The summary records:
        • number of reactions,
        • number of available confirmed features,
        • learned-family counts,
        • Arrhenius target summaries,
        • residual summaries when predictions are available,
        • routing-probability availability,
        • bootstrap-stability availability,
        • RMG/mechanistic-family label availability.
    - This preserves a low-cost audit trail during preliminary tuning runs.

18) Added reaction-family landscape figures
    - Added PCA-based two-dimensional reaction-feature landscape outputs:
        descriptive_figures/family_landscape_pca.csv
        descriptive_figures/family_landscape_by_learned_family.png
        descriptive_figures/family_landscape_by_lnA.png
        descriptive_figures/family_landscape_by_n.png
        descriptive_figures/family_landscape_by_Ea.png
    - These figures show whether learned reaction families and kinetic targets
      organize coherently in the selected descriptor/fingerprint feature space.
    - The PCA projection is used only as a visualization/audit tool, not as
      external validation of generalization.

19) Added residual-colored landscape figures
    - Added PCA landscape plots colored by absolute prediction error:
        descriptive_figures/family_landscape_by_lnA_absolute_error.png
        descriptive_figures/family_landscape_by_n_absolute_error.png
        descriptive_figures/family_landscape_by_Ea_absolute_error.png
    - These plots help identify regions of the learned reaction manifold where
      the specialist/fallback models show larger prediction errors.

20) Added family descriptor-signature heatmap
    - Added family-wise robust-z descriptor signature outputs:
        descriptive_figures/family_descriptor_heatmap_median_robust_z.csv
        descriptive_figures/family_descriptor_heatmap.png
    - Features are selected from confirmed model features using family-separation
      scoring, so the plot focuses on descriptors that vary meaningfully across
      learned reaction families.
    - This provides a chemically interpretable summary of how learned families
      differ in descriptor space.

21) Added descriptor boxplots by learned family
    - Added descriptor distribution outputs:
        descriptive_figures/descriptor_boxplots_by_family_data.csv
        descriptive_figures/descriptor_boxplots_by_family.png
    - These plots show the distribution of selected descriptors across learned
      families and provide a reaction-family-adapted analogue of class-wise
      descriptor boxplots.

22) Added routing-probability heatmap
    - When topic/routing probabilities are available, the module writes:
        descriptive_figures/routing_probability_heatmap_data.csv
        descriptive_figures/routing_probability_heatmap.png
    - This plot visualizes hard vs soft family assignment behavior and highlights
      reactions with mixed membership across learned families.

23) Added residual vs routing-uncertainty diagnostic
    - When routing probabilities and prediction residuals are available, the
      module writes:
        descriptive_figures/residual_vs_routing_uncertainty_data.csv
        descriptive_figures/residual_vs_routing_uncertainty.png
    - Routing uncertainty is computed from normalized entropy of the family
      probability vector.
    - This diagnostic helps test whether boundary/ambiguous reactions also show
      larger kinetic prediction errors.

24) Preserved original SHAP computational-control behavior
    - The original SHAP workflow is preserved:
        shap_sample_size = 0
            disables expensive regular SHAP calculations.
        shap_sample_size > 0
            enables regular SHAP analysis.
        shap_soft_enable = False
            disables only optional soft-probability-weighted SHAP aggregation.
        shap_soft_enable = True
            enables optional soft SHAP aggregation when routing probabilities
            and per-family SHAP files are available.
    - shap_soft_enable does not control the core SHAP calculation.

25) Added paper-ready SHAP figure formatting
    - Added optional SHAP figure-format controls:
        shap_figure_dpi: 600
        shap_figure_font_size: 12
        shap_figure_max_display: 20
    - SHAP figures are saved with bold fonts and minimum 300 dpi resolution.
    - Existing SHAP CSV outputs and SHAP computation logic are preserved.

26) Added paper-ready descriptive-figure formatting
    - All new descriptive PNG figures use bold fonts and minimum 300 dpi
      resolution, with 600 dpi as the default.
    - This makes the new plots suitable for manuscript and reviewer-response
      figure preparation without external post-processing.

27) Added optional soft SHAP PNG output
    - The existing soft SHAP CSV aggregation is retained:
        shap/soft_mean_abs_shap.csv
    - A new manuscript-ready bar plot is added when soft SHAP succeeds:
        shap/soft_mean_abs_shap.png

28) Tuning-safe auto-chooser behavior preserved
    - Fast candidate evaluation now explicitly disables heavy post-hoc outputs:
        shap_sample_size = 0
        descriptive_figures_enable = False
    - This keeps preliminary family-scheme/hyperparameter screening lightweight.
    - The final selected full run respects the user’s base configuration for
      shap_sample_size and descriptive_figures_enable.

29) No changes to core modeling logic
    - No intentional changes were made to:
        • Mordred descriptor generation,
        • DRFP generation,
        • hydrogen-radical special-case features,
        • feature assembly,
        • Boruta selection,
        • topic/family assignment,
        • LDA/FCM/GMM/HDBSCAN/QDA routing,
        • bootstrap stability logic,
        • XGBoost training,
        • global fallback training,
        • prediction equations,
        • metric calculations,
        • diagnostic audit logic,
        • pipeline.pkl saving structure.
    - The new functionality is post-hoc and diagnostic/visualization-oriented.

# XTQSPRKinetic v2.7.0 — What's New

1) Generalized target-property prediction framework
   - XTQSPRKinetic is no longer hard-limited to the three Arrhenius parameters:
       lnA, n, Ea
   - Added the canonical target-column configuration key:
       target_cols
   - Default behavior remains backward-compatible:
       target_cols: ['lnA', 'n', 'Ea']
   - The original Arrhenius-parameter workflow is now the default case study of
     a broader multi-target reaction-property prediction framework.

2) User-defined continuous reaction-level targets
   - Users can specify any numeric reaction-level regression targets present in
     the input CSV, for example:
       target_cols: ['Ea_forward', 'Ea_reverse', 'delta_H_rxn']
       target_cols: ['log_k_1000K', 'barrier_height', 'reaction_enthalpy']
   - The framework is intended for continuous supervised regression targets.
   - Classification labels, temperature-dependent curves, time-series outputs,
     and probabilistic target distributions are not automatically modeled in
     this version.

3) Centralized target resolution
   - Added a unified target-resolution pathway so that model training, feature
     selection, prediction, metrics, diagnostics, SHAP, descriptive figures, and
     pipeline metadata use the same target list.
   - Backward compatibility is preserved by falling back to:
       hp_target_cols
     and then to:
       ['lnA', 'n', 'Ea']
     when target_cols is not explicitly provided.
   - hp_target_cols remains supported for compatibility with the existing
     hyperparameter recommender workflow.

4) Generalized multi-output model training
   - The existing MultiOutputXGBRegressor workflow now supports arbitrary
     target-column lists.
   - One XGBoost regressor is trained per target dimension inside each global
     or family-specialist model.
   - Per-family specialists and the global fallback model infer the number and
     names of targets from target_cols instead of assuming exactly three outputs.

5) Generalized hard, soft, and hybrid prediction outputs
   - Prediction functions now generate output columns dynamically:
       <target>_pred
   - For the default Arrhenius case, outputs remain:
       lnA_pred, n_pred, Ea_pred
   - For custom targets, predictions are named automatically, for example:
       delta_H_rxn_pred
       barrier_height_pred
       log_k_1000K_pred
   - Hard routing, soft routing, and hybrid routing remain compatible with the
     generalized output dimension.

6) Generalized prediction and error tables
   - full_prediction_output.csv now supports arbitrary target columns.
   - Error columns are generated dynamically for each target:
       <target>_abs_err
       <target>_abs_rel_err
       <target>_smape
   - relerr_mode continues to control whether absolute relative error, sMAPE,
     or both are written.
   - Target-safe file names are generated for custom target names.

7) Generalized global, family, and soft-weighted metrics
   - global_stats.csv, family_stats.csv, and family_stats_soft.csv now loop over
     target_cols rather than fixed Arrhenius target names.
   - Metrics remain consistent with the v2.6.x reporting set:
       R2, AdjR2, MAE, RMSE, nRMSE, AARD, RAE
   - Adjusted R2 continues to use the number of selected model features when
     available.
   - Weighted metrics are computed for each target using the active routing
     probability matrix when soft-routing probabilities are available.

8) Generalized SHAP analysis
   - SHAP outputs now use configured target names instead of hard-coded
     Arrhenius names.
   - Per-family SHAP CSV and PNG artifacts are named using the target column:
       shap_values_family_<family>_<target>.csv
       shap_values_family_<family>_<target>.png
       mean_abs_shap_values_family_<family>_<target>.png
   - Global and per-family SHAP behavior remain controlled by shap_sample_size.
   - shap_soft_enable continues to control only optional soft-routing-weighted
     SHAP aggregation.

9) Generalized descriptive figures
   - Target-colored PCA landscape plots are now generated for every configured
     target:
       family_landscape_by_<target>.png
   - Residual-colored landscape plots are generated dynamically:
       family_landscape_by_<target>_absolute_error.png
   - The descriptor heatmap, descriptor boxplots, routing probability heatmaps,
     and residual-vs-routing-uncertainty diagnostics remain compatible with the
     generalized target workflow.

10) Generalized residual diagnostics
    - Residual audit outputs now loop over target_cols.
    - Top-residual files are generated dynamically:
        top_residuals_<target>.csv
    - Residual summaries by learned family and by RMG/mechanistic family now
      support arbitrary configured targets.
    - This preserves the v2.6.0 diagnostic-audit philosophy while removing the
      fixed lnA/n/Ea assumption.

11) Generalized hyperparameter recommender integration
    - The offline hyperparameter recommender now uses the resolved target list.
    - If target_cols is provided, it becomes the canonical target list for
      recommendation, training, final fitting, and reporting.
    - hp_target_cols remains supported for backward compatibility.
    - Per-target recommended XGBoost parameters continue to be mapped by target
      name and target index.

12) Generalized pipeline.pkl metadata
    - pipeline.pkl now stores the configured target list:
        target_cols
    - This allows downstream evaluators and post-processing tools to recover the
      correct output names and number of model outputs.
    - The config_snapshot also records target_cols for reproducibility.
    - Existing metadata for models, scalers, topic assigners, features, Boruta
      provenance, routing mode, diagnostics, and compression remains unchanged.

13) Generalized companion evaluator
    - The Arrhenius-specific kinetics-performance evaluator has been generalized
      for arbitrary continuous reaction-property targets.
    - Recommended evaluator name:
        reaction_property_performance.py
    - A compatibility-named evaluator can be retained as:
        kinetics_performance_v2.7.0.py
    - The evaluator reads target_cols from pipeline.pkl / config_snapshot and
      writes prediction, error, and metric columns dynamically.

14) Backward-compatible Arrhenius case study
    - The default configuration remains:
        target_cols: ['lnA', 'n', 'Ea']
    - Existing Arrhenius datasets and workflows continue to run without requiring
      users to specify target_cols manually.
    - Default output names remain consistent:
        lnA_pred, n_pred, Ea_pred
        lnA_abs_err, n_abs_err, Ea_abs_err
        lnA_smape, n_smape, Ea_smape

15) Added config-controlled parallel Mordred descriptor calculation
    - Added optional joblib-based parallel execution for reaction-level Mordred
      descriptor construction.
    - New DEFAULT_CONFIG keys:
        mordred_parallel_enable: False
        mordred_parallel_n_jobs: -1
        mordred_parallel_backend: 'loky'
        mordred_parallel_batch_size: 'auto'
    - Default behavior remains sequential unless the user explicitly enables
      parallel Mordred calculation.

16) Added worker-safe Mordred descriptor helper layer
    - Added reusable helper functions for process/thread-safe Mordred calculation:
        _get_mordred_worker_state(...)
        _mordred_descriptor_columns(...)
        _mordred_safe_values(...)
        _mordred_reaction_descriptor_row(...)
    - Each worker lazily constructs its own Mordred Calculator and descriptor-name
      list.
    - This avoids repeatedly rebuilding the descriptor list for every reaction
      and supports process-based joblib execution with the recommended 'loky'
      backend.

17) Parallel Mordred preserves existing descriptor behavior
    - The revised Mordred workflow preserves all existing aggregation choices:
        mordred_agg: 'sum'
        mordred_agg: 'concat'
        mordred_agg: 'delta'
    - Column naming remains consistent with the selected aggregation mode:
        <descriptor>_sum
        R_<descriptor> and P_<descriptor>
        <descriptor>_delta
    - Existing missing-value policies are preserved:
        mean
        median
        zero
        drop
    - Infinite values are still converted to NaN before missing-value handling.
    - Output remains a numeric pandas DataFrame compatible with feature assembly,
      Boruta selection, diagnostics, SHAP, and model training.

18) Improved Mordred descriptor cache key and cache metadata
    - Added a dedicated Mordred cache-key helper:
        _mordred_cache_key(...)
    - The new cache key depends on:
        reaction SMILES values,
        reaction index/order,
        Mordred aggregation mode,
        missing-value policy,
        2D/3D Mordred mode.
    - Parallel settings are intentionally not included in the cache key because
      they should change runtime speed only, not descriptor values.
    - get_or_compute_mordred(...) now forwards the parallel Mordred options into
      calculate_descriptors_mordred(...).
    - Existing cache loading behavior is preserved: if a valid Mordred cache
      exists, descriptors are loaded from cache and are not recomputed.
    - Cache metadata records:
        mordred_parallel_enable
        mordred_parallel_n_jobs
        mordred_parallel_backend
        rows
        cols
    - The old reaction-fingerprint cache helper can be removed once all Mordred
      cache calls use _mordred_cache_key(...).

19) Mordred 3D compatibility retained inside the parallel worker
    - The revised Mordred worker respects the existing XT_MORDRED_3D environment
      pathway used by the pipeline when:
        mordred_3d_enable: True
    - If 3D descriptors are enabled, the worker attempts hydrogen addition,
      conformer embedding, and UFF optimization before descriptor evaluation.
    - If conformer generation fails for a molecule, the worker safely falls back
      rather than stopping the full pipeline.

20) Added config-controlled parallel Boruta feature selection
    - Added optional joblib-based parallel execution for independent per-target
      Boruta runs.
    - New DEFAULT_CONFIG keys:
        boruta_parallel_enable: False
        boruta_parallel_targets_enable: True
        boruta_parallel_n_jobs: -1
        boruta_parallel_backend: 'loky'
        boruta_parallel_inner_n_jobs: 1
    - Default behavior remains sequential unless the user explicitly enables
      Boruta parallelization.
    - This naming is intentionally specific to Boruta and is separate from the
      mordred_parallel_* settings.

21) Added oversubscription-safe Boruta parallel execution
    - When target-wise Boruta is parallelized, each target worker receives a
      local copy of the configuration.
    - The inner estimator threading is controlled by:
        boruta_parallel_inner_n_jobs
    - This value is applied to Boruta RF/LGBM estimator n_jobs inside each worker
      to avoid excessive CPU oversubscription.
    - Recommended setting for outer process-based Boruta parallelism:
        boruta_parallel_enable: True
        boruta_parallel_n_jobs: -1
        boruta_parallel_backend: loky
        boruta_parallel_inner_n_jobs: 1

22) Parallel Boruta preserves feature-selection logic
    - The Boruta algorithm, per-target feature-confirmation workflow, feature
      union logic, cache behavior, and Boruta provenance reporting are preserved.
    - The new option changes only the scheduling of independent target-wise
      Boruta calculations.
    - With fixed random_state and the same software environment, the scientific
      feature-selection procedure is intended to remain equivalent to the
      sequential workflow.

23) Added family-level median routing-probability heatmap
    - Added a compact summary heatmap companion to the existing reaction-level
      routing-probability heatmap.
    - New outputs:
        descriptive_figures/routing_probability_family_median_heatmap_data.csv
        descriptive_figures/routing_probability_family_median_heatmap.png
    - Rows correspond to learned families.
    - Columns correspond to routing components/topics.
    - Cell values are median routing probabilities within each learned family.
    - Numeric cell labels are printed directly in the heatmap to improve
      manuscript/SI readability.

24) Preserved the original reaction-level routing heatmap
    - The original detailed heatmap remains unchanged:
        descriptive_figures/routing_probability_heatmap_data.csv
        descriptive_figures/routing_probability_heatmap.png
    - The reaction-level figure shows individual reaction membership patterns,
      boundary reactions, and mixed-membership behavior.
    - The family-level median heatmap summarizes the typical routing signature of
      each learned family.
    - Together, the two figures support both fine-grained routing diagnostics and
      clean manuscript-level interpretation.

25) Improved routing-figure interpretability
    - The reaction-level heatmap can appear visually noisy because it preserves
      one row per reaction.
    - The new family-level median heatmap reduces this complexity to a compact
      family-by-topic matrix.
    - This helps readers identify which routing topics dominate each learned
      family and which learned families have mixed or diffuse routing behavior.
    - The new figure uses the existing publication-ready plotting utilities,
      bold text, and the descriptive_figures_dpi setting with minimum 300 dpi
      behavior preserved.

26) Extended joblib imports for controlled parallel execution
    - Added joblib parallel utilities:
        Parallel
        delayed
        parallel_config
    - Parallel and delayed are used for the Mordred descriptor worker loop and
      the optional independent per-target Boruta worker loop.
    - The main pipeline still avoids uncontrolled global multiprocessing; only
      explicitly enabled independent computational blocks are parallelized.

27) Improved timing-decorator robustness
    - timeit_logger(...) now uses functools.wraps.
    - This preserves decorated function metadata such as function name and
      docstring.
    - Runtime behavior and scientific outputs are unchanged.

28) Updated scientific framing
    - XTQSPRKinetic can now be described as a configurable multi-target
      reaction-property prediction framework.
    - The Arrhenius-parameter workflow is retained as the default kinetics case
      study.
    - Recommended wording:
        "XTQSPRKinetic supports user-defined continuous reaction-level target
          properties, provided they are present in the input dataset and suitable
          for supervised regression from the available reaction descriptors and
          fingerprints."

29) Scope and limitations clarified
    - v2.7.0 supports continuous numeric regression targets.
    - It does not automatically convert the pipeline into a classifier.
    - It does not automatically model temperature-dependent curves or structured
      outputs.
    - Future versions may add target-specific transformations, classification
      targets, uncertainty-aware regression, or functional/temperature-dependent
      target models.

30) No intended change to core scientific modeling logic
    - No intentional changes were made to:
        DRFP generation,
        LDA / FCM / GMM / HDBSCAN / QDA family discovery,
        bootstrap stability logic,
        hydrogen-radical special-case descriptors,
        zero-DRFP repair,
        reaction flags,
        descriptor prefiltering,
        Boruta feature-selection logic,
        specialist XGBoost training,
        global fallback training,
        hard / soft / hybrid prediction equations,
        SHAP calculations,
        diagnostic audit metrics,
        pipeline.pkl schema.
    - v2.7.0 generalizes the target/reporting layer, accelerates optional
      Mordred descriptor construction, optionally parallelizes independent
      per-target Boruta calculations, and improves routing visualization while
      preserving the v2.6.0 reaction-network modeling foundation.

31) Recommended migration and run modes from v2.6.0
    - Existing Arrhenius workflows can continue unchanged with:
        target_cols: ['lnA', 'n', 'Ea']
    - To train on different reaction properties, add the target columns to the
      input CSV and set:
        target_cols: ['target_1', 'target_2', ...]
    - If mechanistic/RMG-family diagnostics are desired, include the relevant
      reference-family column in the input CSV and set:
        rmg_family_col: 'rmg_family'
      or set rmg_family_col to the actual column name used in the dataset.
    - Preliminary auto-chooser or hyperparameter screening is optional.
    - If the family scheme and hyperparameters are already chosen manually, the
      pipeline can be run directly with the selected/default configuration.
    - For computational efficiency during repeated screening runs, users may set:
        shap_sample_size: 0
        descriptive_figures_enable: False
      and then enable SHAP/descriptive figures only for final reporting.

32) Recommended parallel Mordred and Boruta use
    - For normal reproducible runs:
        mordred_parallel_enable: False
        boruta_parallel_enable: False
    - For large datasets or repeated final runs with descriptor recomputation:
        mordred_parallel_enable: True
        mordred_parallel_n_jobs: -1
        mordred_parallel_backend: loky
        mordred_parallel_batch_size: auto
    - For large multi-target Boruta runs:
        boruta_parallel_enable: True
        boruta_parallel_targets_enable: True
        boruta_parallel_n_jobs: -1
        boruta_parallel_backend: loky
        boruta_parallel_inner_n_jobs: 1
    - If descriptor_cache is True and a valid Mordred cache already exists, the
      cache is loaded and Mordred descriptors are not recomputed; therefore,
      Mordred parallel speedup is visible only when descriptors are recomputed.
    - Boruta parallel speedup is visible when boruta_per_target=True and more
      than one target is configured.

33) Added XTQSPRKinetic explicit-hydrogen-aware Mordred descriptor mode
    - Added an optional runtime patch pathway for Mordred / MordredCommunity
      descriptor calculation.
    - New public configuration key:
        mordred_xtqspr_explicit_h_patch_enable: False
    - When enabled, XTQSPRKinetic applies a controlled runtime patch before
      constructing the Mordred Calculator.
    - This avoids manual modification of the installed public Mordred or
      MordredCommunity packages after installation.
    - The patch is intended for radical-rich C/H reaction networks where default
      Mordred hydrogen handling can underrepresent hydrogen-sensitive species
      such as:
        CH4, [CH3], [CH2], [CH], [H], and [H][H].

34) Added runtime Mordred patch module in the package
    - Added:
        src/mordred_xtqspr_patch.py
    - The module applies the XTQSPRKinetic explicit-Hydrogen-aware Mordred
      modifications at runtime.
    - The patch is applied before the local Mordred Calculator is constructed.
    - This is compatible with both sequential and joblib/loky parallel Mordred
      descriptor generation because each worker process applies the patch before
      creating its own Calculator.

35) Replaced manual site-packages editing with controlled runtime behavior
    - The previous manual approach required editing Mordred/MordredCommunity
      source files inside the Python environment.
    - v2.7.0 now supports the same descriptor mode through XTQSPRKinetic code.
    - This improves reproducibility, portability, and package maintenance.
    - Users can enable or disable the modified descriptor mode directly from
      the YAML configuration.

36) Explicit-hydrogen descriptor-class behavior is now patched systematically
    - All Mordred descriptor classes that originally defined:
        explicit_hydrogens = False
      are converted at runtime to:
        explicit_hydrogens = True
      when the XTQSPRKinetic Mordred patch is enabled.
    - Descriptor classes that were already:
        explicit_hydrogens = True
      in original Mordred are preserved and listed only for audit completeness.
    - No new explicit_hydrogens attributes are created in classes where the
      original Mordred code did not define them.

37) Corrected atomic-property handling for C/H radical systems
    - The runtime patch modifies selected Mordred Mordred code did not define them.

38) Corrected atomic-property handling for C/H radical systems
    - The runtime patch modifies atomic-property helper behavior
      to improve sensitivity to hydrogen-rich and radical species.
    - Hydrogen atoms now contribute one valence electron instead of zero.
    - Non-hydrogen atoms retain the original hydrogen-corrected Mordred valence
      expression rather than using the broader raw outer-valence-electron
      replacement.
    - Sigma-neighbor counting includes explicit hydrogen neighbors.
    - Intrinsic-state zero-connectivity cases are regularized to avoid descriptor
      collapse for radical/H-only species.
    - Undefined valence cases with zero denominator remain NaN, preserving the
      distinction between mathematically undefined and chemically zero values.

39) Removed reliance on Mordred Calculator.config(explicit_hydrogens=True)
    - Validation showed that:
        calc.config(explicit_hydrogens=True)
      did not reproduce the manually patched descriptor behavior for the target
      radical-rich C/H reaction network.
    - Therefore, XTQSPRKinetic no longer relies on this public Mordred config
      pathway for the explicit-H descriptor mode.
    - The descriptor modification is instead handled by the controlled
      XTQSPRKinetic runtime patch.

40) Mordred cache key now protects patched and unpatched descriptor modes
    - The Mordred descriptor cache key now includes:
        mordred_xtqspr_explicit_h_patch
    - This prevents accidental reuse of descriptors generated under the wrong
      descriptor mode.
    - Patched and unpatched Mordred descriptor matrices are treated as distinct
      cache states.
    - Users should use a fresh cache, or disable descriptor caching, for the
      first run after enabling the explicit-H patch mode.

41) Training pipeline and external evaluator are synchronized
    - The main XTQSPRKinetic pipeline passes:
        mordred_xtqspr_explicit_h_patch_enable
      into the Mordred descriptor builder.
    - The companion evaluator:
        reaction_property_performance.py
      also supports the same flag.
    - This ensures that external prediction/evaluation using pipeline.pkl
      reconstructs features using the same descriptor mode used during training.

42) Split and monolithic implementations synchronized
    - The explicit-H Mordred runtime-patch pathway was added to both:
        XTQSPRKinetic_V.2.7.0.py
        split-package implementation under src/
    - The key functions are synchronized between the monolithic and split
      versions, including:
        apply_xtqspr_mordred_patch
        _get_mordred_worker_state
        _mordred_cache_key
        calculate_descriptors_mordred
        get_or_compute_mordred
        run_pipeline
    - The split-only extra functions remain limited to repository-driver logic
      in main.py.

43) Scientific descriptor naming clarified
    - When:
        mordred_xtqspr_explicit_h_patch_enable: true
      is used, the descriptors should be described as:
        XTQSPRKinetic explicit-hydrogen-aware modified Mordred/MordredCommunity
        descriptors
      rather than fully standard Mordred descriptors.
    - This clarification is important because the patch intentionally changes
      hydrogen handling and selected atomic-property behavior for radical-rich
      C/H chemistry.

44) Recommended Mordred patch module usage:
    - For standard Mordred behavior:
        mordred_xtqspr_explicit_h_patch_enable: false
    - For radical-rich C/H kinetic networks:
        mordred_xtqspr_explicit_h_patch_enable: true
    - For the first patched-descriptor run:
        descriptor_cache: false or use a fresh descriptor_cache_dir.

45) Refined Bootstrap routing control layer
    - Added explicit configuration switches to separate Bootstrap stability
      diagnostics from predictive routing:
        bootstrap_use_for_soft_routing
        bootstrap_use_for_hybrid_routing
    - Bootstrap artifacts can now be generated for audit purposes without being
      automatically injected into soft-routing predictions.
    - This prevents hidden prediction changes when Bootstrap is enabled only for
      stability analysis.

46) Added Bootstrap routing quality gate
    - Added a conservative quality-control layer before Bootstrap-guided hybrid
      routing is allowed.
    - New configuration keys:
        bootstrap_quality_gate_enable
        bootstrap_min_median_modal_vote
        bootstrap_max_median_norm_entropy
        bootstrap_min_hard_fraction
        bootstrap_max_hard_fraction
    - Hybrid routing now uses Bootstrap probabilities only when the Bootstrap
      stability table passes median modal-vote, normalized-entropy, and hard/soft
      split sanity checks.
    - If the gate fails, the pipeline falls back to the configured non-hybrid
      routing mode rather than forcing unstable Bootstrap-guided routing.

47) Separated requested routing mode from executed routing mode
    - The pipeline now distinguishes:
        routing_mode_requested
        routing_mode
    - routing_mode_requested records the routing mode requested by the user
      through configuration.
    - routing_mode records the routing mode actually executed after Bootstrap
      availability checks, quality gates, and hard-fraction checks.
    - This improves provenance when hybrid routing is requested but the pipeline
      safely falls back to soft or hard routing.

48) Stored actual routing mode in pipeline metadata
    - build_pipeline_components(...) and save_pipeline_components_from_config(...)
      now accept:
        routing_mode_actual
    - pipeline.pkl now stores the actual executed routing mode rather than
      reconstructing routing mode only from configuration flags.
    - pipeline.pkl additionally stores:
        routing_mode_requested
    - This prevents metadata mismatch when use_hybrid_routing=True but hybrid
      routing is disabled by Bootstrap quality gates.

49) Added robust probability-row normalization helper
    - Added _normalize_probability_rows(...) for finite, non-negative row-wise
      probability normalization.
    - The helper guards against NaN, infinity, negative values, empty probability
      rows, and zero-sum rows.
    - It is used in Bootstrap aggregation and prediction-time routing to reduce
      silent numerical failures in soft and hybrid routing.

50) Added configurable Bootstrap sampling semantics
    - Added:
        bootstrap_sampling_mode
    - Supported modes:
        bootstrap
        subsample
    - bootstrap performs classical sampling with replacement.
    - subsample performs sampling without replacement.
    - This allows users to compare classical Bootstrap stability with subsampling
      stability, especially for clustering methods sensitive to duplicate points.

51) Added method-aware Bootstrap enable resolution
    - Added _bootstrap_method_enabled(...) so method-specific Bootstrap switches
      consistently support None-style inheritance from bootstrap_lda_enable.
    - This makes bootstrap_qda_enable, bootstrap_gmm_enable,
      bootstrap_hdbscan_enable, and bootstrap_fcm_enable behavior consistent with
      the documented configuration semantics.
    - The update prevents None-valued method switches from being interpreted
      accidentally as False in candidate runs.

52) Improved Bootstrap-LDA sampling and probability normalization
    - bootstrap_lda_topics(...) now accepts:
        sampling_mode
    - Bootstrap-LDA now uses the shared sampling-mode helper.
    - LDA document-topic probabilities are normalized after transform.
    - Logging now reports the active Bootstrap sampling mode.
    - These changes preserve the original LDA stability workflow while improving
      reproducibility and numerical safety.

53) Improved Bootstrap-FCM sampling and aggregated membership normalization
    - bootstrap_fcm_topics(...) now accepts:
        sampling_mode
    - FCM stability analysis can now use either classical Bootstrap sampling or
      subsampling.
    - Aggregated FCM memberships are normalized before being returned.
    - This makes FCM Bootstrap outputs more consistent with other routing
      probability matrices.

54) Corrected Bootstrap-GMM component alignment
    - The Hungarian component-alignment logic for Bootstrap-GMM was corrected.
    - The revised mapping explicitly selects the Bootstrap component matched to
      each reference component.
    - This prevents probability columns from being silently reordered incorrectly
      when aligned GMM components are averaged across Bootstrap runs.
    - Bootstrap-GMM probabilities are also normalized after prediction and after
      aggregation.

55) Added sampling-mode support to Bootstrap-GMM
    - Bootstrap-GMM now respects:
        bootstrap_sampling_mode
    - Both classical sampling with replacement and subsampling without replacement
      are supported.
    - This allows users to evaluate whether GMM stability is affected by duplicate
      Bootstrap samples.

56) Improved Bootstrap-HDBSCAN full-dataset membership semantics
    - Bootstrap-HDBSCAN now attempts to compute membership vectors for the full
      dataset after each Bootstrap fit using HDBSCAN approximate prediction.
    - If full-dataset membership prediction is unavailable, the previous in-bag
      membership fallback is retained.
    - This makes Bootstrap-HDBSCAN stability semantics more comparable to LDA,
      FCM, GMM, and QDA, where each Bootstrap model contributes probabilities for
      all reactions.
    - Aggregated HDBSCAN probability matrices are normalized before output.

57) Improved Bootstrap-QDA robustness
    - bootstrap_qda_topics(...) now accepts:
        sampling_mode
    - Bootstrap-QDA bags are screened to ensure that sampled class labels contain
      at least two classes and sufficient samples per class before fitting QDA.
    - Failed QDA bags are skipped with warnings instead of crashing the entire
      Bootstrap run.
    - If no valid Bootstrap-QDA bags are produced, the routine falls back to
      deterministic QDA probabilities and records a warning.
    - Bootstrap-QDA probabilities are normalized before aggregation.

58) Added QDA feasibility gate before QDA fitting
    - Added a pre-fit feasibility check for the LDA+QDA path.
    - New configuration keys:
        qda_feasibility_gate_enable
        qda_min_class_count
        qda_auto_skip_infeasible
    - The gate checks whether LDA-derived family labels contain enough samples
      per present class before QDA is fitted.
    - This prevents sklearn QDA failures such as covariance-estimation errors
      caused by classes with only one sample.

59) Added Auto-Chooser skip behavior for infeasible QDA candidates
    - Added AutoChooserSkipCandidate to distinguish scientifically infeasible
      candidates from unexpected runtime errors.
    - In Auto-Chooser mode, infeasible QDA candidates can now be skipped cleanly
      before qda.fit(...) is called.
    - This avoids noisy candidate failures when LDA creates rare families that are
      too small for QDA covariance estimation.
    - In fixed/manual runs, infeasible QDA still raises a clear ValueError so the
      user is explicitly informed of the data limitation.

60) Added QDA feasibility reporting helpers
    - Added _qda_feasibility_report(...) to summarize class counts, minimum
      present class size, missing classes, and feasibility status.
    - Added _format_qda_feasibility_message(...) to produce clear diagnostic
      messages for skipped or invalid QDA candidates.
    - The feasibility message recommends reducing n_topics, removing the QDA
      candidate, or merging rare LDA families when QDA is not statistically
      valid.

61) Made hybrid routing hard path consistent with training families
    - predict_arrhenius_hybrid(...) now accepts:
        hard_topic_series
    - Stable samples in hybrid routing can now be routed through the family label
      actually used for specialist training.
    - run_pipeline passes:
        df['_training_family']
      into hybrid prediction.
    - This prevents mismatch between Bootstrap-derived modal families and the
      specialist models trained on baseline or replacement-controlled families.

62) Added index-safe routing probability alignment
    - Soft and hybrid prediction now align routing probability matrices by
      X_pred.index using reindex(...) before prediction.
    - This avoids silent row-position errors after filtering, reordering, or
      row-dropping operations.
    - Routing weights are normalized after reindexing to ensure valid probability
      rows.

63) Refined active routing probability selection
    - The pipeline now constructs:
        proba_for_routing
        proba_for_active_routing
    - proba_for_routing represents the probability matrix allowed for soft
      routing.
    - proba_for_active_routing represents the probability matrix actually used by
      the executed routing mode.
    - Soft routing uses Bootstrap probabilities only when explicitly enabled and
      quality-approved.
    - Hybrid routing uses Bootstrap probabilities only when hybrid routing is
      actually executed.

64) Corrected soft-weighted family statistics to use active routing probabilities
    - The soft family-statistics writer now uses proba_for_active_routing instead
      of always using the baseline topic probability matrix.
    - This ensures family_stats_soft.csv reflects the same routing probability
      matrix used during prediction.
    - Probability rows are reindexed to X_full.index and normalized before
      weighted metric calculation.

65) Corrected active topic-probability metadata
    - topic_proba_active_df now records the probability matrix actually used by
      the executed routing path.
    - If hybrid routing is disabled by quality gates, metadata no longer
      incorrectly stores Bootstrap probabilities as the active routing matrix.
    - This improves downstream evaluator consistency and pipeline provenance.

66) Improved hybrid-routing fallback behavior
    - When hybrid routing is requested but cannot be used, the pipeline now logs
      that it is falling back to soft or hard routing.
    - Fallback decisions are based on Bootstrap availability, Bootstrap quality
      gates, bootstrap_use_for_hybrid_routing, and hard-fraction sanity checks.
    - This prevents ambiguous runs where the configuration requests hybrid
      routing but the executed prediction mode differs.

67) Added hard-fraction sanity check for hybrid routing
    - Hybrid routing now checks the fraction of samples classified as hard-routed
      under the selected stability threshold.
    - If the hard fraction is too small or too large, hybrid routing is disabled
      and the pipeline falls back to the configured non-hybrid route.
    - This prevents degenerate hybrid behavior where almost all samples are routed
      hard or almost all samples are routed soft.

68) Preserved Bootstrap label-replacement safeguards
    - Bootstrap label replacement remains disabled by default.
    - The revised routing layer does not force replacement or retraining when
      Bootstrap diagnostics are enabled.
    - Existing controls are preserved:
        bootstrap_replace_family_labels
        bootstrap_replace_metric
        bootstrap_replace_threshold
        bootstrap_replace_keep_unstable
        bootstrap_retrain_on_replaced
    - This keeps Bootstrap stability analysis separate from supervised specialist
      retraining unless the user explicitly enables replacement behavior.

69) Improved Auto-Chooser safety for broad candidate screening
    - Auto-Chooser can now screen broad candidate sets containing QDA, Bootstrap,
      hybrid routing, GMM, FCM, and HDBSCAN paths with fewer avoidable runtime
      failures.
    - Scientifically infeasible QDA candidates are skipped rather than treated as
      unexpected crashes.
    - Bootstrap-guided routing candidates are protected by stability and
      hard-fraction gates.
    - This makes large all-method candidate sweeps more reliable and easier to
      audit.

70) Backward-compatible default behavior for non-Bootstrap runs
    - Standard non-Bootstrap hard or soft routing behavior is preserved.
    - Bootstrap probabilities are not used for soft routing by default.
    - Hybrid routing remains opt-in through:
        use_hybrid_routing
    - Bootstrap-guided hybrid routing remains opt-in and quality-gated through:
        bootstrap_use_for_hybrid_routing
        bootstrap_quality_gate_enable
    - Existing workflows that do not enable Bootstrap continue to behave as
      before.

71) Recommended safe Bootstrap/Hybrid execution mode
    - For conservative final runs, the recommended settings are:
        bootstrap_replace_family_labels: false
        bootstrap_retrain_on_replaced: false
        bootstrap_use_for_soft_routing: false
        bootstrap_use_for_hybrid_routing: true
        bootstrap_quality_gate_enable: true
    - This uses Bootstrap primarily to identify stable versus ambiguous routing
      cases while avoiding uncontrolled replacement of learned families.
    - Bootstrap remains a stability and routing-confidence diagnostic rather than
      an automatic guarantee of improved supervised prediction accuracy.

72) Scope of the Bootstrap/QDA refinement
    - The refinement is focused on routing correctness, stability diagnostics,
      Bootstrap probability safety, QDA feasibility, and pipeline provenance.
    - No intentional changes were made to:
        Mordred descriptor generation,
        DRFP generation,
        hydrogen-radical special-case descriptors,
        descriptor prefiltering,
        Boruta feature-selection logic,
        XGBoost specialist training,
        global fallback model training,
        target-column generalization,
        SHAP computation,
        diagnostic-audit metric definitions,
        or the core supervised regression targets.
    - The changes improve safety and reproducibility around Bootstrap-guided
      family routing without redefining the core XTQSPRKinetic modeling workflow.

73) Config-controlled descriptive heatmap colormaps and annotation contrast
    - Added new DEFAULT_CONFIG keys for manuscript-ready control of descriptive
      heatmap appearance:
        descriptive_figures_heatmap_cmap
        descriptive_figures_descriptor_heatmap_cmap
        descriptive_figures_routing_heatmap_cmap
        descriptive_figures_routing_family_heatmap_cmap
        descriptive_figures_heatmap_annotation_auto_contrast
    - Descriptor-signature, reaction-level routing-probability, and family-level
      median routing-probability heatmaps can now use separate colormap choices
      from the run configuration.
    - Added automatic black/white annotation contrast for numeric labels in
      heatmap cells, improving readability when dark or diverging colormaps are
      used.
    - This change affects descriptive figure rendering only and does not change
      descriptor generation, family discovery, feature selection, model training,
      prediction, routing probabilities, or reported metrics.

74) QDA feasibility gate and cleaner Auto-Chooser failure handling
    - Added explicit QDA feasibility safeguards for LDA+QDA routes:
        qda_feasibility_gate_enable
        qda_min_class_count
        qda_auto_skip_infeasible
    - LDA-derived class counts are checked before Quadratic Discriminant Analysis
      is fitted, preventing low-level covariance/class-size failures when one or
      more learned families are too small for stable QDA estimation.
    - In Auto-Chooser mode, infeasible QDA candidates are skipped cleanly and
      recorded as data-feasibility failures rather than unexpected runtime errors.
    - In fixed/manual LDA+QDA runs, the pipeline raises a clear feasibility error
      unless automatic skipping is explicitly enabled.

75) Finite metric safeguards for degenerate statistical cases
    - Added finite-safe metric helpers for R2, adjusted R2, MAE, RMSE, nRMSE,
      AARD, RAE, and weighted routing metrics.
    - Degenerate cases such as one-sample families, constant target values,
      zero target means, zero relative-error denominators, non-finite prediction
      pairs, and n <= p + 1 adjusted-R2 regimes are now handled with explicit
      finite fallbacks instead of propagating NaN/inf values into CSV metrics.
    - The safeguards apply to global, family-wise, soft-weighted, diagnostic, and
      evaluation metric tables where applicable.

76) Pandas concat future-compatibility cleanup
    - Updated family-model importance table construction to avoid iterative
      concatenation into an initially empty DataFrame.
    - Feature-importance rows are now accumulated in a list and concatenated once,
      preventing the pandas FutureWarning about DataFrame concatenation with
      empty or all-NA entries.
    - This change affects only the writing of family_model_formula_<family>.csv
      artifacts and does not change model training, prediction, feature
      selection, routing, SHAP, metrics, or pipeline.pkl contents.

77) Logging and dependency-warning hygiene
    - Added a narrow warning filter for the scikit-learn >=1.6 deprecation message:
        'force_all_finite' was renamed to 'ensure_all_finite'
    - This warning can be emitted repeatedly through HDBSCAN / Bootstrap-HDBSCAN
      validation paths and does not indicate a data-quality, descriptor, routing,
      or model-training error.
    - The filter is intentionally message-specific and does not suppress general
      FutureWarning messages.
    - Scientifically relevant warnings, such as HDBSCAN producing a single
      probability column or degenerate clustering/routing behavior, remain visible
      in the pipeline log.

78) Added config-controlled manuscript-facing figure label indexing
    - Added a figure-only display-label control:
        figure_label_index_base
    - This option controls whether learned-family and topic labels shown in
      manuscript figures are displayed using internal zero-based numbering or
      manuscript-friendly one-based numbering.
    - Supported values:
        figure_label_index_base: 0
            show internal labels such as Family 0 and Topic 0.
        figure_label_index_base: 1
            show manuscript-facing labels such as Family 1 and Topic 1.
    - The setting affects only visual labels in PNG figure axes, legends, and
      titles, including descriptive diagnostic figures and SHAP figure titles.
    - Internal identifiers remain unchanged:
        • CSV columns,
        • learned-family assignments,
        • topic-probability matrices,
        • SHAP filenames,
        • model/scaler dictionaries,
        • pipeline.pkl metadata.
    - This preserves computational reproducibility while allowing figure labels
      to match the one-based numbering convention used in the manuscript.

79) Improved Auto-Chooser candidate-output organization
    - Auto-Chooser candidate folders now use the user-defined candidate name from
      candidate_schemes as a human-readable prefix.
    - Candidate subfolders retain the candidate index and topic-method signature
      for uniqueness and reproducibility.
    - The parent output directory name is no longer repeated inside each candidate
      subfolder, improving readability during multi-candidate sweeps.

80) Added selected-candidate provenance to the final configuration
    - The final Auto-Chooser-selected configuration now records the winning
      candidate metadata directly in final_cfg.
    - New runtime provenance fields include:
        auto_selected_candidate_index
        auto_selected_candidate_name
        auto_selected_candidate_score
        auto_selected_candidate_signature
        auto_selected_candidate_output_dir
    - These fields document which fast-evaluation candidate was selected and why,
      without requiring users to inspect logs manually.
    - These keys are generated runtime metadata and do not need to be added to the
      user-facing DEFAULT_CONFIG block.

81) Added candidate-name and candidate-signature tracking to the Auto-Chooser
   leaderboard
    - auto_chooser_leaderboard.csv now includes the candidate name and topic-method
      signature for each scored candidate.
    - This improves interpretation when multiple candidates share the same broad
      topic method but differ in routing, Bootstrap, FCM, GMM, HDBSCAN, QDA, or
      hybrid settings.

82) Fixed best-row tracking for winner metadata
    - Auto-Chooser now stores the full leaderboard row associated with the current
      best-scoring candidate.
    - This prevents undefined-variable errors when saving selected-candidate
      metadata.
    - The winning row is deep-copied at the time of selection to preserve the exact
      candidate identity, score, output path, and topic signature.

83) Improved handling of auto_eval_max_candidates = None
    - Auto-Chooser now resolves the candidate cap after candidate schemes are
      materialized.
    - When auto_eval_max_candidates is None, all available candidate schemes are
      evaluated.
    - When an integer cap is provided, it is safely bounded between 1 and the number
      of available candidates.
    - This avoids int(None) errors while preserving the existing DEFAULT_CONFIG
      behavior.

84) Separated attempted and valid-scored candidate counts
    - Auto-Chooser now distinguishes candidates that were attempted from candidates
      that completed successfully and produced valid scalar scores.
    - New runtime audit fields include:
        auto_n_candidates_attempted
        auto_n_candidates_scored
    - The previous auto_n_candidates_evaluated field is retained as a
      backward-compatible alias for the number of scored candidates.
    - These values are written as runtime provenance metadata and are not intended
      to be manually configured by users.

85) Added compact Auto-Chooser selection JSON output
    - A new auto_selected_candidate.json file is written to the parent Auto-Chooser
      output directory.
    - The JSON file summarizes the selected candidate, score, metric, candidate
      cap, number of attempted/scored candidates, runtime, topic signature, and
      fast-evaluation output directory.
    - This provides a lightweight audit record for manuscript runs, reviewer
      diagnostics, and result traceability.

86) Improved Auto-Chooser logging
    - Winner logs now report the selected candidate index, candidate name, active
      topic method, score, scoring metric, and candidate output directory.
    - Completion logs now report total Auto-Chooser runtime together with attempted
      and scored candidate counts.
    - This makes large Auto-Chooser runs easier to debug and audit.

87) Preserved existing Auto-Chooser scoring and full-run behavior
    - The selected candidate is still chosen by maximizing auto_eval_metric using
      the existing _compute_eval_score(...) logic.
    - Candidate quick-runs remain isolated in unique output subfolders.
    - After selection, the pipeline still restores the original full-strength
      settings, including xgb_params, boruta_max_iter, shap_sample_size,
      descriptive_figures_enable, pipeline_dump_sidecar_meta, pipeline_compress,
      and boruta_cache.
    - The final full run continues to use the original output_dir.

88) Added duplicate-key warnings for YAML configuration files
    - Added a custom YAML loader that detects duplicate configuration keys during
      config-file loading.
    - When a duplicate key is found, the pipeline logs a clear warning with the
      duplicated key name and its line/column location in the YAML file.
    - The loader preserves the existing PyYAML behavior:
        the later/final value overwrites the earlier value.
    - This prevents silent user mistakes such as defining candidate_schemes more
      than once in the same YAML file.

89) Added duplicate-key provenance to the runtime configuration
    - Duplicate YAML keys are now recorded in the loaded configuration as internal
      runtime metadata:
        _yaml_duplicate_keys
        _yaml_duplicate_key_count
    - These keys do not need to be added to DEFAULT_CONFIG.
    - They are intended only for auditability, config snapshots, and debugging.

90) Auto-Chooser behavior preserved
    - Duplicate candidate_schemes blocks are not automatically merged.
    - If duplicate candidate_schemes entries are present, the final/later
      candidate_schemes block is used, matching normal YAML overwrite behavior.
    - This avoids unsafe automatic merging while making the overwrite visible in
      logs and configuration provenance.

91) Scientific/modeling behavior unchanged after YAML-loading and Auto-Chooser corrections
    - The YAML-loading correction affects duplicate-key detection, duplicate-key
      warning logs, and duplicate-key provenance metadata only.
    - The Auto-Chooser correction affects candidate tracking, output-folder naming,
      leaderboard metadata, selected-candidate provenance, logging, and audit JSON
      writing only.
    - No intentional changes were made to:
        DRFP generation,
        Mordred or XTQSPRKinetic explicit-H Mordred descriptor generation,
        reaction flags,
        LDA / FCM / GMM / HDBSCAN / QDA family discovery,
        Bootstrap stability analysis,
        Bootstrap replacement safeguards,
        hard / soft / hybrid routing equations,
        Boruta feature selection,
        XGBoost specialist training,
        global fallback training,
        Auto-Chooser scoring logic,
        SHAP calculations,
        descriptive figures,
        diagnostic audit metrics,
        prediction/output equations,
        pipeline.pkl schema.
    - Duplicate YAML keys are still resolved using the final/later value, matching
      normal YAML overwrite behavior; they are not automatically merged.
    - Candidate selection is still performed by maximizing auto_eval_metric using
      the existing _compute_eval_score(...) logic.
    - Candidate quick-runs remain isolated in unique output subfolders, and the
      final selected configuration is still rerun using the restored full-strength
      settings.
    - Therefore, these corrections improve configuration safety, Auto-Chooser
      reproducibility, traceability, logging, auditability, and file navigation
      without changing the scientific modeling workflow or prediction equations.