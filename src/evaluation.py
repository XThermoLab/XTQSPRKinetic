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
from .utils import _ensure_dir, _target_cols_from_frame, _safe_target_token

# ------------------------ SHAP INTERPRETABILITY ------------------------------

def _get_first_estimator_from_multioutput(mo):
    return getattr(mo, 'estimators_', [None])[0]

def _publication_ready_rcparams(font_size: int = 12) -> Dict[str, Any]:
    """Return Matplotlib rcParams for bold, publication-ready figures."""
    fs = int(font_size or 12)
    return {
        'font.size': fs,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': fs + 2,
        'axes.labelsize': fs,
        'xtick.labelsize': max(fs - 1, 8),
        'ytick.labelsize': max(fs - 1, 8),
        'legend.fontsize': max(fs - 1, 8),
        'figure.titlesize': fs + 3,
        'savefig.dpi': 600,
    }

def _apply_bold_to_current_figure(font_size: int = 12) -> None:
    """Make all visible Matplotlib text bold, including SHAP-created axes/colorbars."""
    fs = int(font_size or 12)
    fig = plt.gcf()
    for ax in fig.axes:
        try:
            ax.title.set_fontweight('bold')
            ax.title.set_fontsize(fs + 2)
            ax.xaxis.label.set_fontweight('bold')
            ax.yaxis.label.set_fontweight('bold')
            ax.xaxis.label.set_fontsize(fs)
            ax.yaxis.label.set_fontsize(fs)
        except Exception:
            pass
        for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            try:
                lbl.set_fontweight('bold')
                lbl.set_fontsize(max(fs - 1, 8))
            except Exception:
                pass
        for txt in list(getattr(ax, 'texts', [])):
            try:
                txt.set_fontweight('bold')
                txt.set_fontsize(max(fs - 1, 8))
            except Exception:
                pass
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                try:
                    txt.set_fontweight('bold')
                    txt.set_fontsize(max(fs - 1, 8))
                except Exception:
                    pass
            try:
                leg.get_title().set_fontweight('bold')
            except Exception:
                pass

def _save_current_figure_publication(path: str, dpi: int = 600, font_size: int = 12) -> None:
    """Save current Matplotlib figure with enforced >=300 dpi and bold typography."""
    dpi = int(max(300, dpi or 600))
    _apply_bold_to_current_figure(font_size=font_size)
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()

def _coerce_numeric_feature_frame(X: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Return numeric feature matrix restricted to available columns, with infinities removed."""
    if X is None or not isinstance(X, pd.DataFrame):
        return pd.DataFrame()
    avail = [f for f in list(features or []) if f in X.columns]
    if not avail:
        avail = list(X.columns)
    Xn = X[avail].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    Xn = Xn.dropna(axis=1, how='all')
    nunique = Xn.nunique(dropna=True)
    Xn = Xn.loc[:, nunique > 1]
    return Xn

def _robust_impute_numeric(Xn: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numeric features and guarantee finite values for plotting."""
    if Xn is None or Xn.empty:
        return pd.DataFrame(index=(Xn.index if isinstance(Xn, pd.DataFrame) else None))
    Ximp = Xn.copy()
    med = Ximp.median(axis=0, numeric_only=True)
    Ximp = Ximp.fillna(med).fillna(0.0)
    return Ximp.replace([np.inf, -np.inf], 0.0)

def _family_separation_scores(Xn: pd.DataFrame, families: pd.Series) -> pd.Series:
    """
    Rank features by between-family median separation normalized by robust spread.
    This is a descriptive ranking only; it does not alter training or feature selection.
    """
    if Xn is None or Xn.empty:
        return pd.Series(dtype=float)
    fam = pd.Series(families, index=Xn.index, name='family')
    scores = {}
    for col in Xn.columns:
        s = pd.to_numeric(Xn[col], errors='coerce')
        if s.notna().sum() < 3:
            scores[col] = 0.0
            continue
        med_by_fam = s.groupby(fam).median()
        spread = float(s.quantile(0.75) - s.quantile(0.25))
        if not np.isfinite(spread) or spread <= 0:
            spread = float(s.std(ddof=0))
        if not np.isfinite(spread) or spread <= 0:
            scores[col] = 0.0
        else:
            scores[col] = float((med_by_fam.max() - med_by_fam.min()) / spread)
    return pd.Series(scores).sort_values(ascending=False)

def _select_descriptive_features(Xn: pd.DataFrame,
                                 families: pd.Series,
                                 max_features: int = 20) -> List[str]:
    """Select a compact, interpretable feature panel for descriptive figures."""
    if Xn is None or Xn.empty:
        return []
    max_features = int(max(1, max_features or 20))
    scores = _family_separation_scores(Xn, families)
    if scores.empty:
        return list(Xn.columns[:max_features])

    # Prefer human-readable Mordred/reaction-flag features over sparse DRFP bits when enough exist.
    non_drfp = [c for c in scores.index if not str(c).lower().startswith('drfp_')]
    selected = non_drfp[:max_features]
    if len(selected) < max_features:
        selected += [c for c in scores.index if c not in selected][:max_features - len(selected)]
    return selected[:max_features]

def _normalize_figure_label_index_base(index_base: Any = 1) -> int:
    """
    Normalize figure-only label index base.

    Accepted values:
        0, "0", "zero", "zero_based", "internal"   -> internal zero-based display
        1, "1", "one", "one_based", "manuscript"   -> manuscript one-based display

    This setting affects only visual labels in PNG figures/titles/legends.
    It must not be used to modify internal topic/family IDs, CSV columns,
    filenames, model dictionaries, routing probabilities, or pipeline metadata.
    """
    try:
        if isinstance(index_base, str):
            v = index_base.strip().lower().replace("-", "_")
            if v in {"0", "zero", "zero_based", "internal", "python"}:
                return 0
            if v in {"1", "one", "one_based", "manuscript", "paper"}:
                return 1
        base = int(index_base)
    except Exception:
        logging.warning(
            "Invalid figure_label_index_base=%r. Falling back to manuscript one-based labels.",
            index_base,
        )
        return 1

    if base not in (0, 1):
        logging.warning(
            "Invalid figure_label_index_base=%r. Expected 0 or 1. Falling back to 1.",
            index_base,
        )
        return 1

    return base


def _display_index_label(value: Any,
                         prefix: Optional[str] = None,
                         index_base: Any = 1) -> str:
    """
    Return a figure-only display label for zero-based topic/family IDs.

    index_base=0 preserves internal numbering in figures:
        Family 0, Topic 0

    index_base=1 shifts non-negative integer IDs by +1 for manuscript display:
        Family 1, Topic 1

    Negative IDs, such as HDBSCAN noise label -1, are preserved to avoid
    converting an outlier/noise label into a valid family/topic number.

    This helper is display-only and must not be used for CSV data, filenames,
    model keys, family assignments, topic probabilities, or pipeline metadata.
    """
    base = _normalize_figure_label_index_base(index_base)

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    def _fmt_integer(n: int) -> str:
        n = int(n)
        shown = n if n < 0 else n + base
        return f"{prefix}{shown}" if prefix is not None else str(shown)

    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return _fmt_integer(int(value))

    if (
        isinstance(value, (np.floating, float))
        and np.isfinite(float(value))
        and float(value).is_integer()
    ):
        return _fmt_integer(int(value))

    s = str(value)
    ss = s.strip()

    if ss.startswith("-") and ss[1:].isdigit():
        return _fmt_integer(int(ss))

    if ss.isdigit():
        return _fmt_integer(int(ss))

    # Preserve textual prefixes such as topic_0 / family_0 when no explicit
    # prefix is requested, but shift the trailing non-negative integer for display.
    j = len(s) - 1
    while j >= 0 and s[j].isdigit():
        j -= 1

    if j < len(s) - 1:
        try:
            base_text = s[:j + 1]
            n0 = int(s[j + 1:])

            # Do not convert labels such as "-1" or "noise-1" into valid one-based IDs.
            if base_text.endswith("-"):
                return f"{prefix}{s}" if prefix is not None else s

            if prefix is not None:
                return _fmt_integer(n0)

            return f"{base_text}{n0 + base}"
        except Exception:
            return s

    return s


def _display_index_labels(values,
                          prefix: Optional[str] = None,
                          index_base: Any = 1) -> List[str]:
    """Vectorized convenience wrapper for figure-only display labels."""
    return [
        _display_index_label(v, prefix=prefix, index_base=index_base)
        for v in values
    ]


# Backward-compatible aliases for earlier one-based-only patches.
# Existing code that still calls these functions will keep one-based behavior
# unless index_base is explicitly passed.
def _display_one_based_label(value: Any,
                             prefix: Optional[str] = None,
                             index_base: Any = 1) -> str:
    return _display_index_label(value, prefix=prefix, index_base=index_base)


def _display_one_based_labels(values,
                              prefix: Optional[str] = None,
                              index_base: Any = 1) -> List[str]:
    return _display_index_labels(values, prefix=prefix, index_base=index_base)

def _plot_landscape_categorical(plot_df: pd.DataFrame,
                                category_col: str,
                                out_path: str,
                                title: str,
                                dpi: int,
                                font_size: int,
                                figure_label_index_base: Any = 1) -> None:
    """Scatter PC1/PC2 by a categorical label with bold labels and legend."""
    figure_label_index_base = _normalize_figure_label_index_base(figure_label_index_base)

    with plt.rc_context(_publication_ready_rcparams(font_size)):
        plt.figure(figsize=(8.2, 6.4))
        ax = plt.gca()

        is_learned_family = str(category_col).lower() in {"family", "learned_family", "learned family"}

        for lab in sorted(plot_df[category_col].dropna().unique(), key=lambda x: str(x)):
            m = plot_df[category_col].astype(str) == str(lab)

            display_lab = (
                _display_index_label(
                    lab,
                    prefix="Family ",
                    index_base=figure_label_index_base,
                )
                if is_learned_family
                else str(lab)
            )

            ax.scatter(
                plot_df.loc[m, "PC1"],
                plot_df.loc[m, "PC2"],
                s=28,
                alpha=0.82,
                edgecolors="none",
                label=display_lab,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title, fontweight="bold")

        legend_title = "Learned family" if is_learned_family else category_col
        ax.legend(
            title=legend_title,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=False,
        )

        _save_current_figure_publication(out_path, dpi=dpi, font_size=font_size)

def _plot_landscape_continuous(plot_df: pd.DataFrame,
                               value_col: str,
                               out_path: str,
                               title: str,
                               colorbar_label: str,
                               dpi: int,
                               font_size: int) -> None:
    """Scatter PC1/PC2 colored by a continuous diagnostic quantity."""
    if value_col not in plot_df.columns or plot_df[value_col].notna().sum() == 0:
        return
    with plt.rc_context(_publication_ready_rcparams(font_size)):
        plt.figure(figsize=(8.2, 6.4))
        ax = plt.gca()
        sc = ax.scatter(plot_df['PC1'], plot_df['PC2'], c=plot_df[value_col], s=28, alpha=0.86,
                        edgecolors='none')
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(colorbar_label, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(title, fontweight='bold')
        _save_current_figure_publication(out_path, dpi=dpi, font_size=font_size)

def write_descriptive_figures_summary(df: pd.DataFrame,
                                      X: pd.DataFrame,
                                      Y: pd.DataFrame,
                                      preds: pd.DataFrame,
                                      families: List[Any],
                                      features: List[str],
                                      output_dir: str,
                                      topic_proba_df: Optional[pd.DataFrame] = None,
                                      stability_df: Optional[pd.DataFrame] = None,
                                      rmg_family_col: Optional[str] = None) -> None:
    """
    Write a compact summary CSV for the descriptive-figure module.

    This mirrors the SHAP behavior requested by the user: when the full
    figure-producing switch is disabled, the pipeline still records a compact
    summary artifact, but it does not generate PNG figures or companion plot data.
    """
    _ensure_dir(output_dir)
    records: List[Dict[str, Any]] = []

    def _add(section: str, metric: str, value: Any) -> None:
        try:
            if isinstance(value, (np.integer, np.floating)):
                value = float(value)
        except Exception:
            pass
        records.append({'section': section, 'metric': metric, 'value': value})

    try:
        n_rows = int(len(df)) if isinstance(df, pd.DataFrame) else int(len(X))
    except Exception:
        n_rows = np.nan
    _add('dataset', 'n_reactions', n_rows)
    _add('features', 'n_model_features', int(len(features) if features is not None else 0))

    try:
        fam = pd.Series(families, name='family')
        vc = fam.value_counts(dropna=False)
        _add('families', 'n_learned_families', int(fam.dropna().nunique()))
        for k, v in vc.items():
            _add('families', f'family_{k}_count', int(v))
    except Exception:
        _add('families', 'n_learned_families', np.nan)

    try:
        if isinstance(Y, pd.DataFrame):
            for target in _target_cols_from_frame(Y):
                if target in Y.columns:
                    y = pd.to_numeric(Y[target], errors='coerce')
                    _add(f'target_{target}', 'count', int(y.notna().sum()))
                    _add(f'target_{target}', 'mean', y.mean())
                    _add(f'target_{target}', 'std', y.std())
                    _add(f'target_{target}', 'min', y.min())
                    _add(f'target_{target}', 'median', y.median())
                    _add(f'target_{target}', 'max', y.max())
    except Exception as e:
        _add('targets', 'summary_status', f'failed: {e}')

    try:
        if isinstance(preds, pd.DataFrame) and isinstance(Y, pd.DataFrame):
            for target in _target_cols_from_frame(Y):
                pcol = f'{target}_pred'
                if target in Y.columns and pcol in preds.columns:
                    err = (pd.to_numeric(Y[target], errors='coerce') - pd.to_numeric(preds[pcol], errors='coerce')).abs()
                    _add(f'residual_{target}', 'mean_absolute_error', err.mean())
                    _add(f'residual_{target}', 'median_absolute_error', err.median())
                    _add(f'residual_{target}', 'max_absolute_error', err.max())
    except Exception as e:
        _add('residuals', 'summary_status', f'failed: {e}')

    try:
        has_routing = isinstance(topic_proba_df, pd.DataFrame) and not topic_proba_df.empty
        _add('routing', 'routing_probability_table_available', bool(has_routing))
        if has_routing:
            _add('routing', 'n_routing_components', int(topic_proba_df.shape[1]))
            maxp = topic_proba_df.apply(pd.to_numeric, errors='coerce').max(axis=1)
            _add('routing', 'mean_max_routing_probability', maxp.mean())
            _add('routing', 'median_max_routing_probability', maxp.median())
    except Exception as e:
        _add('routing', 'summary_status', f'failed: {e}')

    try:
        has_stability = isinstance(stability_df, pd.DataFrame) and not stability_df.empty
        _add('bootstrap_stability', 'stability_table_available', bool(has_stability))
        if has_stability:
            for col in ['modal_vote', 'vote_fraction', 'entropy', 'norm_entropy']:
                if col in stability_df.columns:
                    s = pd.to_numeric(stability_df[col], errors='coerce')
                    _add('bootstrap_stability', f'{col}_mean', s.mean())
                    _add('bootstrap_stability', f'{col}_median', s.median())
    except Exception as e:
        _add('bootstrap_stability', 'summary_status', f'failed: {e}')

    try:
        if isinstance(df, pd.DataFrame) and rmg_family_col and rmg_family_col in df.columns:
            _add('mechanistic_labels', f'{rmg_family_col}_available', True)
            _add('mechanistic_labels', f'n_unique_{rmg_family_col}', int(df[rmg_family_col].nunique(dropna=True)))
        else:
            _add('mechanistic_labels', 'rmg_family_column_available', False)
    except Exception:
        pass

    pd.DataFrame(records).to_csv(os.path.join(output_dir, 'descriptive_figures_summary.csv'), index=False)

def _write_shap_targets_summary(Y: pd.DataFrame, output_dir: str) -> None:
    """Write the compact SHAP target summary even when SHAP figure generation is disabled."""
    _ensure_dir(output_dir)
    try:
        if isinstance(Y, pd.DataFrame) and len(Y):
            Y.describe().to_csv(os.path.join(output_dir, "shap_targets_summary.csv"))
    except Exception:
        pass

def _heatmap_annotation_color(im, value: float, auto_contrast: bool = True) -> str:
    """Choose black/white annotation text based on heatmap cell brightness."""
    if not auto_contrast:
        return 'black'
    try:
        rgba = im.cmap(im.norm(float(value)))
        r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        return 'black' if brightness >= 0.55 else 'white'
    except Exception:
        return 'black'
    
def compute_and_save_descriptive_figures(df: pd.DataFrame,
                                         X: pd.DataFrame,
                                         Y: pd.DataFrame,
                                         preds: pd.DataFrame,
                                         families: List[Any],
                                         features: List[str],
                                         output_dir: str,
                                         topic_proba_df: Optional[pd.DataFrame] = None,
                                         stability_df: Optional[pd.DataFrame] = None,
                                         rmg_family_col: Optional[str] = None,
                                         max_features: int = 20,
                                         boxplot_max_features: int = 6,
                                         pca_sample_size: Optional[int] = None,
                                         random_state: int = 42,
                                         dpi: int = 600,
                                         font_size: int = 12,
                                         heatmap_cmap: str = 'YlGnBu',
                                         descriptor_heatmap_cmap: str = 'RdBu_r',
                                         routing_heatmap_cmap: str = 'YlGnBu',
                                         routing_family_heatmap_cmap: str = 'YlGnBu',
                                         heatmap_annotation_auto_contrast: bool = True,
                                         figure_label_index_base: Any = 1) -> None:
    """
    Generate manuscript-ready descriptive diagnostics that complement SHAP.

    Outputs are written under output_dir and include a CSV companion for every PNG:
      - PCA reaction-family landscape colored by learned family, targets, and residuals.
      - Family descriptor-signature heatmap using robust z-scored selected features.
      - Family descriptor boxplots for the most family-discriminating features.
      - Routing-probability heatmap and residual-vs-stability plot when available.

    These plots are descriptive/auditing artifacts only; they do not change model fitting,
    routing, Boruta selection, SHAP analysis, or pipeline.pkl.
    """
    _ensure_dir(output_dir)
    write_descriptive_figures_summary(
        df=df, X=X, Y=Y, preds=preds, families=families, features=features,
        output_dir=output_dir, topic_proba_df=topic_proba_df,
        stability_df=stability_df, rmg_family_col=rmg_family_col
    )
    dpi = int(max(300, dpi or 600))
    font_size = int(font_size or 12)

    figure_label_index_base = _normalize_figure_label_index_base(figure_label_index_base)

    try:
        fam = pd.Series(families, index=X.index, name='family')
    except Exception:
        fam = pd.Series(families, name='family')

    Xn = _coerce_numeric_feature_frame(X, features)
    if Xn.empty or len(Xn) < 3:
        logging.info("Descriptive figures skipped: not enough numeric feature data.")
        return
    fam = pd.Series(fam).reindex(Xn.index)
    if fam.notna().nunique() < 1:
        logging.info("Descriptive figures skipped: family labels unavailable.")
        return

    # Optional sampling only for visualization/PCA cost; source data remain unchanged.
    idx_plot = Xn.index
    if pca_sample_size is not None:
        try:
            pca_sample_size = int(pca_sample_size)
            if pca_sample_size > 0 and len(idx_plot) > pca_sample_size:
                idx_plot = pd.Index(pd.Series(idx_plot).sample(n=pca_sample_size, random_state=int(random_state)).values)
        except Exception:
            idx_plot = Xn.index

    Xp = _robust_impute_numeric(Xn.loc[idx_plot])
    fam_plot = fam.reindex(idx_plot)

    # PCA landscape in the final model feature space.
    try:
        if Xp.shape[1] < 2 or Xp.shape[0] < 2:
            raise ValueError('PCA landscape requires at least 2 samples and 2 numeric features.')
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(Xp.values)
        pca = PCA(n_components=2, random_state=int(random_state))
        coords = pca.fit_transform(X_scaled)
        evr = getattr(pca, 'explained_variance_ratio_', np.array([np.nan, np.nan]))

        plot_df = pd.DataFrame({
            'PC1': coords[:, 0],
            'PC2': coords[:, 1],
            'family': fam_plot.values,
        }, index=idx_plot)
        plot_df['PC1_explained_variance_ratio'] = float(evr[0]) if len(evr) > 0 else np.nan
        plot_df['PC2_explained_variance_ratio'] = float(evr[1]) if len(evr) > 1 else np.nan

        if isinstance(df, pd.DataFrame):
            for col in ['reaction_smiles']:
                if col in df.columns:
                    plot_df[col] = df.reindex(idx_plot)[col].values
            if rmg_family_col and rmg_family_col in df.columns:
                plot_df[rmg_family_col] = df.reindex(idx_plot)[rmg_family_col].values

        if isinstance(Y, pd.DataFrame):
            for target in _target_cols_from_frame(Y):
                if target in Y.columns:
                    plot_df[target] = Y.reindex(idx_plot)[target].values
        if isinstance(preds, pd.DataFrame):
            for target in _target_cols_from_frame(Y):
                pcol = f'{target}_pred'
                if pcol in preds.columns:
                    plot_df[pcol] = preds.reindex(idx_plot)[pcol].values
                    if target in plot_df.columns:
                        plot_df[f'{target}_abs_err'] = np.abs(plot_df[target] - plot_df[pcol])

        if isinstance(topic_proba_df, pd.DataFrame) and not topic_proba_df.empty:
            W = topic_proba_df.reindex(idx_plot)
            plot_df['max_routing_probability'] = W.max(axis=1).values
            with np.errstate(divide='ignore', invalid='ignore'):
                Wv = W.fillna(0.0).clip(lower=0.0).values.astype(float)
                denom = max(np.log(max(Wv.shape[1], 2)), 1e-12)
                ent = -np.sum(np.where(Wv > 0, Wv * np.log(Wv), 0.0), axis=1) / denom
            plot_df['routing_norm_entropy'] = ent

        if isinstance(stability_df, pd.DataFrame) and not stability_df.empty:
            for col in ['modal_vote', 'vote_fraction', 'entropy', 'norm_entropy']:
                if col in stability_df.columns:
                    plot_df[f'bootstrap_{col}'] = stability_df.reindex(idx_plot)[col].values

        plot_df.to_csv(os.path.join(output_dir, 'family_landscape_pca.csv'), index=False)

        _plot_landscape_categorical(
            plot_df.dropna(subset=['PC1', 'PC2']), 'family',
            os.path.join(output_dir, 'family_landscape_by_learned_family.png'),
            f'Reaction-family landscape (PC1={100*plot_df["PC1_explained_variance_ratio"].iloc[0]:.1f}%, '
            f'PC2={100*plot_df["PC2_explained_variance_ratio"].iloc[0]:.1f}%)',
            dpi=dpi, font_size=font_size, figure_label_index_base=figure_label_index_base
        )

        for target in _target_cols_from_frame(Y):
            token = _safe_target_token(target)
            if target in plot_df.columns:
                _plot_landscape_continuous(
                    plot_df, target,
                    os.path.join(output_dir, f'family_landscape_by_{token}.png'),
                    f'Reaction-family landscape colored by {target}',
                    target, dpi=dpi, font_size=font_size
                )
            e_col = f'{target}_abs_err'
            if e_col in plot_df.columns:
                _plot_landscape_continuous(
                    plot_df, e_col,
                    os.path.join(output_dir, f'family_landscape_by_{token}_absolute_error.png'),
                    f'Reaction-family landscape colored by |{target} error|',
                    f'|{target} error|', dpi=dpi, font_size=font_size
                )
    except Exception as e:
        logging.warning("Descriptive PCA landscape failed: %s", e)

    # Family descriptor-signature heatmap and boxplot data.
    try:
        selected = _select_descriptive_features(Xn, fam, max_features=max_features)
        if selected:
            Xsel = _robust_impute_numeric(Xn[selected])
            med = Xsel.median(axis=0)
            iqr = Xsel.quantile(0.75, axis=0) - Xsel.quantile(0.25, axis=0)
            std = Xsel.std(axis=0, ddof=0)
            scale = iqr.where(iqr > 0, std).replace(0, np.nan).fillna(1.0)
            Z = (Xsel - med) / scale
            heat = Z.groupby(fam).median().sort_index()
            heat.index.name = 'family'
            heat.to_csv(os.path.join(output_dir, 'family_descriptor_heatmap_median_robust_z.csv'))

            with plt.rc_context(_publication_ready_rcparams(font_size)):
                fig_w = max(8.5, 0.42 * len(heat.columns) + 3.0)
                fig_h = max(4.8, 0.45 * len(heat.index) + 2.2)
                plt.figure(figsize=(fig_w, fig_h))
                ax = plt.gca()
                im = ax.imshow(heat.values, aspect='auto', cmap=str(descriptor_heatmap_cmap or heatmap_cmap))
                ax.set_xticks(np.arange(len(heat.columns)))
                ax.set_xticklabels(heat.columns, rotation=60, ha='right')
                ax.set_yticks(np.arange(len(heat.index)))
                ax.set_yticklabels(
                    _display_index_labels(
                        heat.index,
                        prefix="Family ",
                        index_base=figure_label_index_base,
                    )
                )
                ax.set_xlabel('Selected descriptor / reaction feature')
                ax.set_ylabel('Learned family')
                ax.set_title('Family descriptor signatures (median robust z-score)', fontweight='bold')
                cb = plt.colorbar(im, ax=ax)
                cb.set_label('Median robust z-score', fontweight='bold')
                _save_current_figure_publication(os.path.join(output_dir, 'family_descriptor_heatmap.png'), dpi=dpi, font_size=font_size)

            # Boxplot data + compact multi-panel plot for the most discriminating selected features.
            n_box = int(max(1, min(boxplot_max_features or 6, len(selected))))
            box_features = selected[:n_box]
            long_rows = []
            for feat in box_features:
                tmp = pd.DataFrame({'family': fam.values, 'feature': feat, 'value': Xsel[feat].values}, index=Xsel.index)
                long_rows.append(tmp)
            box_long = pd.concat(long_rows, axis=0, ignore_index=True) if long_rows else pd.DataFrame()
            if not box_long.empty:
                box_long.to_csv(os.path.join(output_dir, 'descriptor_boxplots_by_family_data.csv'), index=False)
                fam_levels = sorted(pd.Series(fam.dropna().unique()).tolist(), key=lambda x: str(x))
                with plt.rc_context(_publication_ready_rcparams(font_size)):
                    fig, axes = plt.subplots(n_box, 1, figsize=(8.5, max(3.0, 2.05 * n_box)), sharex=True)
                    if n_box == 1:
                        axes = [axes]
                    for ax, feat in zip(axes, box_features):
                        vals = [Xsel.loc[fam == fl, feat].dropna().values for fl in fam_levels]
                        ax.boxplot(
                            vals,
                            labels=_display_index_labels(
                                fam_levels,
                                index_base=figure_label_index_base,
                            ),
                            showfliers=True,
                        )
                        ax.set_ylabel(feat, fontweight='bold')
                        ax.grid(axis='y', alpha=0.25)
                    axes[-1].set_xlabel(f"Learned family ({figure_label_index_base}-based display labels)")
                    fig.suptitle('Selected descriptor distributions by learned family', fontweight='bold')
                    _save_current_figure_publication(os.path.join(output_dir, 'descriptor_boxplots_by_family.png'), dpi=dpi, font_size=font_size)
    except Exception as e:
        logging.warning("Descriptive family descriptor plots failed: %s", e)

    # Routing probability heatmap: reactions ordered by learned family.
    try:
        if isinstance(topic_proba_df, pd.DataFrame) and not topic_proba_df.empty:
            W = topic_proba_df.reindex(Xn.index).copy()
            W['family'] = fam.values
            W_sorted = W.sort_values('family')
            # Keep heatmap readable for very large networks.
            if len(W_sorted) > 800:
                W_sorted = W_sorted.sample(n=800, random_state=int(random_state)).sort_values('family')
            W_sorted.to_csv(os.path.join(output_dir, 'routing_probability_heatmap_data.csv'), index=False)
            Wvals = W_sorted.drop(columns=['family']).apply(pd.to_numeric, errors='coerce').fillna(0.0)
            with plt.rc_context(_publication_ready_rcparams(font_size)):
                plt.figure(figsize=(8.2, max(4.5, min(12.0, len(Wvals) / 55.0 + 2.0))))
                ax = plt.gca()
                im = ax.imshow(
                    Wvals.values,
                    aspect='auto',
                    cmap=str(routing_heatmap_cmap or heatmap_cmap),
                    vmin=0.0,
                    vmax=1.0
                )
                ax.set_xlabel('Routing component / topic')
                ax.set_ylabel('Reactions ordered by learned family')
                ax.set_title('Soft-routing probability heatmap', fontweight='bold')
                ax.set_xticks(np.arange(Wvals.shape[1]))
                ax.set_xticklabels(
                    _display_index_labels(
                        Wvals.columns,
                        prefix="Topic ",
                        index_base=figure_label_index_base,
                    ),
                    rotation=45,
                    ha="right",
                )
                cb = plt.colorbar(im, ax=ax)
                cb.set_label('Routing probability', fontweight='bold')
                _save_current_figure_publication(os.path.join(output_dir, 'routing_probability_heatmap.png'), dpi=dpi, font_size=font_size)
    except Exception as e:
        logging.warning("Descriptive routing probability heatmap failed: %s", e)

    # Family-level median routing-probability heatmap.
    # This compact summary complements the reaction-level routing heatmap.
    # Rows = learned families; columns = routing components/topics;
    # values = median routing probability within each learned family.
    try:
        if isinstance(topic_proba_df, pd.DataFrame) and not topic_proba_df.empty:
            W = topic_proba_df.reindex(Xn.index).copy()
            Wvals = W.apply(pd.to_numeric, errors='coerce').fillna(0.0)

            routing_summary = Wvals.copy()
            routing_summary['family'] = fam.values
            routing_summary = routing_summary.dropna(subset=['family'])

            topic_cols = [c for c in routing_summary.columns if c != 'family']

            if topic_cols and not routing_summary.empty:
                family_median = (
                    routing_summary
                    .groupby('family', sort=True)[topic_cols]
                    .median()
                )

                family_median.index.name = 'learned_family'
                family_median.to_csv(os.path.join(output_dir, 'routing_probability_family_median_heatmap_data.csv'))

                with plt.rc_context(_publication_ready_rcparams(font_size)):
                    fig_w = max(7.0, 0.75 * len(topic_cols) + 3.0)
                    fig_h = max(4.8, 0.55 * len(family_median.index) + 2.5)
                    plt.figure(figsize=(fig_w, fig_h))
                    ax = plt.gca()

                    im = ax.imshow(
                        family_median.values,
                        aspect='auto',
                        cmap=str(routing_family_heatmap_cmap or routing_heatmap_cmap or heatmap_cmap),
                        vmin=0.0,
                        vmax=1.0
                    )

                    ax.set_xticks(np.arange(len(family_median.columns)))
                    ax.set_xticklabels(
                        _display_index_labels(
                            family_median.columns,
                            prefix="Topic ",
                            index_base=figure_label_index_base,
                        ),
                        rotation=45,
                        ha="right",
                    )

                    ax.set_yticks(np.arange(len(family_median.index)))
                    ax.set_yticklabels(
                        _display_index_labels(
                            family_median.index,
                            prefix="Family ",
                            index_base=figure_label_index_base,
                        )
                    )

                    ax.set_xlabel('Routing component / topic')
                    ax.set_ylabel('Learned family')
                    ax.set_title(
                        'Family-level median soft-routing probability',
                        fontweight='bold'
                    )

                    cb = plt.colorbar(im, ax=ax)
                    cb.set_label('Median routing probability', fontweight='bold')

                    # Add numeric labels because the matrix is compact.
                    for i in range(family_median.shape[0]):
                        for j in range(family_median.shape[1]):
                            val = family_median.values[i, j]
                            if np.isfinite(val):
                                ax.text(
                                    j,
                                    i,
                                    f'{val:.2f}',
                                    ha='center',
                                    va='center',
                                    fontsize=max(font_size - 3, 7),
                                    fontweight='bold',
                                    color=_heatmap_annotation_color(
                                        im,
                                        val,
                                        auto_contrast=bool(heatmap_annotation_auto_contrast)
                                    )
                                )

                    _save_current_figure_publication(os.path.join(output_dir, 'routing_probability_family_median_heatmap.png'), dpi=dpi, font_size=font_size)

    except Exception as e:
        logging.warning("Descriptive family-level routing probability heatmap failed: %s", e)

    # Residual vs routing/bootstrap uncertainty.
    try:
        rv = pd.DataFrame(index=Xn.index)
        rv['family'] = fam.values
        if isinstance(preds, pd.DataFrame) and isinstance(Y, pd.DataFrame):
            err_cols = []
            for target in _target_cols_from_frame(Y):
                pcol = f'{target}_pred'
                if target in Y.columns and pcol in preds.columns:
                    ecol = f'{target}_abs_err'
                    rv[ecol] = np.abs(Y.reindex(Xn.index)[target] - preds.reindex(Xn.index)[pcol])
                    err_cols.append(ecol)
            if err_cols:
                rv['mean_absolute_error_across_targets'] = rv[err_cols].mean(axis=1)

        uncertainty_col = None
        if isinstance(stability_df, pd.DataFrame) and not stability_df.empty:
            for col in ['norm_entropy', 'entropy', 'modal_vote', 'vote_fraction']:
                if col in stability_df.columns:
                    rv[f'bootstrap_{col}'] = stability_df.reindex(Xn.index)[col].values
            if 'bootstrap_norm_entropy' in rv.columns:
                uncertainty_col = 'bootstrap_norm_entropy'
            elif 'bootstrap_entropy' in rv.columns:
                uncertainty_col = 'bootstrap_entropy'
            elif 'bootstrap_modal_vote' in rv.columns:
                rv['bootstrap_uncertainty_from_vote'] = 1.0 - pd.to_numeric(rv['bootstrap_modal_vote'], errors='coerce')
                uncertainty_col = 'bootstrap_uncertainty_from_vote'
            elif 'bootstrap_vote_fraction' in rv.columns:
                rv['bootstrap_uncertainty_from_vote'] = 1.0 - pd.to_numeric(rv['bootstrap_vote_fraction'], errors='coerce')
                uncertainty_col = 'bootstrap_uncertainty_from_vote'
        if uncertainty_col is None and isinstance(topic_proba_df, pd.DataFrame) and not topic_proba_df.empty:
            W = topic_proba_df.reindex(Xn.index).fillna(0.0).clip(lower=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                Wv = W.values.astype(float)
                denom = max(np.log(max(Wv.shape[1], 2)), 1e-12)
                rv['routing_norm_entropy'] = -np.sum(np.where(Wv > 0, Wv * np.log(Wv), 0.0), axis=1) / denom
            uncertainty_col = 'routing_norm_entropy'

        rv.to_csv(os.path.join(output_dir, 'residual_vs_routing_uncertainty_data.csv'), index=False)
        if uncertainty_col and 'mean_absolute_error_across_targets' in rv.columns:
            plot_rv = rv[[uncertainty_col, 'mean_absolute_error_across_targets', 'family']].dropna()
            if len(plot_rv) >= 3:
                with plt.rc_context(_publication_ready_rcparams(font_size)):
                    plt.figure(figsize=(8.2, 6.2))
                    ax = plt.gca()
                    for lab in sorted(plot_rv['family'].dropna().unique(), key=lambda x: str(x)):
                        m = plot_rv['family'].astype(str) == str(lab)
                        ax.scatter(
                            plot_rv.loc[m, uncertainty_col],
                            plot_rv.loc[m, "mean_absolute_error_across_targets"],
                            s=30,
                            alpha=0.82,
                            edgecolors="none",
                            label=_display_index_label(
                                lab,
                                prefix="Family ",
                                index_base=figure_label_index_base,
                            ),
                        )
                    ax.set_xlabel(uncertainty_col)
                    ax.set_ylabel('Mean absolute error across targets')
                    ax.set_title('Prediction residuals vs routing/bootstrap uncertainty', fontweight='bold')
                    ax.legend(title="Learned family", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
                    _save_current_figure_publication(os.path.join(output_dir, 'residual_vs_routing_uncertainty.png'), dpi=dpi, font_size=font_size)
    except Exception as e:
        logging.warning("Descriptive residual-vs-uncertainty plot failed: %s", e)

def compute_and_save_shap(models: Dict[Any, Any],
                          scalers: Dict[Any, Any],
                          X: pd.DataFrame,
                          Y: pd.DataFrame,
                          topics: List[int],
                          features: List[str],
                          output_dir: str,
                          use_family_scaler: bool = True,
                          shap_sample_size: int = 250,
                          figure_dpi: int = 600,
                          font_size: int = 12,
                          max_display: int = 20,
                          figure_label_index_base: Any = 1):
    """
    Compute SHAP explanations for the global model and per-family specialists.
    Saves PNGs at 600 dpi and CSVs of feature importance summaries.

    Notes:
        SHAP expects raw (unscaled) features for plotting labels; we apply scalers
        only to the data fed into model prediction.
    """
    _ensure_dir(output_dir)
    dpi = int(max(300, figure_dpi or 600))
    font_size = int(font_size or 12)
    max_display = int(max(1, max_display or 20))
    figure_label_index_base = _normalize_figure_label_index_base(figure_label_index_base)

    # Global
    try:
        if 'global' in models and shap_sample_size != 0:
            base_model = _get_first_estimator_from_multioutput(models['global'])
            if base_model is not None:
                Xg = X[features]
                if shap_sample_size and len(Xg) > shap_sample_size:
                    Xg = Xg.sample(n=shap_sample_size, random_state=42)
                Xg_scaled = scalers['global'].transform(Xg)
                explainer = shap.TreeExplainer(base_model)
                sv = explainer.shap_values(Xg_scaled)

                # Generate plot (bar plot-mean absolute SHAP)
                with plt.rc_context(_publication_ready_rcparams(font_size)):
                    shap.summary_plot(sv, Xg, show=False, plot_type="bar", max_display=max_display)
                    plt.title("Global model - Mean |SHAP|", fontweight='bold')
                    _save_current_figure_publication(os.path.join(output_dir, "mean_abs_shap_global.png"), dpi=dpi, font_size=font_size)

                # Generate CSV (mean absolute SHAP)
                mean_abs_g = np.mean(np.abs(sv), axis=0); rank = np.argsort(-mean_abs_g)
                pd.DataFrame({"feature": Xg.columns[rank], "mean_abs_shap": mean_abs_g[rank]}) \
                        .to_csv(os.path.join(output_dir, f"mean_abs_shap_global.csv"), index=False)
    except Exception as e:
        logging.warning(f"Global SHAP failed/skipped: {e}")

    # Per-topic
    topics_series = pd.Series(topics, index=X.index, name='topic')
    summary_records = []
    for topic in sorted(topics_series.unique()):
        if topic not in models or shap_sample_size == 0: continue
        idx = topics_series == topic
        X_fam = X.loc[idx, features]
        if len(X_fam) == 0: continue
        if shap_sample_size and len(X_fam) > shap_sample_size:
            X_fam = X_fam.sample(n=shap_sample_size, random_state=42)
        scaler = scalers[topic] if use_family_scaler and topic in scalers else scalers['global']
        X_fam_scaled = scaler.transform(X_fam)

        # Legacy per-target artifacts
        try:
            for i, out_name in enumerate(_target_cols_from_frame(Y)):
                out_token = _safe_target_token(out_name)
                try: submodel = models[topic].estimators_[i]
                except Exception: submodel = None
                if submodel is None: continue
                expl = shap.TreeExplainer(submodel)
                sv = expl.shap_values(X_fam_scaled)
                
                # Generate SHAP CSV data
                pd.DataFrame(sv, columns=X_fam.columns).to_csv(
                os.path.join(output_dir, f"shap_values_family_{topic}_{out_token}.csv"),
                index=False)

                # Generate plot (point plot)
                try:
                    with plt.rc_context(_publication_ready_rcparams(font_size)):
                        shap.summary_plot(sv, X_fam, show=False, max_display=max_display)
                        plt.title(
                            f"SHAP - Family {_display_index_label(topic, index_base=figure_label_index_base)}_{out_name}",
                            fontweight="bold",
                        )
                        _save_current_figure_publication(os.path.join(output_dir, f"shap_values_family_{topic}_{out_token}.png"), dpi=dpi, font_size=font_size)
                except Exception: pass

                # Generate plot (bar plot-mean absolute SHAP)
                try:
                    with plt.rc_context(_publication_ready_rcparams(font_size)):
                        plt.figure()
                        shap.summary_plot(sv, X_fam, show=False, plot_type="bar", max_display=max_display)
                        plt.title(
                            f"Mean |SHAP| - Family {_display_index_label(topic, index_base=figure_label_index_base)}_{out_name}",
                            fontweight="bold",
                        )
                        _save_current_figure_publication(os.path.join(output_dir, f"mean_abs_shap_values_family_{topic}_{out_token}.png"), dpi=dpi, font_size=font_size)
                except Exception: pass

                # Generate CSV (mean absolute SHAP)
                try:
                    mean_abs = np.mean(np.abs(sv), axis=0); rank = np.argsort(-mean_abs)
                    pd.DataFrame({"feature": X_fam.columns[rank], "mean_abs_shap": mean_abs[rank]}) \
                        .to_csv(os.path.join(output_dir, f"mean_abs_shap_values_family_{topic}_{out_token}.csv"), index=False)
                    if len(rank) > 0:
                        summary_records.append({"family": topic, "output": out_name,
                                                "top_feature": X_fam.columns[rank[0]],
                                                "max_shap": float(mean_abs[rank[0]])})
                except Exception: pass

                try:
                    f_html = shap.force_plot(expl.expected_value, sv[0, :], X_fam.iloc[0, :], matplotlib=False)
                    shap.save_html(os.path.join(output_dir, f"shap_values_family{topic}_{out_token}_force.html"), f_html)
                except Exception: pass

        except Exception as e:
            logging.warning(f"SHAP failed for topic {topic} (per-target): {e}")

    if summary_records:
        pd.DataFrame(summary_records).to_csv(os.path.join(output_dir, "shap_summary.csv"), index=False)

    # Persist a compact summary of targets used during SHAP.
    _write_shap_targets_summary(Y, output_dir)

