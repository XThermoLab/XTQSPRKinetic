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
from .utils import _coerce_prediction_vector, _prediction_columns, _resolve_prediction_target_cols
from .featurization import _normalize_probability_rows

# ------------------------- PREDICTION ROUTING --------------------------------

def predict_arrhenius(
    models: Dict[Any, Any],
    X_pred: pd.DataFrame,
    topics: List[int],
    features: List[str],
    scalers: Dict[Any, Any],
    use_family_scaler: bool,
    features_by_family: Optional[Dict[int, List[str]]] = None,
    target_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Hard routing: use the specialist for each sample's hard family label.
    Supports arbitrary configured continuous targets.
    """
    target_cols = _resolve_prediction_target_cols(target_cols, models)
    n_targets = len(target_cols)
    pred_cols = _prediction_columns(target_cols)
    preds: List[List[float]] = []
    idx_iter = list(X_pred.index)
    use_local = bool(use_family_scaler or (features_by_family is not None))

    for i, idx in enumerate(idx_iter):
        t = int(topics[i]) if i < len(topics) else None
        model = models.get(t, models.get('global'))
        if model is None:
            preds.append([np.nan] * n_targets)
            continue

        cols_this = (features_by_family.get(t, features) if features_by_family else features)

        # Choose scaler: prefer family scaler when per-family features are active
        scaler = scalers.get(t) if (use_local and (t in scalers)) else scalers.get('global')
        if scaler is None:
            preds.append([np.nan] * n_targets)
            continue

        try:
            x_row = X_pred.loc[idx, cols_this].to_frame().T
        except KeyError:
            preds.append([np.nan] * n_targets)
            continue

        x_scaled = scaler.transform(x_row)
        yhat = model.predict(x_scaled)[0]
        preds.append(_coerce_prediction_vector(yhat, n_targets))

    return pd.DataFrame(preds, index=X_pred.index, columns=pred_cols)

def predict_arrhenius_weighted(
    models: Dict[Any, Any],
    X_pred: pd.DataFrame,
    topic_proba_df: pd.DataFrame,
    features: List[str],
    scalers: Dict[Any, Any],
    use_family_scaler: bool,
    min_weight: float = 1e-6,
    features_by_family: Optional[Dict[int, List[str]]] = None,
    target_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Soft routing: blend specialists according to topic probabilities per sample.
    Supports arbitrary configured continuous targets.
    """
    target_cols = _resolve_prediction_target_cols(target_cols, models)
    n_targets = len(target_cols)
    pred_cols = _prediction_columns(target_cols)
    preds: List[List[float]] = []
    idx_iter = list(X_pred.index)
    use_local = bool(use_family_scaler or (features_by_family is not None))

    # Align by index, not by position. This avoids silent routing mistakes after filtering/reordering.
    W = _normalize_probability_rows(topic_proba_df.reindex(X_pred.index).values)

    for i, idx in enumerate(idx_iter):
        if i >= W.shape[0]:
            preds.append([np.nan] * n_targets)
            continue

        w_row = np.asarray(W[i], dtype=float)
        topic_ids = [t for t in range(len(w_row)) if (w_row[t] > min_weight) and (models.get(t) is not None)]
        if not topic_ids:
            g_model = models.get('global')
            g_scaler = scalers.get('global')
            if (g_model is None) or (g_scaler is None):
                preds.append([np.nan] * n_targets)
                continue
            try:
                x_row = X_pred.loc[idx, features].to_frame().T
            except KeyError:
                preds.append([np.nan] * n_targets)
                continue
            x_scaled = g_scaler.transform(x_row)
            yhat = g_model.predict(x_scaled)[0]
            preds.append(_coerce_prediction_vector(yhat, n_targets))
            continue

        w_filtered = np.array([w_row[t] for t in topic_ids], dtype=float)
        w_sum = float(w_filtered.sum())
        if w_sum <= 0.0:
            g_model = models.get('global')
            g_scaler = scalers.get('global')
            if (g_model is None) or (g_scaler is None):
                preds.append([np.nan] * n_targets)
                continue
            try:
                x_row = X_pred.loc[idx, features].to_frame().T
            except KeyError:
                preds.append([np.nan] * n_targets)
                continue
            x_scaled = g_scaler.transform(x_row)
            yhat = g_model.predict(x_scaled)[0]
            preds.append(_coerce_prediction_vector(yhat, n_targets))
            continue

        w_norm = w_filtered / w_sum
        agg = np.zeros(n_targets, dtype=float)
        contributed = False

        for t, wt in zip(topic_ids, w_norm):
            cols_this = (features_by_family.get(t, features) if features_by_family else features)
            scaler = scalers.get(t) if (use_local and (t in scalers)) else scalers.get('global')
            model = models.get(t, None)
            if (model is None) or (scaler is None):
                continue
            try:
                x_row_t = X_pred.loc[idx, cols_this].to_frame().T
            except KeyError:
                continue
            x_scaled_t = scaler.transform(x_row_t)
            yhat_t = _coerce_prediction_vector(model.predict(x_scaled_t)[0], n_targets)
            agg += wt * np.asarray(yhat_t, dtype=float)
            contributed = True

        preds.append([float(x) for x in agg] if contributed else [np.nan] * n_targets)

    return pd.DataFrame(preds, index=X_pred.index, columns=pred_cols)

def predict_arrhenius_hybrid(
    models: Dict[Any, Any],
    X_pred: pd.DataFrame,
    agg_topic_proba_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    features: List[str],
    scalers: Dict[Any, Any],
    use_family_scaler: bool,
    metric: str = 'entropy',
    threshold: float = 0.35,
    min_weight: float = 1e-6,
    features_by_family: Optional[Dict[int, List[str]]] = None,
    target_cols: Optional[Sequence[str]] = None,
    hard_topic_series: Optional[Union[pd.Series, Sequence[int]]] = None,
) -> pd.DataFrame:
    """
    Hybrid routing: hard-route stable samples, soft-route others.
    Supports arbitrary configured continuous targets.
    """
    target_cols = _resolve_prediction_target_cols(target_cols, models)
    n_targets = len(target_cols)
    pred_cols = _prediction_columns(target_cols)
    preds: List[List[float]] = []
    idx_iter = list(X_pred.index)
    use_local = bool(use_family_scaler or (features_by_family is not None))

    # Align by index, not by position. This prevents silent routing errors after row filtering/reordering.
    W = _normalize_probability_rows(agg_topic_proba_df.reindex(X_pred.index).values)

    if hard_topic_series is not None:
        if isinstance(hard_topic_series, pd.Series):
            hard_topics = hard_topic_series.reindex(X_pred.index)
        else:
            hard_topics = pd.Series(list(hard_topic_series), index=X_pred.index[:len(hard_topic_series)]).reindex(X_pred.index)
    else:
        hard_topics = None

    if metric.lower() in ('vote', 'vote_fraction', 'vf'):
        stab_series = stability_df['modal_vote'].reindex(X_pred.index)
        stable_mask = (stab_series.values >= float(threshold))
    else:
        ent_series = stability_df['norm_entropy'].reindex(X_pred.index)
        stable_mask = (ent_series.values <= float(threshold))
    stable_mask = np.asarray(stable_mask, dtype=bool)

    for i, idx in enumerate(idx_iter):
        if i >= W.shape[0]:
            preds.append([np.nan] * n_targets)
            continue

        w_row = np.asarray(W[i], dtype=float)

        if bool(stable_mask[i]):  # HARD path for stable samples
            # Route stable samples to the active training family when supplied.
            # This avoids a mismatch when bootstrap probabilities are used only for stability
            # while specialists were trained on baseline families.
            if hard_topics is not None and pd.notna(hard_topics.iloc[i]):
                t = int(hard_topics.iloc[i])
            else:
                t = int(np.argmax(w_row))
            model = models.get(t, models.get('global'))
            if model is None:
                preds.append([np.nan] * n_targets)
                continue
            cols_this = (features_by_family.get(t, features) if features_by_family else features)
            scaler = scalers.get(t) if (use_local and (t in scalers)) else scalers.get('global')
            if scaler is None:
                preds.append([np.nan] * n_targets)
                continue
            try:
                x_row = X_pred.loc[idx, cols_this].to_frame().T
            except KeyError:
                preds.append([np.nan] * n_targets)
                continue
            x_scaled = scaler.transform(x_row)
            yhat = model.predict(x_scaled)[0]
            preds.append(_coerce_prediction_vector(yhat, n_targets))

        else:  # SOFT path for ambiguous samples
            topic_ids = [t for t in range(len(w_row)) if (w_row[t] > min_weight) and (models.get(t) is not None)]
            if not topic_ids:
                g_model = models.get('global')
                g_scaler = scalers.get('global')
                if (g_model is None) or (g_scaler is None):
                    preds.append([np.nan] * n_targets)
                    continue
                try:
                    x_row = X_pred.loc[idx, features].to_frame().T
                except KeyError:
                    preds.append([np.nan] * n_targets)
                    continue
                x_scaled = g_scaler.transform(x_row)
                yhat = g_model.predict(x_scaled)[0]
                preds.append(_coerce_prediction_vector(yhat, n_targets))
                continue

            w_filtered = np.array([w_row[t] for t in topic_ids], dtype=float)
            w_sum = float(w_filtered.sum())
            if w_sum <= 0.0:
                g_model = models.get('global')
                g_scaler = scalers.get('global')
                if (g_model is None) or (g_scaler is None):
                    preds.append([np.nan] * n_targets)
                    continue
                try:
                    x_row = X_pred.loc[idx, features].to_frame().T
                except KeyError:
                    preds.append([np.nan] * n_targets)
                    continue
                x_scaled = g_scaler.transform(x_row)
                yhat = g_model.predict(x_scaled)[0]
                preds.append(_coerce_prediction_vector(yhat, n_targets))
                continue

            w_norm = w_filtered / w_sum
            agg = np.zeros(n_targets, dtype=float)
            contributed = False
            for t, wt in zip(topic_ids, w_norm):
                cols_this = (features_by_family.get(t, features) if features_by_family else features)
                scaler = scalers.get(t) if (use_local and (t in scalers)) else scalers.get('global')
                model = models.get(t, None)
                if (model is None) or (scaler is None):
                    continue
                try:
                    x_row_t = X_pred.loc[idx, cols_this].to_frame().T
                except KeyError:
                    continue
                x_scaled_t = scaler.transform(x_row_t)
                yhat_t = _coerce_prediction_vector(model.predict(x_scaled_t)[0], n_targets)
                agg += wt * np.asarray(yhat_t, dtype=float)
                contributed = True

            preds.append([float(x) for x in agg] if contributed else [np.nan] * n_targets)

    return pd.DataFrame(preds, index=X_pred.index, columns=pred_cols)
