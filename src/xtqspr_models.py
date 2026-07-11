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

"""Shared model utilities for XTQSPRKinetic pipeline.

This module exists so that custom estimators are picklable and can be loaded
by separate evaluation programs (i.e., the class lives in an importable module,
not in __main__).
"""

from __future__ import annotations

from typing import Dict, Optional, Any
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor

class MultiOutputXGBRegressor(BaseEstimator, RegressorMixin):
    """Multi-output regressor using one XGBRegressor per target.

    Parameters
    ----------
    base_params:
        Dict of base XGBRegressor params applied to all targets.
    params_by_target:
        Optional dict mapping target index (int) to a dict of XGBRegressor params.
        These params are merged over `base_params` for that target.
    """

    def __init__(self, base_params: Optional[Dict[str, Any]] = None,
                 params_by_target: Optional[Dict[int, Dict[str, Any]]] = None):
        self.base_params = base_params
        self.params_by_target = params_by_target

    def fit(self, X, y):
        Y = y.values if hasattr(y, "values") else np.asarray(y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_outputs_ = int(Y.shape[1])

        base = dict(self.base_params) if isinstance(self.base_params, dict) else {}
        pbt = self.params_by_target if isinstance(self.params_by_target, dict) else {}

        self.estimators_ = []
        self.fitted_params_by_target_ = {}
        for j in range(self.n_outputs_):
            params = dict(base)
            override = pbt.get(j, None)
            if override is None:
                override = pbt.get(str(j), None)
            if isinstance(override, dict):
                params.update(override)

            est = XGBRegressor(**params)
            est.fit(X, Y[:, j])
            self.estimators_.append(est)
            self.fitted_params_by_target_[j] = params

        return self

    def predict(self, X):
        preds = [est.predict(X) for est in getattr(self, "estimators_", [])]
        if not preds:
            return np.zeros((len(X), 0))
        return np.column_stack(preds)
