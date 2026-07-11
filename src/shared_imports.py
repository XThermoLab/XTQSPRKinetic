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

import os
import sys
import json
import yaml
import math
import time
import copy
import joblib
import shap
import logging
import warnings
import errno  
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Union, Any, Optional, Sequence
import random
import hashlib
import platform
import shutil
from joblib import Parallel, delayed, parallel_config
from datetime import datetime, timezone
from contextlib import contextmanager
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# PCA for DRFP dimensionality reduction before FCM 
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
    v_measure_score,
)
from sklearn.model_selection import GridSearchCV, KFold
# GMM (soft probabilistic clustering)
from sklearn.mixture import GaussianMixture
# Optional HDBSCAN for density-based clusters + soft strengths
try:
    import hdbscan
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False

from xgboost import XGBRegressor
from boruta import BorutaPy

from drfp import DrfpEncoder
from mordred import Calculator, descriptors
# Optional RDKit for BRICS tokenization
try:
    from rdkit import Chem
    from rdkit.Chem import BRICS
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False

import matplotlib.pyplot as plt

# Optional LightGBM for faster Boruta
try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

# Optional SciPy for Hungarian matching (topic alignment)
try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Optional Gensim for topic coherence
try:
    from gensim.corpora import Dictionary  # type: ignore
    from gensim.models.coherencemodel import CoherenceModel  # type: ignore
    HAVE_GENSIM = True
except Exception:
    HAVE_GENSIM = False
    
# Optional scikit-fuzzy for FCM
try:
    import skfuzzy as fuzz
    HAVE_FCM = True
except Exception:
    HAVE_FCM = False

try:
    import lz4.frame as lz4  # ensure the 'frame' API exists (what joblib expects)
    HAVE_LZ4 = True
except Exception:
    HAVE_LZ4 = False

# LZMA (stdlib; available on all normal CPython builds)
try:
    import lzma as lzma
    HAVE_LZMA = True
except Exception:
    HAVE_LZMA = False
