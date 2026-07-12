# XTQSPRKinetic Python 3.13 clean installation with mordredcommunity

This is the recommended Python 3.13 profile for XTQSPRKinetic v2.7.0 when
using Mordred descriptors. It uses the community-maintained Mordred package
while preserving the same Python import path used by the code:

```python
from mordred import Calculator, descriptors
```

## Generated requirement file

```text
requirements-all-py313-mordredcommunity.txt
```

The profile intentionally uses:

```text
mordredcommunity
numpy>=2.2.2
drfp==0.3.7
```

This avoids the NumPy 1.x conflict produced by the original Mordred 1.2.0
compatibility override and should allow `python -m pip check` to pass in a
clean Python 3.13 environment, subject to the rest of your platform wheels.

## Recommended commands

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-all-py313-mordredcommunity.txt
python -m pip check
```

## Runtime validation

```bash
python - << 'PY'
import numpy, pandas, sklearn, xgboost, shap
from mordred import Calculator, descriptors
from rdkit import Chem
print('numpy', numpy.__version__)
print('mordred descriptors', len(Calculator(descriptors, ignore_3D=True).descriptors))
print('rdkit molecule OK', Chem.MolFromSmiles('CC') is not None)
PY
```

## Scientific validation note

Before final manuscript/release use, compare a small descriptor table from the
original Mordred environment and the mordredcommunity environment for a fixed
set of reaction SMILES. If descriptor names and values are consistent within
acceptable numerical tolerance, no XTQSPRKinetic source-code change is needed.
