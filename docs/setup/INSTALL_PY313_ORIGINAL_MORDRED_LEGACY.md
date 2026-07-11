# XTQSPRKinetic Python 3.13 legacy installation with original mordred==1.2.0

This profile preserves the original Mordred package for compatibility testing
or exact legacy reproducibility. It is not the preferred clean Python 3.13
profile because drfp 0.3.7 may declare a NumPy >=2.2.2 requirement, while
original Mordred 1.2.0 is commonly stabilized by forcing NumPy 1.26.4.

## Generated files

```text
requirements-bootstrap-py313.txt
requirements-all-update-py313-original-mordred.txt
requirements-runtime-py313-original-mordred.txt
```

## Recommended commands

```bash
python -m pip install --upgrade -r requirements-bootstrap-py313.txt
python -m pip install -r requirements-all-update-py313-original-mordred.txt
python -m pip install --force-reinstall -r requirements-runtime-py313-original-mordred.txt
```

The final runtime file intentionally sets:

```text
mordred==1.2.0
numpy==1.26.4
```

## Important pip-check warning

This legacy profile can be runtime-working while still failing metadata checks.
For example, `python -m pip check` may report that drfp requires a newer
NumPy than 1.26.4. That is expected for this legacy override profile and is
why the mordredcommunity profile is preferred for clean Python 3.13 installs.

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
