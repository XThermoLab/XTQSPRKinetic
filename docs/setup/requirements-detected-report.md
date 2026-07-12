# XTQSPRKinetic requirements detection report

Files scanned: 22
Core distributions: 12
Optional distributions: 10

## Core

- Boruta `0.4.3`
- drfp `0.3.7`
- joblib `1.5.3`
- matplotlib `3.10.9`
- mordred `version unavailable`
- numpy `2.4.6`
- pandas `3.0.3`
- PyYAML `6.0.3`
- rdkit `2026.3.2`
- scikit-learn `1.9.0`
- shap `0.52.0`
- xgboost `3.2.0`

## Optional / feature-specific

- gensim `4.4.0`
- hdbscan `0.8.44`
- importlib-metadata `9.0.0`
- lightgbm `4.6.0`
- lz4 `4.4.5`
- openpyxl `3.1.5`
- pyarrow `24.0.0`
- scikit-fuzzy `0.5.0`
- scipy `1.17.1`
- zstandard `0.25.0`

## Unresolved import modules


## Notes

- Optional dependencies are needed only for specific configurations/features.
- RDKit installation name may depend on the package manager; conda-forge `rdkit` is often reliable.
- Standard files reflect static import detection and the selected `--pin` mode.
- For Python 3.13, use `--profile py313-mordredcommunity` for the clean community-Mordred profile.
- Use `--profile py313-original-mordred` only for legacy original Mordred compatibility testing.
