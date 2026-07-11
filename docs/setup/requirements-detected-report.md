# XTQSPRKinetic requirements detection report

Files scanned: 23
Core distributions: 12
Optional distributions: 10

## Core

- Boruta `0.4.3`
- drfp `0.3.7`
- joblib `1.5.2`
- matplotlib `3.10.5`
- mordred `1.2.0`
- numpy `1.26.4`
- pandas `2.3.2`
- PyYAML `6.0.2`
- rdkit `2025.3.5`
- scikit-learn `1.7.1`
- shap `0.48.0`
- xgboost `3.0.4`

## Optional / feature-specific

- gensim `4.4.0`
- hdbscan `0.8.40`
- importlib-metadata `8.7.0`
- lightgbm `4.6.0`
- lz4 `4.4.4`
- openpyxl `3.1.5`
- pyarrow `21.0.0`
- scikit-fuzzy `0.5.0`
- scipy `1.16.1`
- zstandard `version unavailable`

## Unresolved import modules


## Notes

- Optional dependencies are needed only for specific configurations/features.
- RDKit installation name may depend on the package manager; conda-forge `rdkit` is often reliable.
- Standard files reflect static import detection and the selected `--pin` mode.
- For Python 3.13, use `--profile py313-mordredcommunity` for the clean community-Mordred profile.
- Use `--profile py313-original-mordred` only for legacy original Mordred compatibility testing.
