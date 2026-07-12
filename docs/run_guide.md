# XTQSPRKinetic Run Guide

**Maintainer:** Hamidreza Najafi (XThermo Lab)  
**Last updated:** 2026-06-08

This guide explains how to run the **XTQSPRKinetic** repository layout using:
- `main/main.py` (primary XTQSPRKinetic pipeline driver), and
- `main/reaction_property_performance.py` (companion evaluator for saved `pipeline.pkl` artifacts).

It complements the repository `README.md` and focuses on practical execution workflows.

---

## 1. Repository layout assumptions

This guide assumes the repository layout:

```text
XTQSPRKinetic/
  requirements.txt
  requirements-optional.txt
  requirements-all.txt
  requirements-all-py313-mordredcommunity.txt
  requirements-bootstrap-py313.txt
  requirements-all-update-py313-original-mordred.txt
  requirements-runtime-py313-original-mordred.txt
  gen_xtqspr_requirements.py
  configs/
  data/
  docs/
    run_guide.md
    setup/
      INSTALL_PY313_MORDREDCOMMUNITY.md
      INSTALL_PY313_ORIGINAL_MORDRED_LEGACY.md
      requirements-detected-report.md
  main/
    main.py
    reaction_property_performance.py  # generalized evaluator
  src/
    mordred_xtqspr_patch.py  # optional explicit-H Mordred runtime patch
  output/
  eval_output/
```

### Roles of the main directories
- `src/`: XTQSPRKinetic pipeline modules (loaded by `main/main.py`), including the optional explicit-H Mordred runtime patch module
- `main/`: executable entry scripts for XTQSPRKinetic
- `configs/`: preset and user-created YAML/JSON configurations
- `data/`: default test inputs and other input datasets
- `output/`: default output root for XTQSPRKinetic pipeline runs
- `eval_output/`: default output root for `reaction_property_performance.py`
- `docs/`: project documentation; installation/setup guides and requirements reports are kept under `docs/setup/`

---

## 2. Environment setup and requirements files

Before running XTQSPRKinetic, install dependencies using the standard generated requirements files or one of the Python 3.13 Mordred profiles.

### 2.1 Standard requirements files

Use this for standard XTQSPRKinetic runs:

```bash
python -m pip install -r requirements.txt
```

Use this when you need optional clustering routes, topic coherence, Excel export, compression backends, or optional Boruta estimators:

```bash
python -m pip install -r requirements-optional.txt
```

Use this for full manuscript reproduction, full diagnostics, SHAP/descriptive figures, and external evaluation when the dependency resolver is compatible with your selected Mordred route:

```bash
python -m pip install -r requirements-all.txt
```

### 2.2 Python 3.13 recommended clean profile: `mordredcommunity`

For Python 3.13, the recommended clean profile uses `mordredcommunity` instead of the original `mordred==1.2.0`. The installed distribution is `mordredcommunity`, but the code can still use:

```python
from mordred import Calculator, descriptors
```

Detailed setup guide:

```text
docs/setup/INSTALL_PY313_MORDREDCOMMUNITY.md
```

Install with:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-all-py313-mordredcommunity.txt
python -m pip check
```

This is the preferred Python 3.13 route because it keeps NumPy 2.x compatibility for `drfp==0.3.7` and avoids the DRFP/NumPy metadata conflict.

### 2.3 Python 3.13 legacy profile: original `mordred==1.2.0`

Use this only when strict legacy reproducibility with original Mordred is required. Detailed setup guide:

```text
docs/setup/INSTALL_PY313_ORIGINAL_MORDRED_LEGACY.md
```

```bash
python -m pip install --upgrade -r requirements-bootstrap-py313.txt
python -m pip install -r requirements-all-update-py313-original-mordred.txt
python -m pip install --force-reinstall -r requirements-runtime-py313-original-mordred.txt
```

The final runtime file pins:

```text
mordred==1.2.0
numpy==1.26.4
```

This route can be runtime-working, but `pip check` may report that `drfp==0.3.7` requires `numpy>=2.2.2`. That is an expected metadata conflict for the legacy override route. Use the validation test below rather than treating `pip check` as the only acceptance criterion.

### 2.4 Generate all requirements profiles

The revised generator can write both Mordred routes:

```bash
python gen_xtqspr_requirements.py ./src ./main \
  --xtqspr-defaults \
  --pin none \
  --profile py313-both \
  --out-dir .
```

Supported profiles:

```text
standard
py313-mordredcommunity
py313-original-mordred
py313-both
py313-mordred   # backward-compatible alias for py313-both
all             # alias for py313-both
```

Recommended repository layout:

```text
XTQSPRKinetic/
  requirements.txt
  requirements-optional.txt
  requirements-all.txt
  requirements-all-py313-mordredcommunity.txt
  requirements-bootstrap-py313.txt
  requirements-all-update-py313-original-mordred.txt
  requirements-runtime-py313-original-mordred.txt
  docs/
    setup/
      INSTALL_PY313_MORDREDCOMMUNITY.md
      INSTALL_PY313_ORIGINAL_MORDRED_LEGACY.md
      requirements-detected-report.md
```

The generated `docs/setup/requirements-detected-report.md` file is optional for normal users, but it is useful for maintainers because it documents detected imports, package-name mappings, and optional dependency decisions.

If the generator writes setup documents to the repository root, organize them with:

```bash
mkdir -p docs/setup
mv -f INSTALL_PY313_MORDREDCOMMUNITY.md docs/setup/ 2>/dev/null || true
mv -f INSTALL_PY313_ORIGINAL_MORDRED_LEGACY.md docs/setup/ 2>/dev/null || true
mv -f requirements-detected-report.md docs/setup/ 2>/dev/null || true
```

### 2.5 Validate the environment

Run this after installation:

```bash
python - << "PY"
import numpy
import pandas
import sklearn
import xgboost
from rdkit import Chem
from mordred import Calculator, descriptors

mol = Chem.MolFromSmiles("CC")
calc = Calculator(descriptors, ignore_3D=True)
vals = calc(mol).asdict()

print("numpy", numpy.__version__)
print("mordred descriptors", len(vals))
print("rdkit molecule OK", mol is not None)
PY
```

For the `mordredcommunity` profile, `python -m pip check` should also be clean. For the original Mordred legacy profile, a DRFP/NumPy metadata conflict may appear in `pip check`; this is expected for that compatibility-override route.

---

## 3. Running XTQSPRKinetic (primary pipeline) via `main/main.py`

### 3.1 What `main/main.py` does
`main/main.py` is the primary driver for the XTQSPRKinetic pipeline. It:
- resolves `--config` from `configs/` when a bare name is provided,
- loads modules directly from `src/` (no package installation required for the driver),
- normalizes project paths against the fixed repo layout (`data/`, `output/`, `configs/`),
- optionally runs the auto-chooser when `family_selection_mode: auto`, and
- runs the full pipeline with the merged configuration.

### 3.2 CLI arguments supported by `main/main.py`
```text
--config <name-or-path>         (required)
--chooser-log <filename>        (optional; auto-mode chooser log filename)
--dump-effective-config         (optional; writes resolved_config.yaml to output_dir)
```

> `main/main.py` does **not** expose arbitrary config-key CLI overrides. Run-specific settings should be managed in YAML/JSON config files under `configs/` (or explicit config paths).

### 3.3 Quick start (XTQSPRKinetic)

#### A) Use a config by bare name (resolved under `configs/`)
```bash
python main/main.py --config balanced
```

#### B) Use an explicit config path
```bash
python main/main.py --config configs/benchmark_lda_fcm.yaml
```

#### C) Dump the effective (merged + resolved) config into the run folder
```bash
python main/main.py --config benchmark_lda_fcm --dump-effective-config
```

#### D) Auto family-scheme selection (configured in the YAML/JSON)
Set `family_selection_mode: auto` in the config file, then run:

```bash
python main/main.py --config auto_scheme --chooser-log auto_chooser.log
```

### 3.4 Config path behavior (`--config` resolution)
`main/main.py` resolves `--config` in this order:
1. absolute path (if the file exists),
2. path as provided (relative to the current working directory),
3. bare name without extension -> tries `configs/<name>.yml`, `.yaml`, `.json`,
4. path under `configs/`.

Examples:
```bash
python main/main.py --config balanced
python main/main.py --config balanced.yml
python main/main.py --config configs/my_runs/lda_soft.yaml
python main/main.py --config /absolute/path/to/run_config.yaml
```

### 3.5 Normal/public configs and the Diagnostic Audit module
Preset configs are intended for normal/public runs and therefore keep the optional diagnostic audit disabled by default:

```yaml
diagnostics_enable: false
diagnostics_random_controls_mode: off
```

To generate diagnostic-audit or manuscript-reproducibility outputs, explicitly enable the module in a dedicated run config:

```yaml
diagnostics_enable: true
diagnostics_random_controls_mode: size_matched
diagnostics_random_controls_n: 25
```

The `size_matched` random-control mode randomly permutes the learned-family labels, preserving the exact family-size distribution while removing the chemical association between reactions and learned families. Diagnostic outputs are written under `diagnostics/`.

### 3.6 Project path normalization in the main driver
The driver applies project-aware path normalization, including:
- `input_csv` -> resolved under `data/` if relative
- `output_dir` -> resolved under `output/` if relative
- cache directories (e.g., descriptor/Boruta caches) -> resolved under the run `output_dir` when relative or enabled without explicit paths

This behavior makes config files more portable across machines when using the standard repository layout.

### 3.7 Normal/public configs versus diagnostics
The code includes an optional diagnostic audit layer. The preset configs supplied here are normal/public run profiles, so they explicitly keep this layer disabled:

```yaml
diagnostics_enable: false
diagnostics_random_controls_mode: off
```

For diagnostic-audit or manuscript-reproducibility runs, enable the audit layer explicitly in a separate run config, for example:

```yaml
diagnostics_enable: true
diagnostics_random_controls_mode: size_matched
diagnostics_random_controls_n: 25
```

The `size_matched` random-control mode preserves the exact learned-family size distribution by permuting learned-family labels, while removing the chemical association between reactions and learned families.



### 3.8 Target-column behavior

XTQSPRKinetic uses `target_cols` as the canonical list of supervised continuous reaction-property targets:

```yaml
target_cols: ["lnA", "n", "Ea"]
```

This default preserves the Arrhenius case study. For a generalized reaction-property run, include the desired numeric target columns in the input CSV and set `target_cols`, for example:

```yaml
target_cols: ["Ea_forward", "Ea_reverse", "delta_H_rxn"]
```

The minimum training input is therefore `reaction_smiles` plus every column named in `target_cols`. Optional mechanistic alignment diagnostics additionally require the column named by `rmg_family_col` when those diagnostics are enabled.


### 3.9 Optional explicit-Hydrogen-aware Mordred descriptor mode

For radical-rich C/H kinetic networks, XTQSPRKinetic can apply a controlled runtime patch to Mordred/MordredCommunity descriptor calculation. This avoids manually editing installed public package files while preserving the patched descriptor behavior in a reproducible, config-controlled way.

Enable the mode only when needed:

```yaml
mordred_xtqspr_explicit_h_patch_enable: true
```

Use the default for standard Mordred/MordredCommunity behavior:

```yaml
mordred_xtqspr_explicit_h_patch_enable: false
```

The patch is intended for networks containing hydrogen-sensitive radical species such as:

```text
CH4, [CH3], [CH2], [CH], [H], and [H][H]
```

The patch changes descriptor values and is part of the Mordred cache key. Therefore, for the first run after changing this setting, use a fresh descriptor cache or disable cache reuse:

```yaml
descriptor_cache: false
```

When this mode is enabled, describe the resulting descriptors as **XTQSPRKinetic explicit-hydrogen-aware modified Mordred/MordredCommunity descriptors**.


### 3.10 Bootstrap/hybrid-routing and QDA feasibility safeguards

The modular pipeline separates Bootstrap diagnostics from predictive routing. Bootstrap stability artifacts can be generated without automatically changing soft-routing predictions. The recommended conservative controls are:

```yaml
bootstrap_use_for_soft_routing: false
bootstrap_use_for_hybrid_routing: true
bootstrap_quality_gate_enable: true
bootstrap_min_median_modal_vote: 0.65
bootstrap_max_median_norm_entropy: 0.55
bootstrap_min_hard_fraction: 0.05
bootstrap_max_hard_fraction: 0.95
bootstrap_sampling_mode: bootstrap
```

The pipeline records both:

```text
routing_mode_requested
routing_mode
```

`routing_mode_requested` is the config request, while `routing_mode` is the route actually executed after Bootstrap availability, quality, and hard-fraction checks.

For `lda+qda` candidates, the final split code checks LDA-family class counts before fitting QDA:

```yaml
qda_feasibility_gate_enable: true
qda_min_class_count: 5
qda_auto_skip_infeasible: true
```

In Auto-Chooser mode, infeasible QDA candidates are skipped cleanly. In fixed/manual runs, the pipeline raises a clear feasibility error instead of failing inside scikit-learn QDA covariance estimation.

---

## 4. Running the companion evaluator: `main/reaction_property_performance.py`

`reaction_property_performance.py` is the generalized companion program for evaluating a saved XTQSPRKinetic `pipeline.pkl` on a reaction-network CSV and exporting predictions + performance summaries. It replaces the older Arrhenius-specific `kinetics_performance.py` naming. If a compatibility script using the earlier kinetics-performance naming is kept in the repository, it should point to the same generalized logic.

### 4.1 What it evaluates
The evaluator uses the saved `pipeline.pkl` as the source of truth for:
- family/topic routing behavior,
- feature requirements and feature ordering,
- the Mordred descriptor mode, including `mordred_xtqspr_explicit_h_patch_enable` when stored in `config_snapshot`,
- model/scaler objects,
- routing mode behavior (hard / soft / hybrid / auto), including the actual `routing_mode` stored in `pipeline.pkl`, and
- prediction reproducibility on new or comparison datasets.

For saved pipelines that used Bootstrap-guided hybrid routing, exact training-time Bootstrap stability decisions are not available for new external reactions. Therefore, the final evaluator uses soft routing in `routing_mode: auto` unless approximate hybrid evaluation is explicitly allowed.

It is designed for **evaluation/comparison**, not retraining.

### 4.2 Typical outputs (written under `eval_output/` by default)
- `<input_basename>_property_performance.xlsx`
- `topic_proba_pred.csv`
- `family_assignments_pred.csv`
- `eval_summary.txt`
- `property_eval.log`

If the input CSV contains the trained target columns recorded in `pipeline.pkl` / `config_snapshot["target_cols"]`, the evaluator also computes performance metrics. The default target set remains `lnA`, `n`, and `Ea`.

### 4.3 CLI usage patterns
`reaction_property_performance.py` imports `XTQSPRKinetic.src.*`, so package import resolution matters.

#### Recommended (package-style invocation from the parent directory of the repository)
If your directory structure is:
```text
<parent>/XTQSPRKinetic/
```
run:

```bash
python -m XTQSPRKinetic.main.reaction_property_performance --help_config
```

#### Example evaluation run (package-style)
```bash
python -m XTQSPRKinetic.main.reaction_property_performance \
  --input_csv XTQSPRKinetic/data/RS_1.csv \
  --pipeline_pkl XTQSPRKinetic/output/my_run/pipeline.pkl \
  --output_dir XTQSPRKinetic/eval_output/my_run_eval \
  --routing_mode auto
```

#### Alternative (running from inside the repository root)
If you want to run `python main/reaction_property_performance.py` from inside the repo root, ensure the **parent directory of the repository** is on `PYTHONPATH`:

```bash
PYTHONPATH="$(dirname "$PWD")" python main/reaction_property_performance.py --help_config
```

On Windows PowerShell (example):
```powershell
$env:PYTHONPATH = (Resolve-Path ..).Path
python .\main\reaction_property_performance.py --help_config
```

### 4.4 Configuring the evaluator
`reaction_property_performance.py` supports:
- `--help_config` to print configurable keys and defaults,
- `--config <yaml/json>` for a config file, and
- direct CLI overrides for its `DEFAULT_CONFIG` keys (e.g., `--input_csv`, `--pipeline_pkl`, `--output_dir`, `--routing_mode`, `--hybrid_metric`, `--hybrid_threshold`, `--allow_hybrid_approximation`, `--normalize_routing_probabilities`).

Example with a config file plus overrides:
```bash
python -m XTQSPRKinetic.main.reaction_property_performance \
  --config XTQSPRKinetic/configs/eval_compare.yaml \
  --pipeline_pkl XTQSPRKinetic/output/benchmark_lda_fcm/pipeline.pkl \
  --input_csv XTQSPRKinetic/data/RS_external.csv \
  --routing_mode hybrid \
  --hybrid_metric entropy \
  --hybrid_threshold 0.35 \
  --allow_hybrid_approximation true \
  --output_dir XTQSPRKinetic/eval_output/external_compare
```

Use `--routing_mode auto` for the conservative default. Use `--allow_hybrid_approximation true` only when you intentionally want the evaluator to approximate hybrid routing from external routing probabilities rather than from training-time Bootstrap stability diagnostics.

---

## 5. Suggested end-to-end workflow (train + evaluate)

### Workflow A: Train a pipeline, then evaluate it on a comparison dataset
1. Run XTQSPRKinetic with `main/main.py` using a run config in `configs/`.
2. Preserve the exact run config inside the run output folder.
3. Locate the generated `pipeline.pkl` in the XTQSPRKinetic output directory.
4. Run `main/reaction_property_performance.py` on a comparison CSV.
5. Archive the evaluator config/command alongside the `eval_output/` artifacts.
6. If `mordred_xtqspr_explicit_h_patch_enable` was used during training, keep the same setting for evaluation or rely on the value stored in `pipeline.pkl/config_snapshot`.

### Example sequence
```bash
# Step 1: XTQSPRKinetic training/inference pipeline (from repo root)
python main/main.py --config benchmark_lda_fcm --dump-effective-config

# Step 2: Evaluator (from repo parent; package-style)
python -m XTQSPRKinetic.main.reaction_property_performance \
  --input_csv XTQSPRKinetic/data/RS_2.csv \
  --pipeline_pkl XTQSPRKinetic/output/benchmark_lda_fcm/pipeline.pkl \
  --output_dir XTQSPRKinetic/eval_output/benchmark_lda_fcm_RS2 \
  --routing_mode auto
```

---

## 6. Reproducibility practices (recommended)

### 6.1 Keep per-run configs with outputs
Because XTQSPRKinetic is configuration-dependent, the exact artifact set and behavior can vary across runs. Store the corresponding config file inside each run folder.

Suggested pattern:
```text
output/
  benchmark_lda_fcm_run01/
    config.yaml                  # user config for the run
    resolved_config.yaml         # optional; from --dump-effective-config
    pipeline.pkl
    ...other artifacts...

eval_output/
  benchmark_lda_fcm_RS2/
    eval_config.yaml             # optional (if using evaluator config file)
    ...evaluation artifacts...
```

### 6.2 Use `--dump-effective-config` for XTQSPRKinetic runs
This is especially helpful when relative paths are normalized by `main/main.py`, or when defaults are merged with user overrides.

### 6.3 Record commands used to launch runs
Keep the exact launch commands (or a small shell/batch script) for training and evaluation runs.

---

## 7. Troubleshooting

### Problem: `Config file not found` in `main/main.py`
- Check whether the config name/path exists.
- If using a bare name, confirm the file is in `configs/` and has `.yml`, `.yaml`, or `.json`.
- Use an explicit path to avoid ambiguity.

### Problem: `ModuleNotFoundError: No module named 'XTQSPRKinetic'` when running `reaction_property_performance.py`
- Use the package-style invocation from the **parent directory** of the repository:
  `python -m XTQSPRKinetic.main.reaction_property_performance ...`
- Or set `PYTHONPATH` to the parent directory before running `python main/reaction_property_performance.py`.

### Problem: Missing optional dependency errors
Some methods/features require optional libraries (e.g., `scikit-fuzzy`, `hdbscan`, `rdkit`, `gensim`, `lightgbm`, `openpyxl`).
Install the relevant package(s), install `requirements-optional.txt` / `requirements-all.txt`, or disable those features in the config.


### Problem: `routing_mode auto` falls back to soft routing for a Bootstrap-hybrid pipeline
This is expected in the final evaluator. Bootstrap stability tables describe the training reactions and cannot be exactly reconstructed for new external reactions. Auto mode therefore uses the actual saved routing metadata but avoids treating external hybrid routing as exact Bootstrap-hybrid routing. To force an approximate hybrid evaluation, set:

```bash
--routing_mode hybrid --allow_hybrid_approximation true
```

### Problem: Boolean evaluator CLI options unexpectedly override defaults
Use the final evaluator version, where boolean options accept explicit values such as `true` or `false`. This avoids old `store_true` behavior that could accidentally override YAML/default boolean settings when flags were omitted.

### Problem: Evaluator writes predictions but no metrics
Your evaluation CSV likely does not contain one or more of the trained target columns recorded in `pipeline.pkl` / `target_cols`.
This is expected behavior; predictions can still be exported without scoring.


### Problem: Patched Mordred descriptors appear unchanged or old results are reused
- Confirm that the run config contains `mordred_xtqspr_explicit_h_patch_enable: true`.
- Confirm that the current run is not reusing an old descriptor cache. Use `descriptor_cache: false` for the first patched run, or set a fresh `descriptor_cache_dir`.
- In parallel Mordred runs, the patch is applied inside each worker before the worker constructs its local Mordred `Calculator`.
- Do not rely on `calc.config(explicit_hydrogens=True)` for this XTQSPRKinetic mode; the intended behavior comes from `src/mordred_xtqspr_patch.py`.

### Problem: `pip check` reports `drfp 0.3.7 has requirement numpy>=2.2.2`
This is expected only for the **original Mordred legacy profile**, because that profile forces `numpy==1.26.4` for `mordred==1.2.0` compatibility. Use the `mordredcommunity` profile for a clean Python 3.13 environment, or accept the legacy route only after the XTQSPRKinetic runtime validation test passes.

### Problem: Mordred import fails after installing `mordredcommunity`
The pip distribution is named `mordredcommunity`, but the import path should remain:

```python
from mordred import Calculator, descriptors
```

If this fails, reinstall the clean profile in a fresh environment and verify that no old `mordred==1.2.0` installation remains.

---

## 8. Developer notes for documentation consistency

This section is intended for maintainers/contributors to keep the README, run guide, and repository structure descriptions consistent across releases.

- **Primary XTQSPRKinetic entry point:** `main/main.py`
- **Auxiliary evaluation tool:** `main/reaction_property_performance.py`
- **Default XTQSPRKinetic output directory:** `output/`
- **Default reaction_property_performance output directory:** `eval_output/`
- **Pipeline modules:** `src/`
- **Documentation:** `docs/`; setup/install documentation: `docs/setup/`
- **Preset/user configurations:** `configs/`
- **Default test and user input data:** `data/`

When updating documentation, ensure that:
1. `README.md` and `docs/run_guide.md` both describe `main/main.py` as the primary driver for XTQSPRKinetic.
2. `main/reaction_property_performance.py` is documented as a separate post-processing/evaluation utility for comparing results using generated pipeline `.pkl` files.
3. Any outdated references to alternative/older entry-point scripts are removed from user-facing documentation (unless explicitly mentioned for development history).
4. Output-directory descriptions remain aligned with the current defaults (`output/` for XTQSPRKinetic and `eval_output/` for reaction_property_performance).
5. Example commands remain consistent with the current CLI behavior and package import structure.
6. Installation guides and requirements detection reports remain under `docs/setup/`, while requirement `.txt` files stay at the repository root.
7. Documentation for `mordred_xtqspr_explicit_h_patch_enable` should remain consistent across `README.md`, `docs/run_guide.md`, tutorial configuration notes, and version history.
