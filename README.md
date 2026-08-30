# Physics-Mamba

Physics-constrained state-space learning for geographical zero-shot solar irradiance forecasting.

This repository accompanies the manuscript **Physics-Mamba: Physics-Constrained State-Space Learning for Geographical Zero-Shot Solar Irradiance Forecasting**. It contains the reproducibility code, frozen protocol, station split metadata, trained checkpoints, sealed-test results, and paper figures. It does not contain BSRN observations.

## Main experiment

One global model uses the latest 24 hours of 15-minute observations to forecast global horizontal irradiance at 15, 30, 60, 180, and 360 minutes. The location split is fully station-disjoint:

- 33 stations for training and normalization
- 7 geographically unseen stations for validation and model selection
- 11 additional unseen stations for the sealed test

Physics-Mamba combines a Mamba-2 encoder with deterministic target-time solar geometry. Its output is a bounded transmission-like coefficient multiplied by top-of-atmosphere horizontal irradiance. This gives nonnegative predictions and exact zero output during astronomical night.

The primary sealed-test endpoint is daylight station-macro MAE. Mean MAE across seeds 42, 43, and 44 was:

| Model | 15 min | 30 min | 1 h | 3 h | 6 h |
|---|---:|---:|---:|---:|---:|
| Physics-Mamba | **36.07** | **46.37** | **56.52** | **76.23** | **90.72** |
| iTransformer | 36.64 | 47.08 | 57.58 | 77.85 | 91.45 |

Values are W/m2. Paired station-bootstrap intervals support the difference through 3 hours. The 6-hour interval includes zero.

## Repository contents

- `scripts/`: download, audit, preprocessing, training, evaluation, analysis, and figure-generation code
- `configs/protocol_v1.json`: machine-readable forecasting protocol
- `manifests/`: fixed station assignments, archive checksums, coverage audit, and preprocessing audit summaries
- `freezes/`: frozen pre-test manifests and hashes
- `checkpoints/`: trained model checkpoints, model metadata, and validation summaries
- `results/`: sealed-test station metrics, paired analyses, uncertainty tables, and subgroup analyses
- `figures/`: publication figures in PNG and PDF formats
- `docs/`: protocol, test plan, execution audit, and concise results reports
- `tests/`: preprocessing and baseline unit tests

## Data access

The measured observations come from the quality-controlled BSRN snapshot archived by PANGAEA:

> Lanconelli, C. et al. (2023). Baseline surface radiation data snapshot 2023-03-31. PANGAEA. https://doi.org/10.1594/PANGAEA.957398

The snapshot is distributed under the Baseline Surface Radiation Network License 1.0. BSRN data are therefore not copied into this repository. Obtain the original station archives directly from PANGAEA and cite the snapshot and the relevant station datasets. See [`DATA_ACCESS.md`](DATA_ACCESS.md) for the complete data statement and reconstruction procedure.

## Reconstruct the benchmark

Create a CPU environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-cpu.txt
```

Download and audit the canonical snapshot:

```bash
python scripts/download_bsrn_snapshot.py data/raw/bsrn_snapshot_957398 --workers 2 --attempts 40
python scripts/audit_bsrn_snapshot.py data/raw/bsrn_snapshot_957398
```

The resulting `final_station_manifest.csv` should match the copy in `manifests/`. Rebuild the 15-minute Parquet files:

```bash
python scripts/preprocess_bsrn.py \
  data/raw/bsrn_snapshot_957398 \
  data/processed/bsrn_15min \
  --include-test

python scripts/audit_processed_bsrn.py data/processed/bsrn_15min
python -m unittest discover -s tests -v
```

The `--include-test` switch is explicit because test-station preprocessing was locked during model development. It is appropriate for independent reproduction of the completed, frozen experiment.

## Train Physics-Mamba

The GPU run used PyTorch 2.4.1 with CUDA 12.4, `mamba-ssm==2.2.5`, and `causal-conv1d` tag `v1.6.2.post1` on an RTX 4090. The environment commands are recorded in `scripts/setup_runpod.sh` and `requirements-gpu.txt`.

Example:

```bash
python scripts/train_lstm.py \
  data/processed/bsrn_15min \
  outputs/physics_mamba_seed42 \
  --architecture physics_mamba \
  --hidden-size 128 \
  --layers 3 \
  --batch-size 1024 \
  --epochs 20 \
  --patience 4 \
  --seed 42
```

Repeat with seeds 43 and 44. The exact configuration used by every released checkpoint is stored beside it in `metadata.json`.

## Evaluate the frozen test

The complete sealed-test procedure, including the expected freeze hash, is documented in `docs/TEST_EVALUATION_PLAN.md` and `docs/SEALED_TEST_AUDIT.md`. The frozen suite is run with:

```bash
python scripts/run_frozen_test_suite.py \
  . \
  freezes/validation_freeze_manifest.json \
  --expected-freeze-sha256 35435f3639f0ee6583504376cf4450d7a29800f6661624cd3c3421d1b73ddfbf
```

The repository already contains the completed station-level outputs and aggregate analyses. No raw or processed BSRN observations are present.

## Checkpoints

Released weights cover Physics-Mamba, vanilla Mamba, night-mask Mamba, iTransformer, LSTM, TCN, DLinear, PatchTST-style, and the five horizon-specific LightGBM models. `CHECKPOINTS.sha256` provides an integrity hash for every released model file. Duplicate `best_state.pt` copies were intentionally omitted.

## Citation

If you use this code, protocol, or released checkpoints, cite the accompanying manuscript and this repository. If you use the observations, also cite the BSRN snapshot and the relevant station datasets supplied by PANGAEA.

## Licence

The code and original repository material are released under the MIT License. BSRN observations remain subject to BSRN-1.0 and are not redistributed here.
