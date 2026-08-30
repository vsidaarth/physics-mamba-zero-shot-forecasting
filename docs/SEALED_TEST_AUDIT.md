# Sealed-test execution audit

## Preflight event

The first suite invocation stopped before inference because no test Parquet
files existed on RunPod. It produced only a 50-byte log stating that zero test
stations were found. The directory was preserved as
`results/sealed_test_preflight_failed_no_data`; no predictions or metrics were
created.

The eleven test stations were then preprocessed locally using the frozen
protocol. Their byte-level hashes are recorded in
`freezes/final_pretest_freeze_2026-08-10/test_preprocessing_manifest.json` and
were verified after upload.

## Supporting-baseline compatibility event

The single-pass suite completed all twelve primary and physical-ablation runs,
then stopped before LSTM inference. The legacy seed-42 LSTM metadata predates
the `hyperparameters.architecture` field and raised `KeyError: architecture`.
No LSTM output directory or metric file was created.

The evaluator was changed only to fall back to the frozen metadata field
`model` (`LSTM`) when `hyperparameters.architecture` is absent. Checkpoint
content and SHA-256 verification, data eligibility, metrics and every model
parameter remain unchanged. The twelve completed primary/ablation runs are not
rerun. A dedicated resume script evaluates only the seven missing supporting
methods. No partial test metric was inspected before this compatibility fix.
