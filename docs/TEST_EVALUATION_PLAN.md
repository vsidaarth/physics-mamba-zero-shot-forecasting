# Frozen sealed-test evaluation plan

## Opening rule

The eleven test stations may be evaluated once only after the final validation
freeze manifest has been created. Every evaluator must verify the manifest
SHA-256 before loading test-station arrays. Test output directories must not
exist beforehand and must never be overwritten.

## Frozen methods

Primary comparison:

- Physics-Mamba, seeds 42, 43 and 44.
- iTransformer, seeds 42, 43 and 44.

Physical-output ablation:

- Vanilla Mamba, seeds 42, 43 and 44.
- Mamba with astronomical night mask, seeds 42, 43 and 44.

Supporting fixed baselines:

- Persistence and smart persistence.
- Train-only climatology.
- LightGBM, seed 42, with one frozen model per horizon.
- LSTM, TCN, DLinear and PatchTST-style models, seed 42.

No model is retrained or adapted using test-station observations. The neural
models use their frozen training-station normalization statistics.

## Endpoints

- Primary: daylight station-macro MAE at 15 min, 30 min, 1 h, 3 h and 6 h.
- Secondary: station-macro RMSE, nMAE, micro metrics,
  daylight-to-daylight metrics and smart-persistence skill.
- Primary uncertainty: paired bootstrap over the eleven test stations after
  averaging seeds within station.

## Reporting rule

All results are reported regardless of direction. Validation claims are
replaced by test claims only where the sealed-test evidence supports them. No
hyperparameter, checkpoint, subgroup boundary, metric or manuscript claim may
be changed in response to the test results. Any later exploratory test analysis
must be clearly labelled post hoc.
