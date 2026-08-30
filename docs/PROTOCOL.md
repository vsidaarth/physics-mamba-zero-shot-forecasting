# Location-zero-shot forecasting protocol v1

## Primary claim

Given a recent observation context at a previously unseen BSRN station, forecast
future global horizontal irradiance without station-specific training or
fine-tuning. This is **contextual geographic zero-shot forecasting**, not a
claim of measurement-free cold start.

## Frozen station roles

- 33 training stations fit model parameters and normalization statistics.
- 7 geographically unseen validation stations select architectures,
  hyperparameters, context length, and stopping checkpoints.
- 11 geographically unseen test stations remain sealed until the pipeline and
  model-selection decisions are frozen.

The immutable assignments are in
`data/raw/bsrn_snapshot_957398/final_station_manifest.csv`. A test station's
records must never contribute to fitting, normalization, early stopping, or
model selection.

## Target and resolution

The primary target is 15-minute mean short-wave downward radiation (`SWD`),
reported as non-negative global horizontal irradiance in W/m². The primary
context is 24 hours (96 steps), with a context-length sweep from 6 hours to 7
days. Forecast horizons are 15 minutes, 30 minutes, 1 hour, 3 hours, and 6
hours.

All models and baselines use the same eligible forecast origins. The primary
96-step context must span exactly 24 hours at 15-minute cadence, contain valid
GHI in at least 80% of its steps, and have valid GHI at both the forecast origin
and target. The target must be exactly the requested number of 15-minute steps
after the origin. Models may use missingness masks within the remaining 20%.

Older BSRN records can have native cadences of 2, 3, 5, or 10 minutes. Cadence
is inferred independently for every monthly file. A 15-minute aggregate is
valid only when at least 80% of the expected native samples are available.
Native cadence and valid-sample counts remain in the processed data for audit.

## Quality control

BSRN quality codes are bit fields. The primary analysis excludes observations
with either physical-limit bit set (`QC & 3 != 0`). Extremely rare and
cross-instrument comparison flags are retained because they can represent real
events, including cloud enhancement. A strict `QC == 0` analysis is reported
as a sensitivity test.

No missing value is silently replaced with zero. Models receive explicit
availability masks. Irradiance is clipped to zero only after valid observations
have been averaged within a bin.

## Information available at prediction time

Allowed inputs are historical station observations, deterministic calendar and
solar-geometry variables, and static station coordinates/elevation. Future
measured temperature, humidity, pressure, or radiation components are not
allowed. A later experiment may add genuinely forecast NWP covariates, but it
must be reported as a separate information setting.

## Evaluation

All models use identical windows, features, horizons, station splits, and
metric code. Primary metrics are MAE, RMSE, normalized MAE, and skill relative
to smart persistence. Results are reported for daylight periods and for the
complete timeline, by station, region, season, horizon, and sky-variability
regime. Training time, peak GPU memory, parameter count, and inference latency
are reported to test the efficiency claim of state-space models.

Trainable neural models use at least three random seeds. Paired station-level
bootstrap confidence intervals are used for headline model differences.

## Development lock

The preprocessor handles training and validation stations by default. Creating
test artifacts requires an explicit `--include-test` flag. That flag should not
be used until model families, hyperparameter budgets, metrics, and ablations
are frozen in the experiment registry.
