# Final Results Status

Protocol: `bsrn-location-zero-shot-v1-final-pretest-freeze-2026-08-10`.

The single pre-specified evaluation of the eleven geographically unseen test stations is complete. The test set was opened only after the validation package, checkpoints, analysis plan, and claims were frozen. All methods use identical eligible forecast origins: a 96-step (24-hour) context with at least 80% valid GHI, a valid current observation, and a valid target at the exact requested lead time.

## Primary sealed-test result

Daylight station-macro MAE in W/m², mean ± sample standard deviation across seeds 42, 43, and 44:

| Method | 15 min | 30 min | 1 h | 3 h | 6 h |
|---|---:|---:|---:|---:|---:|
| iTransformer | 36.64 ± 0.10 | 47.08 ± 0.11 | 57.58 ± 0.11 | 77.85 ± 0.21 | 91.45 ± 0.19 |
| Physics-Mamba | **36.07 ± 0.43** | **46.37 ± 0.52** | **56.52 ± 0.51** | **76.23 ± 0.41** | **90.72 ± 0.39** |

The paired station bootstrap supports lower Physics-Mamba error at 15 minutes, 30 minutes, 1 hour, and 3 hours. At 6 hours the mean remains favorable, but the 95% interval includes zero, so superiority is not claimed.

| Horizon | Physics-Mamba − iTransformer | 95% station-bootstrap CI | Wins |
|---:|---:|---:|---:|
| 15 min | −0.56 | [−0.82, −0.28] | 9/11 |
| 30 min | −0.71 | [−0.95, −0.45] | 11/11 |
| 1 h | −1.06 | [−1.26, −0.83] | 11/11 |
| 3 h | −1.62 | [−2.25, −1.13] | 11/11 |
| 6 h | −0.73 | [−1.75, 0.35] | 7/11 |

## Physical-output ablation

| Method | 15 min | 30 min | 1 h | 3 h | 6 h |
|---|---:|---:|---:|---:|---:|
| Vanilla Mamba | 36.83 ± 0.20 | 46.98 ± 0.11 | 57.41 ± 0.02 | 78.16 ± 0.56 | 92.51 ± 0.64 |
| + astronomical night mask | 36.11 ± 0.30 | **46.24 ± 0.14** | 56.76 ± 0.20 | 77.31 ± 0.96 | 91.89 ± 1.14 |
| + complete solar envelope | **36.07 ± 0.43** | 46.37 ± 0.52 | **56.52 ± 0.51** | **76.23 ± 0.41** | **90.72 ± 0.39** |

Night masking improves over vanilla Mamba at every horizon. The complete solar envelope improves over vanilla Mamba throughout and adds its clearest incremental benefit beyond night masking at 3 and 6 hours. The 30-minute night-mask mean is slightly lower than the complete-envelope mean; the paired interval for their difference includes zero.

## Supporting baselines

| Method | 15 min | 30 min | 1 h | 3 h | 6 h |
|---|---:|---:|---:|---:|---:|
| Persistence | 48.11 | 74.42 | 116.89 | 251.86 | 348.79 |
| Smart persistence | 38.44 | 52.70 | 69.67 | 128.54 | 226.70 |
| Train-only global climatology | 124.19 | 124.29 | 124.45 | 124.79 | 124.77 |
| LightGBM | 36.61 | 48.04 | 58.99 | 82.07 | 97.18 |
| LSTM | 37.37 | 47.97 | 58.75 | 80.25 | 94.85 |
| TCN | 38.19 | 48.02 | 59.35 | 80.48 | 94.00 |
| DLinear | 55.82 | 64.29 | 74.20 | 97.20 | 109.84 |
| PatchTST | 37.54 | 48.88 | 60.35 | 84.48 | 101.41 |

## Reproducibility state

- Freeze manifest SHA-256: `35435f3639f0ee6583504376cf4450d7a29800f6661624cd3c3421d1b73ddfbf`.
- All eleven test Parquet files were hash-verified after upload.
- No partial test metrics were inspected before the suite-completion sentinel was written.
- A preflight attempt that found no test data was preserved separately; it performed no inference and produced no metrics.
- One compatibility fallback was required for legacy LSTM metadata. It changed model loading only, not weights, data, predictions, metrics, or the analysis plan.
- The completed package contains all planned primary, ablation, and supporting-model outputs.

The defensible paper claim is therefore narrow and strong: Physics-Mamba improves geographical zero-shot irradiance forecasting over the matched iTransformer from 15 minutes through 3 hours, while the 6-hour contrast remains inconclusive.
