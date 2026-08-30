# Final sealed-test results

The eleven geographically unseen test stations were evaluated once after the
validation package, checkpoints, metrics, figures and claims were frozen. No
model was retrained or adapted using test-station observations. Seeds are
averaged within station and uncertainty is estimated by a paired bootstrap over
the eleven stations with 100,000 replicates.

## Primary comparison

Daylight station-macro MAE in W/m², mean ± sample standard deviation across
seeds 42, 43 and 44:

| Horizon | Physics-Mamba | iTransformer | Paired difference | Relative reduction | 95% station-bootstrap CI | Station wins |
|---:|---:|---:|---:|---:|---:|---:|
| 15 min | **36.07 ± 0.43** | 36.64 ± 0.10 | -0.56 | 1.54% | [-0.82, -0.28] | 9/11 |
| 30 min | **46.37 ± 0.52** | 47.08 ± 0.11 | -0.71 | 1.50% | [-0.95, -0.45] | 11/11 |
| 1 h | **56.52 ± 0.51** | 57.58 ± 0.11 | -1.06 | 1.84% | [-1.26, -0.83] | 11/11 |
| 3 h | **76.23 ± 0.41** | 77.85 ± 0.21 | -1.62 | 2.08% | [-2.25, -1.13] | 11/11 |
| 6 h | **90.72 ± 0.39** | 91.45 ± 0.19 | -0.73 | 0.80% | [-1.75, 0.35] | 7/11 |

Negative paired differences favor Physics-Mamba. The sealed test supports a
geographical zero-shot advantage through 3 hours. No superiority claim is made
at 6 hours because its confidence interval includes zero.

## Three-seed physical-output ablation

| Horizon | Vanilla Mamba | + night mask | Physics-Mamba |
|---:|---:|---:|---:|
| 15 min | 36.83 ± 0.20 | 36.11 ± 0.30 | **36.07 ± 0.43** |
| 30 min | 46.98 ± 0.11 | **46.24 ± 0.14** | 46.37 ± 0.52 |
| 1 h | 57.41 ± 0.02 | 56.76 ± 0.20 | **56.52 ± 0.51** |
| 3 h | 78.16 ± 0.56 | 77.31 ± 0.96 | **76.23 ± 0.41** |
| 6 h | 92.51 ± 0.64 | 91.89 ± 1.14 | **90.72 ± 0.39** |

Night masking improves over vanilla Mamba at every horizon with paired
station-bootstrap intervals below zero. Physics-Mamba improves over vanilla at
every horizon. Relative to night masking alone, the complete envelope is
supported at 3 and 6 hours; differences at 15 minutes, 30 minutes and 1 hour
are inconclusive. At 30 minutes the mean is 0.13 W/m² worse than night masking,
which is retained and reported.

## Supporting test baselines

| Method | 15 min | 30 min | 1 h | 3 h | 6 h |
|---|---:|---:|---:|---:|---:|
| Persistence | 48.11 | 74.42 | 116.89 | 251.86 | 348.79 |
| Smart persistence | 38.44 | 52.70 | 69.67 | 128.54 | 226.70 |
| Train-only climatology | 124.19 | 124.29 | 124.45 | 124.79 | 124.77 |
| LightGBM | 36.61 | 48.04 | 58.99 | 82.07 | 97.18 |
| LSTM | 37.37 | 47.97 | 58.75 | 80.25 | 94.85 |
| TCN | 38.19 | 48.02 | 59.35 | 80.48 | 94.00 |
| DLinear | 55.82 | 64.29 | 74.20 | 97.20 | 109.84 |
| PatchTST-style | 37.54 | 48.88 | 60.35 | 84.48 | 101.41 |
| iTransformer (3 seeds) | 36.64 | 47.08 | 57.58 | 77.85 | 91.45 |
| Physics-Mamba (3 seeds) | **36.07** | **46.37** | **56.52** | **76.23** | **90.72** |

Values are daylight station-macro MAE in W/m². Single-seed supporting neural
models are retained as fixed benchmark references; the primary neural
comparison and physical-output ablation use three seeds.

## Audit status

- Freeze manifest SHA-256: `35435f3639f0ee6583504376cf4450d7a29800f6661624cd3c3421d1b73ddfbf`.
- Test stations: 11.
- Partial metrics inspected before suite completion: false.
- Execution irregularities and compatibility handling are documented in
  `SEALED_TEST_AUDIT.md`.
