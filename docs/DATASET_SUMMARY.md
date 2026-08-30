# BSRN download and audit summary

Completed on 2026-07-18.

## Source and integrity

- Dataset: BSRN Toolbox snapshot, DOI `10.1594/PANGAEA.957398`
- Canonical station archives downloaded: 76 of 76
- Compressed size: 7,344,832,122 bytes (6.84 GiB)
- Monthly quality-controlled combined files indexed: 12,879
- Every completed download passed ZIP validation before its atomic rename
- SHA-256 hashes were generated for all 76 archives

## Benchmark cohort

- Minimum usable continuity: 60 consecutive months
- Selected locations: 51
- Train: 33 locations
- Validation: 7 locations
- Unseen zero-shot test: 11 locations
- Regions represented: Africa 2, Antarctica 4, Asia 7, Europe/Arctic 13,
  North America 14, Oceania 7, South America 4

The cohort excludes stations below the continuity threshold and one member of
each close pair: E13 (near BIL), BOU (near BOS), and DWN (near DAR).

## Data character

The monthly text members contain one-minute measured radiation and supporting
meteorology, including downward shortwave radiation, direct and diffuse
components where available, longwave radiation, pressure, relative humidity,
air temperature, solar geometry, and source quality-control flags. Exact
variable availability can vary by station and period and must be represented
with masks during preprocessing.

## Leakage safeguards

The split unit is the station, not an individual row or random time window.
All records from a test station stay out of model development. Close station
pairs were removed before splitting. Any scaler or normalization statistics
must be fitted on training stations only. A separate chronological holdout
inside each station is still required to measure forecasting through time.

## Next implementation stage

Stream the selected ZIP members, apply the supplied QC flags, aggregate the
one-minute observations into a regular forecasting resolution, write compact
Parquet shards, and define horizon-specific chronological windows. That
processed representation can then be copied to RunPod for GPU training.
