# Physics-Mamba model card

## Intended use

Physics-Mamba forecasts global horizontal irradiance at previously unseen BSRN stations from a 24-hour local observation context, deterministic future solar geometry, and static station coordinates. It directly produces forecasts for 15, 30, 60, 180, and 360 minutes.

## Inputs

- 96 historical steps at 15-minute cadence
- 14 historical value channels and corresponding availability masks
- 7 known-future solar and calendar variables at each target horizon
- latitude, longitude, and elevation

The model does not use future measured meteorology.

## Output construction

The Mamba-2 sequence representation is mapped to five transmission-like coefficients bounded between zero and two. Each coefficient is multiplied by target-time top-of-atmosphere horizontal irradiance. Predictions are nonnegative and exactly zero during astronomical night.

## Training and evaluation

- Training stations: 33
- Validation stations: 7 unseen locations
- Test stations: 11 additional unseen locations
- Seeds: 42, 43, and 44 for the primary comparison
- Primary endpoint: daylight station-macro MAE

The released checkpoint metadata records the scaler statistics, feature definitions, model configuration, and validation results.

## Limitations

The model is contextual zero-shot, not measurement-free cold start. A held-out station must supply its recent observations. It does not use numerical weather prediction or cloud imagery, and forecast uncertainty grows at longer horizons when cloud development cannot be inferred from the past context. The 6-hour Physics-Mamba versus iTransformer interval included zero, so no superiority claim is made at that horizon.

## Data licence

The checkpoints do not contain or redistribute the BSRN time series. Independent evaluation requires users to obtain the original observations from PANGAEA under BSRN-1.0 and reconstruct the benchmark with the supplied scripts.
