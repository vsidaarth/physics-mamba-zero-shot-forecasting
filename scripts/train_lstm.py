#!/usr/bin/env python3
"""Train a global multi-horizon LSTM on station-disjoint BSRN data."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.evaluate_persistence import (
        CLEARSKY_FLOOR_WM2, CLEARSKY_INDEX_MAX, aggregate_metrics,
        forecast_origin_mask, ineichen_ghi, metric_row,
    )
except ModuleNotFoundError:
    from evaluate_persistence import (
        CLEARSKY_FLOOR_WM2, CLEARSKY_INDEX_MAX, aggregate_metrics,
        forecast_origin_mask, ineichen_ghi, metric_row,
    )


CONTEXT_FEATURES = (
    "ghi_wm2", "dni_wm2", "dhi_wm2", "lwd_wm2", "air_temperature_c",
    "relative_humidity_pct", "pressure_hpa", "solar_zenith_deg",
    "cos_zenith", "toa_horizontal_wm2", "utc_hour_sin", "utc_hour_cos",
    "day_of_year_sin", "day_of_year_cos",
)
FUTURE_FEATURES = (
    "solar_zenith_deg", "cos_zenith", "toa_horizontal_wm2",
    "utc_hour_sin", "utc_hour_cos", "day_of_year_sin", "day_of_year_cos",
)
STATIC_FEATURES = ("latitude", "longitude", "elevation_m")
READ_COLUMNS = tuple(dict.fromkeys(
    ("timestamp", "station_code", *CONTEXT_FEATURES, *FUTURE_FEATURES, *STATIC_FEATURES)
))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/protocol_v1.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-per-station", type=int, default=50_000)
    parser.add_argument("--val-per-station", type=int, default=10_000)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--architecture",
        choices=(
            "lstm", "tcn", "dlinear", "patchtst", "itransformer",
            "mamba", "night_mask_mamba", "physics_mamba",
        ),
        default="lstm",
        help="Sequence encoder to train; both use the identical data protocol and head.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int, deterministic_cudnn: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic_cudnn
    torch.backends.cudnn.benchmark = False


def fit_scaler(files: list[Path], columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(len(columns), dtype=np.float64)
    square = np.zeros(len(columns), dtype=np.float64)
    count = np.zeros(len(columns), dtype=np.int64)
    for path in files:
        values = pd.read_parquet(path, columns=list(columns)).to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0)
        total += safe.sum(axis=0)
        square += np.square(safe).sum(axis=0)
        count += finite.sum(axis=0)
    mean = total / np.maximum(count, 1)
    variance = square / np.maximum(count, 1) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass
class StationData:
    code: str
    timestamps: pd.DatetimeIndex
    context: np.ndarray
    future: np.ndarray
    static: np.ndarray
    static_raw: np.ndarray
    ghi: np.ndarray
    toa: np.ndarray
    valid_targets: np.ndarray
    origins: np.ndarray


def fit_static_scaler(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([
        pd.read_parquet(path, columns=list(STATIC_FEATURES)).iloc[0].to_numpy(dtype=np.float64)
        for path in files
    ])
    return values.mean(axis=0).astype(np.float32), np.maximum(
        values.std(axis=0), 1e-3
    ).astype(np.float32)


def load_station(
    path: Path,
    horizons: list[int],
    context_steps: int,
    minimum_fraction: float,
    context_mean: np.ndarray,
    context_std: np.ndarray,
    static_mean: np.ndarray,
    static_std: np.ndarray,
) -> StationData:
    frame = pd.read_parquet(path, columns=list(READ_COLUMNS))
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    raw_context = frame[list(CONTEXT_FEATURES)].to_numpy(dtype=np.float32)
    context_mask = np.isfinite(raw_context)
    normalized_context = np.where(
        context_mask, (raw_context - context_mean) / context_std, 0.0
    ).astype(np.float32)
    context = np.concatenate(
        [normalized_context, context_mask.astype(np.float32)], axis=1
    )
    future_positions = [CONTEXT_FEATURES.index(name) for name in FUTURE_FEATURES]
    future = normalized_context[:, future_positions]
    raw_static = frame[list(STATIC_FEATURES)].iloc[0].to_numpy(dtype=np.float32)
    static = ((raw_static - static_mean) / static_std).astype(np.float32)
    ghi = frame["ghi_wm2"].to_numpy(dtype=np.float32)
    toa = frame["toa_horizontal_wm2"].to_numpy(dtype=np.float32)
    valid_targets = np.zeros((len(frame), len(horizons)), dtype=bool)
    for column, horizon in enumerate(horizons):
        mask = forecast_origin_mask(
            timestamps, ghi, horizon, context_steps, minimum_fraction
        )
        valid_targets[: len(mask), column] = mask
    origins = np.flatnonzero(valid_targets.any(axis=1)).astype(np.int32)
    return StationData(
        path.stem, timestamps, context, future, static, raw_static, ghi, toa,
        valid_targets, origins,
    )


class WindowDataset(Dataset):
    def __init__(
        self,
        stations: list[StationData],
        horizons: list[int],
        context_steps: int,
        maximum_per_station: int | None,
        seed: int,
    ) -> None:
        self.stations = stations
        self.horizons = np.asarray(horizons, dtype=np.int32)
        self.context_steps = context_steps
        rng = np.random.default_rng(seed)
        pairs: list[np.ndarray] = []
        for station_index, station in enumerate(stations):
            origins = station.origins
            if maximum_per_station and len(origins) > maximum_per_station:
                origins = np.sort(rng.choice(origins, maximum_per_station, replace=False))
            station_column = np.full(len(origins), station_index, dtype=np.int32)
            pairs.append(np.column_stack([station_column, origins]))
        self.pairs = np.concatenate(pairs, axis=0)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        station_index, origin = self.pairs[index]
        station = self.stations[int(station_index)]
        origin = int(origin)
        start = origin - self.context_steps + 1
        context = station.context[start : origin + 1]
        target_indices = np.minimum(origin + self.horizons, len(station.ghi) - 1)
        future = station.future[target_indices]
        target_mask = station.valid_targets[origin].astype(np.float32)
        targets = np.where(
            target_mask.astype(bool), station.ghi[target_indices] / 1000.0, 0.0
        ).astype(np.float32)
        return context, future, station.static, targets, target_mask


class LSTMForecaster(nn.Module):
    def __init__(
        self, context_dim: int, future_dim: int, static_dim: int,
        horizons: int, hidden_size: int, layers: int, dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            context_dim, hidden_size, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        head_input = hidden_size + horizons * future_dim + static_dim
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, horizons), nn.Softplus(),
        )

    def forward(
        self, context: torch.Tensor, future: torch.Tensor, static: torch.Tensor
    ) -> torch.Tensor:
        _, (hidden, _) = self.encoder(context)
        combined = torch.cat([hidden[-1], future.flatten(1), static], dim=1)
        return self.head(combined)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.padding = padding
        self.conv = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def causal(self, convolution: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        output = convolution(values)
        return output[:, :, :-self.padding] if self.padding else output

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.dropout(self.activation(self.causal(self.conv, values)))
        return values + residual


class TCNForecaster(nn.Module):
    def __init__(
        self, context_dim: int, future_dim: int, static_dim: int,
        horizons: int, channels: int, dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(context_dim, channels, 1)
        self.encoder = nn.Sequential(*[
            CausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16, 32)
        ])
        head_input = channels + horizons * future_dim + static_dim
        self.head = nn.Sequential(
            nn.Linear(head_input, channels), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channels, horizons), nn.Softplus(),
        )

    def forward(
        self, context: torch.Tensor, future: torch.Tensor, static: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(context.transpose(1, 2)))[:, :, -1]
        combined = torch.cat([encoded, future.flatten(1), static], dim=1)
        return self.head(combined)


class DLinearForecaster(nn.Module):
    """Univariate decomposition-linear baseline with known-future covariates."""

    def __init__(
        self, context_steps: int, future_dim: int, static_dim: int,
        horizons: int, dropout: float, moving_average: int = 25,
    ) -> None:
        super().__init__()
        self.moving_average = moving_average
        self.seasonal = nn.Linear(context_steps, horizons)
        self.trend = nn.Linear(context_steps, horizons)
        nn.init.constant_(self.seasonal.weight, 1.0 / context_steps)
        nn.init.constant_(self.trend.weight, 1.0 / context_steps)
        nn.init.zeros_(self.seasonal.bias)
        nn.init.zeros_(self.trend.bias)
        known_dim = horizons * future_dim + static_dim
        self.known_future = nn.Sequential(
            nn.Linear(known_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, horizons),
        )

    def forward(
        self, context: torch.Tensor, future: torch.Tensor, static: torch.Tensor
    ) -> torch.Tensor:
        series = context[:, :, 0]
        padding = self.moving_average // 2
        padded = F.pad(series.unsqueeze(1), (padding, padding), mode="replicate")
        trend = F.avg_pool1d(padded, self.moving_average, stride=1).squeeze(1)
        seasonal = series - trend
        base = self.seasonal(seasonal) + self.trend(trend)
        known = self.known_future(torch.cat([future.flatten(1), static], dim=1))
        return F.softplus(base + known)


class PatchTSTForecaster(nn.Module):
    """Channel-independent GHI patch encoder with a direct irregular-horizon head."""

    def __init__(
        self, context_steps: int, future_dim: int, static_dim: int,
        horizons: int, d_model: int, layers: int, dropout: float,
        patch_length: int = 16, stride: int = 8,
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.mask_index = len(CONTEXT_FEATURES)
        self.patch_count = 1 + (context_steps - patch_length) // stride
        self.patch_projection = nn.Linear(patch_length * 2, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        known_dim = horizons * future_dim + static_dim
        self.head = nn.Sequential(
            nn.Linear(self.patch_count * d_model + known_dim, d_model),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, horizons),
            nn.Softplus(),
        )

    def forward(
        self, context: torch.Tensor, future: torch.Tensor, static: torch.Tensor
    ) -> torch.Tensor:
        values = context[:, :, 0].unfold(1, self.patch_length, self.stride)
        masks = context[:, :, self.mask_index].unfold(1, self.patch_length, self.stride)
        patches = self.patch_projection(torch.cat([values, masks], dim=-1))
        encoded = self.encoder(patches + self.position)
        known = torch.cat([future.flatten(1), static], dim=1)
        return self.head(torch.cat([encoded.flatten(1), known], dim=1))


class ITransformerForecaster(nn.Module):
    """Inverted transformer: variables are tokens and time is embedded per token."""

    def __init__(
        self, context_steps: int, context_dim: int, future_dim: int,
        static_dim: int, horizons: int, d_model: int, layers: int, dropout: float,
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(context_steps, d_model)
        self.variable_embedding = nn.Parameter(torch.zeros(1, context_dim, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        known_dim = horizons * future_dim + static_dim
        self.head = nn.Sequential(
            nn.Linear(d_model + known_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, horizons), nn.Softplus(),
        )

    def forward(
        self, context: torch.Tensor, future: torch.Tensor, static: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.token_projection(context.transpose(1, 2)) + self.variable_embedding
        encoded = self.encoder(tokens)
        known = torch.cat([future.flatten(1), static], dim=1)
        return self.head(torch.cat([encoded[:, 0], known], dim=1))


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as error:
            raise RuntimeError(
                "Mamba is not installed; run scripts/setup_runpod.sh on the GPU Pod."
            ) from error
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba2(
            d_model=d_model, d_state=64, d_conv=4, expand=2,
            headdim=64 if d_model % 64 == 0 else 32,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.dropout(self.mixer(self.norm(values)))


class MambaForecaster(nn.Module):
    """Vanilla Mamba-2 sequence encoder used as the state-space comparator."""

    def __init__(
        self, context_dim: int, future_dim: int, static_dim: int,
        horizons: int, d_model: int, layers: int, dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(context_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaResidualBlock(d_model, dropout) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        known_dim = horizons * future_dim + static_dim
        self.head = nn.Sequential(
            nn.Linear(d_model + known_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, horizons), nn.Softplus(),
        )

    def forward(
        self, context: torch.Tensor, future: torch.Tensor, static: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.input_projection(context)
        for block in self.blocks:
            encoded = block(encoded)
        summary = self.final_norm(encoded)[:, -1]
        known = torch.cat([future.flatten(1), static], dim=1)
        return self.head(torch.cat([summary, known], dim=1))


class PhysicsMambaForecaster(nn.Module):
    """Mamba-2 with a solar-envelope output parameterization."""

    def __init__(
        self, context_dim: int, future_dim: int, static_dim: int,
        horizons: int, d_model: int, layers: int, dropout: float,
        toa_mean: float, toa_std: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(context_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaResidualBlock(d_model, dropout) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        known_dim = horizons * future_dim + static_dim
        self.attenuation_head = nn.Sequential(
            nn.Linear(d_model + known_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, horizons),
        )
        self.toa_future_index = FUTURE_FEATURES.index("toa_horizontal_wm2")
        self.register_buffer("toa_mean", torch.tensor(float(toa_mean)))
        self.register_buffer("toa_std", torch.tensor(float(toa_std)))

    def forward(
        self, context: torch.Tensor, future: torch.Tensor, static: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.input_projection(context)
        for block in self.blocks:
            encoded = block(encoded)
        summary = self.final_norm(encoded)[:, -1]
        known = torch.cat([future.flatten(1), static], dim=1)
        # The network predicts atmospheric attenuation, while deterministic
        # extraterrestrial horizontal irradiance supplies the solar envelope.
        attenuation = 2.0 * torch.sigmoid(
            self.attenuation_head(torch.cat([summary, known], dim=1))
        )
        normalized_toa = future[:, :, self.toa_future_index]
        toa_wm2 = (normalized_toa * self.toa_std + self.toa_mean).clamp_min(0.0)
        return attenuation * (toa_wm2 / 1000.0)


class NightMaskedMambaForecaster(MambaForecaster):
    """Direct-output Mamba with the sole addition of an astronomical night mask."""

    def __init__(
        self, context_dim: int, future_dim: int, static_dim: int,
        horizons: int, d_model: int, layers: int, dropout: float,
        toa_mean: float, toa_std: float,
    ) -> None:
        super().__init__(
            context_dim, future_dim, static_dim, horizons, d_model, layers, dropout
        )
        self.toa_future_index = FUTURE_FEATURES.index("toa_horizontal_wm2")
        self.register_buffer("toa_mean", torch.tensor(float(toa_mean)))
        self.register_buffer("toa_std", torch.tensor(float(toa_std)))

    def forward(
        self, context: torch.Tensor, future: torch.Tensor, static: torch.Tensor
    ) -> torch.Tensor:
        direct_ghi = super().forward(context, future, static)
        normalized_toa = future[:, :, self.toa_future_index]
        toa_wm2 = normalized_toa * self.toa_std + self.toa_mean
        return direct_ghi * (toa_wm2 > 0.0).to(direct_ghi.dtype)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1.0)


def run_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device,
    optimizer: torch.optim.Optimizer | None, scaler: torch.amp.GradScaler | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    weight_sum = 0.0
    for context, future, static, target, mask in loader:
        context = context.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(context, future, static)
            loss = masked_mae(prediction, target, mask)
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        weight = float(mask.sum())
        loss_sum += float(loss.detach()) * weight
        weight_sum += weight
    return loss_sum / max(weight_sum, 1.0)


@torch.inference_mode()
def predict_dataset(
    model: nn.Module, dataset: WindowDataset, device: torch.device,
    batch_size: int, workers: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    predictions: list[np.ndarray] = []
    model.eval()
    for context, future, static, _, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                context.to(device, non_blocking=True),
                future.to(device, non_blocking=True),
                static.to(device, non_blocking=True),
            )
        predictions.append(output.float().cpu().numpy() * 1000.0)
    return np.concatenate(predictions, axis=0)


def evaluate(
    model: nn.Module, stations: list[StationData], horizons: list[int],
    context_steps: int, device: torch.device, batch_size: int, workers: int,
    model_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for station_index, station in enumerate(stations):
        dataset = WindowDataset([station], horizons, context_steps, None, seed=0)
        predictions = np.maximum(
            predict_dataset(model, dataset, device, batch_size, workers), 0.0
        )
        origins = dataset.pairs[:, 1]
        clear = ineichen_ghi(
            station.timestamps,
            float(station.static_raw[0]), float(station.static_raw[1]),
            float(station.static_raw[2]),
        )
        for column, horizon in enumerate(horizons):
            valid = station.valid_targets[origins, column]
            selected_origins = origins[valid]
            observed = station.ghi[selected_origins + horizon].astype(np.float64)
            predicted = predictions[valid, column].astype(np.float64)
            current = station.ghi[selected_origins].astype(np.float64)
            clear_index = np.clip(
                current / np.maximum(clear[selected_origins], CLEARSKY_FLOOR_WM2),
                0.0, CLEARSKY_INDEX_MAX,
            )
            smart = np.maximum(clear_index * clear[selected_origins + horizon], 0.0)
            regimes = {
                "all": np.ones(len(observed), dtype=bool),
                "daylight": station.toa[selected_origins + horizon] > 0.0,
                "daylight_to_daylight": (station.toa[selected_origins] > 0.0)
                & (station.toa[selected_origins + horizon] > 0.0),
            }
            for regime, mask in regimes.items():
                rows.append(metric_row(station.code, horizon, regime, model_name, observed[mask], predicted[mask]))
                rows.append(metric_row(station.code, horizon, regime, "smart_persistence", observed[mask], smart[mask]))
        print(f"Evaluated {station_index + 1}/{len(stations)}: {station.code}", flush=True)
    return rows


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic_cudnn=args.architecture == "lstm")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    horizons = [int(value) for value in config["forecast_horizon_steps"]]
    eligibility = config["evaluation_eligibility"]
    context_steps = int(eligibility["context_steps"])
    minimum_fraction = float(eligibility["minimum_context_ghi_fraction"])
    train_files = sorted((args.processed_dir / "train").glob("*.parquet"))
    val_files = sorted((args.processed_dir / "val").glob("*.parquet"))
    if len(train_files) != 33 or len(val_files) != 7:
        raise SystemExit("Expected exactly 33 train and 7 validation stations.")
    if args.smoke:
        train_files, val_files = train_files[:2], val_files[:1]
        args.train_per_station = min(args.train_per_station, 256)
        args.val_per_station = min(args.val_per_station, 128)
        args.epochs = min(args.epochs, 2)

    print("Fitting normalization on training stations only", flush=True)
    context_mean, context_std = fit_scaler(train_files, CONTEXT_FEATURES)
    static_mean, static_std = fit_static_scaler(train_files)
    print("Loading station arrays", flush=True)
    train_stations = [
        load_station(path, horizons, context_steps, minimum_fraction,
                     context_mean, context_std, static_mean, static_std)
        for path in train_files
    ]
    val_stations = [
        load_station(path, horizons, context_steps, minimum_fraction,
                     context_mean, context_std, static_mean, static_std)
        for path in val_files
    ]
    train_dataset = WindowDataset(
        train_stations, horizons, context_steps, args.train_per_station, args.seed
    )
    val_dataset = WindowDataset(
        val_stations, horizons, context_steps, args.val_per_station, args.seed + 1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.architecture == "lstm":
        model = LSTMForecaster(
            len(CONTEXT_FEATURES) * 2, len(FUTURE_FEATURES), len(STATIC_FEATURES),
            len(horizons), args.hidden_size, args.layers, args.dropout,
        ).to(device)
    elif args.architecture == "tcn":
        model = TCNForecaster(
            len(CONTEXT_FEATURES) * 2, len(FUTURE_FEATURES), len(STATIC_FEATURES),
            len(horizons), args.hidden_size, args.dropout,
        ).to(device)
    elif args.architecture == "dlinear":
        model = DLinearForecaster(
            context_steps, len(FUTURE_FEATURES), len(STATIC_FEATURES),
            len(horizons), args.dropout,
        ).to(device)
    elif args.architecture == "patchtst":
        model = PatchTSTForecaster(
            context_steps, len(FUTURE_FEATURES), len(STATIC_FEATURES),
            len(horizons), args.hidden_size, args.layers, args.dropout,
        ).to(device)
    elif args.architecture == "itransformer":
        model = ITransformerForecaster(
            context_steps, len(CONTEXT_FEATURES) * 2, len(FUTURE_FEATURES),
            len(STATIC_FEATURES), len(horizons), args.hidden_size,
            args.layers, args.dropout,
        ).to(device)
    elif args.architecture == "mamba":
        model = MambaForecaster(
            len(CONTEXT_FEATURES) * 2, len(FUTURE_FEATURES), len(STATIC_FEATURES),
            len(horizons), args.hidden_size, args.layers, args.dropout,
        ).to(device)
    elif args.architecture == "night_mask_mamba":
        toa_position = CONTEXT_FEATURES.index("toa_horizontal_wm2")
        model = NightMaskedMambaForecaster(
            len(CONTEXT_FEATURES) * 2, len(FUTURE_FEATURES), len(STATIC_FEATURES),
            len(horizons), args.hidden_size, args.layers, args.dropout,
            float(context_mean[toa_position]), float(context_std[toa_position]),
        ).to(device)
    else:
        toa_position = CONTEXT_FEATURES.index("toa_horizontal_wm2")
        model = PhysicsMambaForecaster(
            len(CONTEXT_FEATURES) * 2, len(FUTURE_FEATURES), len(STATIC_FEATURES),
            len(horizons), args.hidden_size, args.layers, args.dropout,
            float(context_mean[toa_position]), float(context_std[toa_position]),
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed)
    # Forking DataLoader workers after the CUDA-backed Mamba extension has been
    # initialized can deadlock or segfault. Keep every Mamba data path in the
    # main process; the other architectures retain configurable workers.
    data_workers = 0 if "mamba" in args.architecture else args.workers
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=data_workers, pin_memory=device.type == "cuda",
        persistent_workers=data_workers > 0, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=data_workers, pin_memory=device.type == "cuda",
        persistent_workers=data_workers > 0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, scaler)
        with torch.inference_mode():
            val_loss = run_epoch(model, val_loader, device, None, None)
        history.append({"epoch": epoch, "train_mae_scaled": train_loss, "val_mae_scaled": val_loss})
        print(f"Epoch {epoch:02d}: train={train_loss * 1000:.3f}, val={val_loss * 1000:.3f} W/m2", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.output_dir / "best_state.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping", flush=True)
                break
    model.load_state_dict(torch.load(args.output_dir / "best_state.pt", map_location=device, weights_only=True))
    evaluation_workers = data_workers
    rows = evaluate(
        model, val_stations, horizons, context_steps, device,
        args.batch_size, evaluation_workers, args.architecture,
    )
    station_metrics = pd.DataFrame(rows)
    aggregate = aggregate_metrics(station_metrics)
    station_metrics.to_csv(args.output_dir / "station_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    checkpoint = {
        "state_dict": model.state_dict(), "context_mean": context_mean,
        "context_std": context_std, "static_mean": static_mean,
        "static_std": static_std, "horizons": horizons,
        "context_features": CONTEXT_FEATURES, "future_features": FUTURE_FEATURES,
        "static_features": STATIC_FEATURES,
    }
    torch.save(checkpoint, args.output_dir / f"{args.architecture}_checkpoint.pt")
    metadata = {
        "protocol_id": config["protocol_id"], "model": args.architecture.upper(),
        "seed": args.seed, "device": str(device), "epochs_run": len(history),
        "best_validation_mae_wm2": best_loss * 1000.0,
        "train_stations": len(train_files), "validation_stations": len(val_files),
        "train_samples": len(train_dataset), "early_stop_samples": len(val_dataset),
        "evaluation_eligibility": eligibility, "test_evaluated": False,
        "deterministic_cudnn": args.architecture == "lstm",
        "data_workers": data_workers,
        "evaluation_workers": evaluation_workers,
        "hyperparameters": vars(args), "history": history,
    }
    metadata["hyperparameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata["hyperparameters"].items()
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(aggregate[aggregate["baseline"] == args.architecture].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
