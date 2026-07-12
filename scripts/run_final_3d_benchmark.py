#!/usr/bin/env python
"""Run a fixed 3D benchmark across model, causal filter, and prediction target.

Dimensions
----------
Models: LSTM baseline, HybridQNN1, HybridQNN2
Filters: vanilla, fixed Savitzky-Golay, fixed 10-day rolling average
Targets: binary next-bar direction, continuous next-bar close

This is a fixed benchmark, not a hyperparameter search. Filter configurations
are loaded from ``config/fixed_feature_filter_configs.json`` and were selected
previously without test leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam

from run_direction_classification_benchmark import (
    DirectionPartition,
    LSTMClassifier,
    classification_metrics,
    fit_classifier,
    fit_hqnn1_pretrain,
    predict_probabilities,
)
from train_hybridqnn1 import (
    ARTIFACT_FORMAT_VERSION,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    HybridQNN1Regressor,
    add_technical_indicators,
    artifact_manifest,
    canonicalize_ohlcv_columns,
    close_artifact_file_handlers,
    configure_logging,
    set_reproducible_seed,
    write_json,
)
from train_hybridqnn2 import HybridQNN2Regressor
from tune_feature_filters import FEATURE_CANDIDATES, apply_feature_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_FILTER_CONFIG_PATH = PROJECT_ROOT / "config" / "fixed_feature_filter_configs.json"
QNN1_SOURCE_PATH = Path(__file__).resolve().with_name("train_hybridqnn1.py")
QNN2_SOURCE_PATH = Path(__file__).resolve().with_name("train_hybridqnn2.py")


@dataclass
class FinalBenchmarkConfig:
    data_path: Path
    output_dir: Path
    fixed_filter_config_path: Path
    ticker: str
    profile: str
    lookback: int
    k_features: int
    n_qubits: int
    q_layers: int
    lstm_hidden_size: int
    lstm_layers: int
    lstm_dropout: float
    fusion_hidden_size: int
    holdout_ratio: float
    validation_ratio: float
    batch_size: int
    baseline_epochs: int
    hqnn1_pretrain_epochs: int
    hqnn1_quantum_epochs: int
    hqnn2_joint_epochs: int
    baseline_lr: float
    hqnn1_lr: float
    hqnn2_lr: float
    patience: int
    cpu_threads: int
    seed: int
    max_train_sequences: int
    max_validation_sequences: int
    max_test_sequences: int

    def validate(self) -> None:
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Data file was not found: {self.data_path}")
        if not self.fixed_filter_config_path.is_file():
            raise FileNotFoundError(
                "Fixed filter configurations were not found. Run "
                "scripts/tune_feature_filters.py first."
            )
        if self.k_features != self.n_qubits:
            raise ValueError("k_features must equal n_qubits.")
        if not 1 <= self.n_qubits <= 5:
            raise ValueError("n_qubits must be between 1 and 5 for this local benchmark.")
        if self.lookback < 1 or self.batch_size < 1:
            raise ValueError("lookback and batch_size must be positive.")
        if not 0.0 < self.holdout_ratio < 0.5:
            raise ValueError("holdout_ratio must be between 0 and 0.5.")
        if not 0.0 < self.validation_ratio < 0.5:
            raise ValueError("validation_ratio must be between 0 and 0.5.")

    def to_jsonable(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        payload["output_dir"] = str(self.output_dir)
        payload["fixed_filter_config_path"] = str(self.fixed_filter_config_path)
        return payload


@dataclass
class RegressionPartition:
    X: np.ndarray
    y_scaled: np.ndarray
    y_price: np.ndarray
    timestamps: List[str]


@dataclass
class RegressionData:
    train: RegressionPartition
    validation: RegressionPartition
    test: RegressionPartition
    selector: SelectKBest
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler
    selected_features: List[str]


@dataclass
class RegressionFitReport:
    best_epoch: int
    epochs_ran: int
    best_validation_loss: float
    history: List[Dict[str, float]]
    quantum_gradient_seen: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "best_validation_loss": self.best_validation_loss,
            "history": self.history,
            "quantum_gradient_seen": self.quantum_gradient_seen,
        }


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, config: FinalBenchmarkConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(config.lstm_hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(-1)


def _copy_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def ordered_batches(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for start in range(0, len(X), batch_size):
        yield X[start:start + batch_size], y[start:start + batch_size]


def _as_regression_partition(
    sequences: List[np.ndarray],
    targets_scaled: List[np.ndarray],
    targets_price: List[float],
    timestamps: List[str],
    lookback: int,
    n_features: int,
    name: str,
) -> RegressionPartition:
    if not sequences:
        raise ValueError(f"{name} has no valid regression sequences.")
    return RegressionPartition(
        X=np.asarray(sequences, dtype=np.float32).reshape(-1, lookback, n_features),
        y_scaled=np.asarray(targets_scaled, dtype=np.float32).reshape(-1),
        y_price=np.asarray(targets_price, dtype=np.float64).reshape(-1),
        timestamps=timestamps,
    )


def _cap_regression_partition(
    partition: RegressionPartition,
    max_sequences: int,
) -> RegressionPartition:
    if max_sequences <= 0 or len(partition.X) <= max_sequences:
        return partition
    start = len(partition.X) - max_sequences
    return RegressionPartition(
        X=partition.X[start:],
        y_scaled=partition.y_scaled[start:],
        y_price=partition.y_price[start:],
        timestamps=partition.timestamps[start:],
    )


def _cap_direction_partition(
    partition: DirectionPartition,
    max_sequences: int,
) -> DirectionPartition:
    if max_sequences <= 0 or len(partition.X) <= max_sequences:
        return partition
    start = len(partition.X) - max_sequences
    return DirectionPartition(
        X=partition.X[start:],
        y=partition.y[start:],
        timestamps=partition.timestamps[start:],
        prior_close=partition.prior_close[start:],
    )


def split_indices(row_count: int, config: FinalBenchmarkConfig) -> Tuple[int, int]:
    test_start = int(row_count * (1.0 - config.holdout_ratio))
    train_end = int(test_start * (1.0 - config.validation_ratio))
    if train_end <= config.lookback:
        raise ValueError("Training partition is too short for the selected lookback.")
    return train_end, test_start


def prepare_binary_data(
    data: pd.DataFrame,
    config: FinalBenchmarkConfig,
    filter_name: str,
    filter_config: Dict[str, Any],
) -> Tuple[DirectionPartition, DirectionPartition, DirectionPartition, Dict[str, Any]]:
    row_count = len(data)
    train_end, test_start = split_indices(row_count, config)
    candidate_features = [
        feature for feature in FEATURE_CANDIDATES if feature in data.columns
    ]
    implementation_name = "none" if filter_name == "vanilla" else filter_name
    filtered = apply_feature_filter(
        data,
        candidate_features,
        implementation_name,
        filter_config,
        train_end,
    )
    close = data["close"].to_numpy(dtype=np.float64)
    directions = (close[1:] > close[:-1]).astype(np.int64)
    features = filtered[candidate_features]

    selector = SelectKBest(score_func=f_classif, k=config.k_features)
    selector.fit(features.iloc[:train_end - 1], directions[:train_end - 1])
    selected_all = selector.transform(features)
    selected_features = features.columns[selector.get_support()].tolist()
    x_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    x_scaler.fit(selected_all[:train_end])
    X_scaled = x_scaler.transform(selected_all)

    buckets: Dict[str, Dict[str, List[Any]]] = {
        "train": {"X": [], "y": [], "timestamps": [], "prior": []},
        "validation": {"X": [], "y": [], "timestamps": [], "prior": []},
        "test": {"X": [], "y": [], "timestamps": [], "prior": []},
    }
    for target_index in range(config.lookback, row_count):
        bucket_name = (
            "train"
            if target_index < train_end
            else "validation"
            if target_index < test_start
            else "test"
        )
        bucket = buckets[bucket_name]
        bucket["X"].append(X_scaled[target_index - config.lookback:target_index])
        bucket["y"].append(int(close[target_index] > close[target_index - 1]))
        bucket["timestamps"].append(str(data.index[target_index]))
        bucket["prior"].append(float(close[target_index - 1]))

    def make_partition(name: str) -> DirectionPartition:
        values = buckets[name]
        return DirectionPartition(
            X=np.asarray(values["X"], dtype=np.float32).reshape(
                -1,
                config.lookback,
                config.k_features,
            ),
            y=np.asarray(values["y"], dtype=np.float32),
            timestamps=values["timestamps"],
            prior_close=np.asarray(values["prior"], dtype=np.float64),
        )

    metadata = {
        "selector": selector,
        "x_scaler": x_scaler,
        "selected_features": selected_features,
    }
    return (
        _cap_direction_partition(make_partition("train"), config.max_train_sequences),
        _cap_direction_partition(
            make_partition("validation"),
            config.max_validation_sequences,
        ),
        _cap_direction_partition(make_partition("test"), config.max_test_sequences),
        metadata,
    )


def prepare_regression_data(
    data: pd.DataFrame,
    config: FinalBenchmarkConfig,
    filter_name: str,
    filter_config: Dict[str, Any],
) -> RegressionData:
    row_count = len(data)
    train_end, test_start = split_indices(row_count, config)
    candidate_features = [
        feature for feature in FEATURE_CANDIDATES if feature in data.columns
    ]
    implementation_name = "none" if filter_name == "vanilla" else filter_name
    filtered = apply_feature_filter(
        data,
        candidate_features,
        implementation_name,
        filter_config,
        train_end,
    )
    close = data["close"].to_numpy(dtype=np.float64)
    features = filtered[candidate_features]

    # Associate feature row t with future close t+1 for leakage-safe selection.
    selector = SelectKBest(score_func=f_regression, k=config.k_features)
    selector.fit(features.iloc[:train_end - 1], close[1:train_end])
    selected_all = selector.transform(features)
    selected_features = features.columns[selector.get_support()].tolist()
    x_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    y_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    x_scaler.fit(selected_all[:train_end])
    y_scaler.fit(close[:train_end].reshape(-1, 1))
    X_scaled = x_scaler.transform(selected_all)
    y_scaled = y_scaler.transform(close.reshape(-1, 1))

    buckets: Dict[str, Dict[str, List[Any]]] = {
        "train": {"X": [], "y_scaled": [], "y_price": [], "timestamps": []},
        "validation": {"X": [], "y_scaled": [], "y_price": [], "timestamps": []},
        "test": {"X": [], "y_scaled": [], "y_price": [], "timestamps": []},
    }
    for target_index in range(config.lookback, row_count):
        bucket_name = (
            "train"
            if target_index < train_end
            else "validation"
            if target_index < test_start
            else "test"
        )
        bucket = buckets[bucket_name]
        bucket["X"].append(X_scaled[target_index - config.lookback:target_index])
        bucket["y_scaled"].append(y_scaled[target_index])
        bucket["y_price"].append(float(close[target_index]))
        bucket["timestamps"].append(str(data.index[target_index]))

    def make_partition(name: str) -> RegressionPartition:
        values = buckets[name]
        return _as_regression_partition(
            values["X"],
            values["y_scaled"],
            values["y_price"],
            values["timestamps"],
            config.lookback,
            config.k_features,
            name,
        )

    return RegressionData(
        train=_cap_regression_partition(make_partition("train"), config.max_train_sequences),
        validation=_cap_regression_partition(
            make_partition("validation"),
            config.max_validation_sequences,
        ),
        test=_cap_regression_partition(make_partition("test"), config.max_test_sequences),
        selector=selector,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        selected_features=selected_features,
    )


def regression_tensors(
    partition: RegressionPartition,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(partition.X, dtype=torch.float32, device=device),
        torch.as_tensor(partition.y_scaled, dtype=torch.float32, device=device),
    )


def regression_validation_loss(
    model: nn.Module,
    partition: RegressionPartition,
    device: torch.device,
) -> float:
    X, y = regression_tensors(partition, device)
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(X), y).cpu())


def fit_regression_model(
    model: nn.Module,
    train: RegressionPartition,
    validation: RegressionPartition,
    epochs: int,
    learning_rate: float,
    config: FinalBenchmarkConfig,
    device: torch.device,
    logger: logging.Logger,
    label: str,
) -> RegressionFitReport:
    X_train, y_train = regression_tensors(train, device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    quantum_gradient_seen = False
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for X_batch, y_batch in ordered_batches(X_train, y_train, config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(X_batch)
            loss = nn.functional.mse_loss(prediction, y_batch)
            loss.backward()
            quantum_weights = getattr(model, "q_weights", None)
            if quantum_weights is not None and quantum_weights.grad is not None:
                quantum_gradient_seen = quantum_gradient_seen or bool(
                    torch.isfinite(quantum_weights.grad).all().item()
                )
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)
        current_validation_loss = regression_validation_loss(model, validation, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": float(total_loss / total_count),
                "validation_mse": current_validation_loss,
            }
        )
        if current_validation_loss < best_loss - 1e-12:
            best_loss = current_validation_loss
            best_state = _copy_state_dict(model)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                logger.info("%s early stopped at epoch %s.", label, epoch)
                break

    if best_state is None:
        raise RuntimeError(f"{label} did not produce a valid state.")
    model.load_state_dict(best_state)
    return RegressionFitReport(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_validation_loss=best_loss,
        history=history,
        quantum_gradient_seen=quantum_gradient_seen,
    )


def fit_hqnn1_regression_pretrain(
    model: HybridQNN1Regressor,
    train: RegressionPartition,
    validation: RegressionPartition,
    config: FinalBenchmarkConfig,
    device: torch.device,
    logger: logging.Logger,
) -> RegressionFitReport:
    X_train, y_train = regression_tensors(train, device)
    optimizer = Adam(model.extractor.parameters(), lr=config.hqnn1_lr)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.hqnn1_pretrain_epochs + 1):
        model.extractor.train()
        total_loss = 0.0
        total_count = 0
        for X_batch, y_batch in ordered_batches(X_train, y_train, config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            _, prediction = model.extractor(X_batch)
            loss = nn.functional.mse_loss(prediction, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)

        X_validation, y_validation = regression_tensors(validation, device)
        model.extractor.eval()
        with torch.no_grad():
            _, validation_prediction = model.extractor(X_validation)
            current_validation_loss = float(
                nn.functional.mse_loss(validation_prediction, y_validation).cpu()
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": float(total_loss / total_count),
                "validation_mse": current_validation_loss,
            }
        )
        if current_validation_loss < best_loss - 1e-12:
            best_loss = current_validation_loss
            best_state = _copy_state_dict(model.extractor)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                logger.info("HybridQNN1 regression Stage 1 early stopped at epoch %s.", epoch)
                break

    if best_state is None:
        raise RuntimeError("HybridQNN1 regression Stage 1 did not produce a state.")
    model.extractor.load_state_dict(best_state)
    return RegressionFitReport(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_validation_loss=best_loss,
        history=history,
    )


def regression_predictions(
    model: nn.Module,
    partition: RegressionPartition,
    device: torch.device,
    y_scaler: MinMaxScaler,
) -> np.ndarray:
    X, _ = regression_tensors(partition, device)
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(X).cpu().numpy().reshape(-1, 1)
    return y_scaler.inverse_transform(prediction_scaled).reshape(-1)


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def run_binary_combo(
    model_name: str,
    train: DirectionPartition,
    validation: DirectionPartition,
    test: DirectionPartition,
    config: FinalBenchmarkConfig,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if model_name == "LSTM baseline":
        model: nn.Module = LSTMClassifier(config.k_features, config).to(device)
        started = time.perf_counter()
        fit = fit_classifier(
            model,
            train,
            validation,
            config.baseline_epochs,
            config.baseline_lr,
            config,
            device,
            logger,
            "Binary LSTM baseline",
        )
        details: Dict[str, Any] = fit.to_jsonable()
    elif model_name == "HybridQNN1":
        model = HybridQNN1Regressor(config.k_features, config).to(device)
        started = time.perf_counter()
        stage_1 = fit_hqnn1_pretrain(model, train, validation, config, device, logger)
        model.freeze_feature_extractor()
        stage_2 = fit_classifier(
            model,
            train,
            validation,
            config.hqnn1_quantum_epochs,
            config.hqnn1_lr,
            config,
            device,
            logger,
            "Binary HybridQNN1 Stage 2",
        )
        fit = stage_2
        details = {"stage_1": stage_1.to_jsonable(), "stage_2": stage_2.to_jsonable()}
    elif model_name == "HybridQNN2":
        model = HybridQNN2Regressor(config.k_features, config).to(device)
        started = time.perf_counter()
        fit = fit_classifier(
            model,
            train,
            validation,
            config.hqnn2_joint_epochs,
            config.hqnn2_lr,
            config,
            device,
            logger,
            "Binary HybridQNN2",
        )
        details = fit.to_jsonable()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    probabilities = predict_probabilities(model, test, device)
    return model, {
        "best_epoch": fit.best_epoch,
        "training": details,
        "training_seconds": time.perf_counter() - started,
        "test": classification_metrics(test.y.astype(int), probabilities),
        "test_probabilities": probabilities.tolist(),
    }


def run_continuous_combo(
    model_name: str,
    data: RegressionData,
    config: FinalBenchmarkConfig,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if model_name == "LSTM baseline":
        model: nn.Module = LSTMRegressor(config.k_features, config).to(device)
        started = time.perf_counter()
        fit = fit_regression_model(
            model,
            data.train,
            data.validation,
            config.baseline_epochs,
            config.baseline_lr,
            config,
            device,
            logger,
            "Continuous LSTM baseline",
        )
        details: Dict[str, Any] = fit.to_jsonable()
    elif model_name == "HybridQNN1":
        model = HybridQNN1Regressor(config.k_features, config).to(device)
        started = time.perf_counter()
        stage_1 = fit_hqnn1_regression_pretrain(
            model,
            data.train,
            data.validation,
            config,
            device,
            logger,
        )
        model.freeze_feature_extractor()
        stage_2 = fit_regression_model(
            model,
            data.train,
            data.validation,
            config.hqnn1_quantum_epochs,
            config.hqnn1_lr,
            config,
            device,
            logger,
            "Continuous HybridQNN1 Stage 2",
        )
        fit = stage_2
        details = {"stage_1": stage_1.to_jsonable(), "stage_2": stage_2.to_jsonable()}
    elif model_name == "HybridQNN2":
        model = HybridQNN2Regressor(config.k_features, config).to(device)
        started = time.perf_counter()
        fit = fit_regression_model(
            model,
            data.train,
            data.validation,
            config.hqnn2_joint_epochs,
            config.hqnn2_lr,
            config,
            device,
            logger,
            "Continuous HybridQNN2",
        )
        details = fit.to_jsonable()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    prediction = regression_predictions(model, data.test, device, data.y_scaler)
    return model, {
        "best_epoch": fit.best_epoch,
        "training": details,
        "training_seconds": time.perf_counter() - started,
        "test": regression_metrics(data.test.y_price, prediction),
        "test_predictions": prediction.tolist(),
    }


def fixed_filter_options(fixed_configs: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("vanilla", fixed_configs["no_filter"]),
        ("savgol", fixed_configs["savgol"]),
        ("rolling", fixed_configs["rolling"]),
    ]


def profile_defaults(profile: str) -> Dict[str, Any]:
    defaults: Dict[str, Dict[str, Any]] = {
        "final": {
            "lookback": 5,
            "k_features": 3,
            "n_qubits": 3,
            "q_layers": 1,
            "lstm_hidden_size": 16,
            "lstm_layers": 1,
            "lstm_dropout": 0.0,
            "fusion_hidden_size": 16,
            "holdout_ratio": 0.20,
            "validation_ratio": 0.20,
            "batch_size": 8,
            "baseline_epochs": 15,
            "hqnn1_pretrain_epochs": 8,
            "hqnn1_quantum_epochs": 8,
            "hqnn2_joint_epochs": 12,
            "baseline_lr": 1e-3,
            "hqnn1_lr": 5e-3,
            "hqnn2_lr": 1e-3,
            "patience": 3,
            "cpu_threads": 4,
            "max_train_sequences": 0,
            "max_validation_sequences": 0,
            "max_test_sequences": 0,
        },
        "smoke": {
            "lookback": 2,
            "k_features": 3,
            "n_qubits": 3,
            "q_layers": 1,
            "lstm_hidden_size": 6,
            "lstm_layers": 1,
            "lstm_dropout": 0.0,
            "fusion_hidden_size": 6,
            "holdout_ratio": 0.20,
            "validation_ratio": 0.25,
            "batch_size": 8,
            "baseline_epochs": 2,
            "hqnn1_pretrain_epochs": 2,
            "hqnn1_quantum_epochs": 2,
            "hqnn2_joint_epochs": 2,
            "baseline_lr": 1e-2,
            "hqnn1_lr": 1e-2,
            "hqnn2_lr": 1e-2,
            "patience": 1,
            "cpu_threads": 2,
            "max_train_sequences": 64,
            "max_validation_sequences": 16,
            "max_test_sequences": 16,
        },
    }
    return defaults[profile].copy()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed 3D model-filter-target benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=("final", "smoke"), default="final")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR.parent / "final_3d_benchmark",
    )
    parser.add_argument(
        "--fixed-filter-config-path",
        type=Path,
        default=FIXED_FILTER_CONFIG_PATH,
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FinalBenchmarkConfig:
    values = profile_defaults(args.profile)
    return FinalBenchmarkConfig(
        data_path=args.data_path.resolve(),
        output_dir=args.output_dir.resolve(),
        fixed_filter_config_path=args.fixed_filter_config_path.resolve(),
        ticker=args.ticker.upper(),
        profile=args.profile,
        seed=args.seed,
        **values,
    )


def package_artifact_bundle(
    final_directory: Path,
    temporary_directory: Path,
    config: FinalBenchmarkConfig,
    fixed_filters: Dict[str, Any],
    results: List[Dict[str, Any]],
    preprocessors: Dict[str, Any],
    models: Dict[str, nn.Module],
) -> Path:
    shutil.copy2(Path(__file__), temporary_directory / "run_final_3d_benchmark.py")
    shutil.copy2(QNN1_SOURCE_PATH, temporary_directory / "train_hybridqnn1.py")
    shutil.copy2(QNN2_SOURCE_PATH, temporary_directory / "train_hybridqnn2.py")
    shutil.copy2(
        Path(__file__).resolve().with_name("tune_feature_filters.py"),
        temporary_directory / "tune_feature_filters.py",
    )
    write_json(temporary_directory / "fixed_filter_configs.json", fixed_filters)
    write_json(temporary_directory / "final_3d_results.json", {"results": results})
    write_json(
        temporary_directory / "benchmark_config.json",
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "config": config.to_jsonable(),
            "grid": {
                "models": ["LSTM baseline", "HybridQNN1", "HybridQNN2"],
                "filters": ["vanilla", "savgol", "rolling"],
                "targets": ["binary_direction", "continuous_close"],
            },
            "runtime": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "torch": torch.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )
    for key, preprocessor in preprocessors.items():
        joblib.dump(preprocessor, temporary_directory / f"{key}_preprocessor.joblib")
    for key, model in models.items():
        torch.save(
            {
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "model_key": key,
                "state_dict": model.state_dict(),
            },
            temporary_directory / f"{key}_state.pt",
        )

    binary_rows = [result for result in results if result["target"] == "binary_direction"]
    continuous_rows = [
        result for result in results if result["target"] == "continuous_close"
    ]
    pd.DataFrame(
        [
            {
                "Model": row["model"],
                "Filter": row["filter"],
                "Accuracy": row["test"]["accuracy"],
                "BalancedAccuracy": row["test"]["balanced_accuracy"],
                "F1Up": row["test"]["f1_up"],
                "ROCAUC": row["test"]["roc_auc"],
                "TrainingSeconds": row["training_seconds"],
            }
            for row in binary_rows
        ]
    ).to_csv(temporary_directory / "binary_grid.csv", index=False)
    pd.DataFrame(
        [
            {
                "Model": row["model"],
                "Filter": row["filter"],
                "RMSE": row["test"]["rmse"],
                "MAE": row["test"]["mae"],
                "R2": row["test"]["r2"],
                "TrainingSeconds": row["training_seconds"],
            }
            for row in continuous_rows
        ]
    ).to_csv(temporary_directory / "continuous_grid.csv", index=False)
    write_json(
        temporary_directory / "manifest.json",
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "files_sha256": artifact_manifest(temporary_directory),
        },
    )
    if final_directory.exists():
        raise FileExistsError(f"Artifact directory already exists: {final_directory}")
    temporary_directory.replace(final_directory)
    return final_directory


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    config.validate()
    set_reproducible_seed(config.seed, config.cpu_threads)

    run_id = (
        f"{config.ticker.lower()}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    final_directory = config.output_dir / run_id
    temporary_directory = config.output_dir / f".{run_id}.tmp"
    temporary_directory.mkdir(parents=False, exist_ok=False)
    logger = configure_logging(temporary_directory / "training.log")
    logger.info("Starting fixed %s 3D benchmark.", config.profile)
    logger.info("Configuration: %s", json.dumps(config.to_jsonable(), default=str))

    try:
        raw = pd.read_parquet(config.data_path).dropna()
        data = add_technical_indicators(canonicalize_ohlcv_columns(raw))
        fixed_filters = json.loads(config.fixed_filter_config_path.read_text())
        device = torch.device("cpu")
        models = ["LSTM baseline", "HybridQNN1", "HybridQNN2"]
        results: List[Dict[str, Any]] = []
        saved_models: Dict[str, nn.Module] = {}
        preprocessors: Dict[str, Any] = {}

        for filter_index, (filter_name, filter_config) in enumerate(
            fixed_filter_options(fixed_filters)
        ):
            logger.info("Preparing %s filter datasets.", filter_name)
            binary_train, binary_validation, binary_test, binary_metadata = (
                prepare_binary_data(data, config, filter_name, filter_config)
            )
            regression_data = prepare_regression_data(
                data,
                config,
                filter_name,
                filter_config,
            )
            preprocessors[f"binary_{filter_name}"] = binary_metadata
            preprocessors[f"continuous_{filter_name}"] = {
                "selector": regression_data.selector,
                "x_scaler": regression_data.x_scaler,
                "y_scaler": regression_data.y_scaler,
                "selected_features": regression_data.selected_features,
                "lookback": config.lookback,
            }

            for model_index, model_name in enumerate(models):
                combo_seed = config.seed + (filter_index * len(models)) + model_index
                set_reproducible_seed(combo_seed, config.cpu_threads)
                binary_model, binary_result = run_binary_combo(
                    model_name,
                    binary_train,
                    binary_validation,
                    binary_test,
                    config,
                    device,
                    logger,
                )
                binary_key = (
                    f"binary_{filter_name}_{model_name.lower().replace(' ', '_')}"
                )
                saved_models[binary_key] = binary_model
                results.append(
                    {
                        "target": "binary_direction",
                        "filter": filter_name,
                        "filter_config": filter_config,
                        "model": model_name,
                        **binary_result,
                    }
                )
                logger.info(
                    "Binary | %s | %s: AUC %s | balanced accuracy %.4f | %.2fs",
                    filter_name,
                    model_name,
                    (
                        f"{binary_result['test']['roc_auc']:.4f}"
                        if binary_result["test"]["roc_auc"] is not None
                        else "n/a"
                    ),
                    binary_result["test"]["balanced_accuracy"],
                    binary_result["training_seconds"],
                )

                set_reproducible_seed(combo_seed, config.cpu_threads)
                continuous_model, continuous_result = run_continuous_combo(
                    model_name,
                    regression_data,
                    config,
                    device,
                    logger,
                )
                continuous_key = (
                    f"continuous_{filter_name}_{model_name.lower().replace(' ', '_')}"
                )
                saved_models[continuous_key] = continuous_model
                results.append(
                    {
                        "target": "continuous_close",
                        "filter": filter_name,
                        "filter_config": filter_config,
                        "model": model_name,
                        **continuous_result,
                    }
                )
                logger.info(
                    "Continuous | %s | %s: RMSE %.4f | R2 %.4f | %.2fs",
                    filter_name,
                    model_name,
                    continuous_result["test"]["rmse"],
                    continuous_result["test"]["r2"],
                    continuous_result["training_seconds"],
                )

        close_artifact_file_handlers(logger)
        artifact_path = package_artifact_bundle(
            final_directory,
            temporary_directory,
            config,
            fixed_filters,
            results,
            preprocessors,
            saved_models,
        )
        logger.info("Final 3D benchmark artifact saved to %s", artifact_path)
        print(f"Final 3D benchmark artifact saved: {artifact_path}")
        return 0
    except Exception:
        logger.exception(
            "Final 3D benchmark failed. Temporary directory retained: %s",
            temporary_directory,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
