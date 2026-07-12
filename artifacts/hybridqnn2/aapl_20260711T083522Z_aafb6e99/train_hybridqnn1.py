#!/usr/bin/env python
"""Train and package a reproducible HybridQNN1 model bundle.

The script intentionally treats HybridQNN1 as an experimental model entering a
staging workflow, not as an automatically production-ready forecasting system.
It enforces chronological splits, keeps a final holdout set untouched until
evaluation, records a naive baseline, and writes a self-contained artifact
bundle for audit and later inference work.

Examples
--------
Meaningful local staging run (the default profile):
    conda run -n env_qubit python scripts/train_hybridqnn1.py

Fast pipeline verification:
    conda run -n env_qubit python scripts/train_hybridqnn1.py --profile smoke

Use a different input file and artifact root:
    conda run -n env_qubit python scripts/train_hybridqnn1.py ^
        --data-path E:/fyp_qnn/data/yfinance/MSFT.parquet ^
        --ticker MSFT --output-dir artifacts/hybridqnn1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
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
import pennylane as qml
import sklearn
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = Path(r"E:\fyp_qnn\data\yfinance\AAPL.parquet")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "hybridqnn1"
ARTIFACT_FORMAT_VERSION = 1
FEATURE_CANDIDATES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi_14",
    "macd",
    "macd_signal",
    "adx_14",
)


@dataclass
class TrainingConfig:
    """All hyperparameters and release-gate settings for one training run."""

    data_path: Path
    output_dir: Path
    ticker: str
    profile: str
    lookback: int
    k_features: int
    n_qubits: int
    q_layers: int
    lstm_hidden_size: int
    lstm_layers: int
    lstm_dropout: float
    holdout_ratio: float
    validation_ratio: float
    batch_size: int
    pretrain_epochs: int
    quantum_epochs: int
    pretrain_lr: float
    quantum_lr: float
    patience: int
    max_train_sequences: int
    max_validation_sequences: int
    max_test_sequences: int
    cpu_threads: int
    seed: int
    refit_on_train_validation: bool
    require_beat_naive: bool
    max_test_rmse: Optional[float]
    allow_large_circuits: bool

    def validate(self) -> None:
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Data file was not found: {self.data_path}")
        if self.k_features != self.n_qubits:
            raise ValueError("k_features must equal n_qubits for one feature per qubit.")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive.")
        if self.n_qubits > 5 and not self.allow_large_circuits:
            raise ValueError(
                "Refusing to run more than five exact-simulator qubits. "
                "Pass --allow-large-circuits only after measuring resource use."
            )
        if not 0.0 < self.holdout_ratio < 0.5:
            raise ValueError("holdout_ratio must be between 0 and 0.5.")
        if not 0.0 < self.validation_ratio < 0.5:
            raise ValueError("validation_ratio must be between 0 and 0.5.")
        if self.lookback < 1:
            raise ValueError("lookback must be at least one.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.pretrain_epochs < 1 or self.quantum_epochs < 1:
            raise ValueError("Both training stages need at least one epoch.")
        if self.patience < 1:
            raise ValueError("patience must be at least one.")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be at least one.")

    def to_jsonable(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass
class SequencePartition:
    """A chronological sequence partition and its naive last-close baseline."""

    X: np.ndarray
    y_scaled: np.ndarray
    timestamps: List[str]
    naive_close: np.ndarray


@dataclass
class PreparedData:
    """Preprocessed partitions plus all state required for artifact inference."""

    train: SequencePartition
    validation: SequencePartition
    test: SequencePartition
    selector: SelectKBest
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler
    selected_features: List[str]
    candidate_features: List[str]
    raw_rows: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    date_start: str
    date_end: str


@dataclass
class StageFitReport:
    """Chronological validation outcome for one training stage."""

    best_epoch: int
    epochs_ran: int
    best_validation_rmse: float
    history: List[Dict[str, float]]
    quantum_gradient_seen: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "best_validation_rmse": self.best_validation_rmse,
            "history": self.history,
            "quantum_gradient_seen": self.quantum_gradient_seen,
        }


class LSTMFeatureExtractor(nn.Module):
    """Stage 1: learn temporal features and project them into the qubit range."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        n_qubits: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.qubit_projection = nn.Linear(hidden_size, n_qubits)
        self.pretrain_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.lstm(x)
        final_hidden = hidden[-1]
        qubit_features = torch.tanh(self.qubit_projection(final_hidden))
        baseline_prediction = self.pretrain_head(final_hidden).squeeze(-1)
        return qubit_features, baseline_prediction


class HybridQNN1Regressor(nn.Module):
    """Sequential LSTM -> angle encoder -> shallow variational QNN regressor.

    The article does not specify the `Ppr` blocks drawn in Fig. 6. This
    implementation therefore follows the textual specification: RY/RZ angle
    encoding, trainable rotations, nearest-neighbour CNOT/CZ entanglement, and
    Pauli-Z expectation values. It must be described as Fig. 6-inspired rather
    than an exact gate-level reproduction.
    """

    def __init__(self, n_features: int, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.extractor = LSTMFeatureExtractor(
            n_features=n_features,
            hidden_size=config.lstm_hidden_size,
            n_qubits=config.n_qubits,
            n_layers=config.lstm_layers,
            dropout=config.lstm_dropout,
        )
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(config.q_layers, config.n_qubits, 2)
        )
        self.output_head = nn.Linear(config.n_qubits, 1)
        self.quantum_device = qml.device(
            "default.qubit",
            wires=config.n_qubits,
            shots=None,
        )

        @qml.qnode(self.quantum_device, interface="torch", diff_method="backprop")
        def qnn_circuit(features: torch.Tensor, weights: torch.Tensor):
            clipped = qml.math.clip(features, -0.999999, 0.999999)
            for wire in range(config.n_qubits):
                qml.RY(qml.math.arcsin(clipped[wire]), wires=wire)
                qml.RZ(qml.math.arccos(clipped[wire]), wires=wire)

            for layer in range(config.q_layers):
                for wire in range(config.n_qubits):
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                for wire in range(config.n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                    qml.CZ(wires=[wire, wire + 1])

            return [qml.expval(qml.PauliZ(wire)) for wire in range(config.n_qubits)]

        self.qnode = qnn_circuit

    def freeze_feature_extractor(self) -> None:
        """Make Stage 2 sequential, preserving the HybridQNN1 distinction."""

        for parameter in self.extractor.parameters():
            parameter.requires_grad_(False)
        self.extractor.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qubit_features, _ = self.extractor(x)
        quantum_outputs = []
        for sample in qubit_features:
            q_result = self.qnode(sample, self.q_weights)
            quantum_outputs.append(torch.stack(list(q_result)))

        q_features = torch.stack(quantum_outputs).to(dtype=qubit_features.dtype)
        return self.output_head(q_features).squeeze(-1)


def configure_logging(log_path: Path) -> logging.Logger:
    """Log to both console and the artifact directory."""

    logger = logging.getLogger("train_hybridqnn1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def close_artifact_file_handlers(logger: logging.Logger) -> None:
    """Release Windows file locks before atomically moving the artifact folder."""

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


def set_reproducible_seed(seed: int, cpu_threads: int) -> None:
    """Set random seeds and cap CPU use for the local exact simulator."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)


def canonicalize_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize case variants such as `Close` and `Volume` to lower case."""

    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [str(column[0]) for column in frame.columns]

    renamed: Dict[Any, str] = {}
    for column in frame.columns:
        normalized = str(column).strip().lower()
        if normalized in {"open", "high", "low", "close", "volume"}:
            renamed[column] = normalized
    output = frame.rename(columns=renamed).copy()

    required = {"open", "high", "low", "close"}
    missing = required.difference(output.columns)
    if missing:
        raise ValueError(
            "The input parquet file must provide OHLC columns. "
            f"Missing after normalization: {sorted(missing)}"
        )

    if not output.index.is_monotonic_increasing:
        output = output.sort_index()
    return output


def add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the RSI, MACD, and ADX indicators used by the notebook pipeline."""

    output = frame.copy()
    delta = output["close"].diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)
    average_gain = gain.rolling(14).mean()
    average_loss = loss.rolling(14).mean()
    rs = average_gain / (average_loss + 1e-12)
    output["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    ema_12 = output["close"].ewm(span=12, adjust=False).mean()
    ema_26 = output["close"].ewm(span=26, adjust=False).mean()
    output["macd"] = ema_12 - ema_26
    output["macd_signal"] = output["macd"].ewm(span=9, adjust=False).mean()

    high_low = output["high"] - output["low"]
    high_close = (output["high"] - output["close"].shift(1)).abs()
    low_close = (output["low"] - output["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    up_move = output["high"].diff()
    down_move = -output["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    atr_14 = true_range.rolling(14).mean()
    plus_di = 100.0 * (
        pd.Series(plus_dm, index=output.index).rolling(14).mean() / (atr_14 + 1e-12)
    )
    minus_di = 100.0 * (
        pd.Series(minus_dm, index=output.index).rolling(14).mean() / (atr_14 + 1e-12)
    )
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    output["adx_14"] = dx.rolling(14).mean()
    return output.dropna()


def _as_partition(
    rows: List[np.ndarray],
    targets: List[np.ndarray],
    timestamps: List[str],
    naive_close: List[float],
    lookback: int,
    n_features: int,
    partition_name: str,
) -> SequencePartition:
    if not rows:
        raise ValueError(
            f"{partition_name} has no valid sequences. "
            "Provide more data or reduce lookback/split ratios."
        )
    return SequencePartition(
        X=np.asarray(rows, dtype=np.float32).reshape(-1, lookback, n_features),
        y_scaled=np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        timestamps=timestamps,
        naive_close=np.asarray(naive_close, dtype=np.float64),
    )


def _cap_recent_partition(
    partition: SequencePartition,
    max_sequences: int,
) -> SequencePartition:
    """Keep the newest bounded window when using the smoke profile."""

    if max_sequences <= 0 or len(partition.X) <= max_sequences:
        return partition
    start = len(partition.X) - max_sequences
    return SequencePartition(
        X=partition.X[start:],
        y_scaled=partition.y_scaled[start:],
        timestamps=partition.timestamps[start:],
        naive_close=partition.naive_close[start:],
    )


def prepare_data(config: TrainingConfig, logger: logging.Logger) -> PreparedData:
    """Build leakage-safe train/validation/test sequences from the parquet data."""

    raw = pd.read_parquet(config.data_path).dropna()
    data = add_technical_indicators(canonicalize_ohlcv_columns(raw))
    if len(data) <= config.lookback + 20:
        raise ValueError("Not enough post-indicator rows to construct safe sequences.")

    candidate_features = [
        column for column in FEATURE_CANDIDATES if column in data.columns
    ]
    if len(candidate_features) < config.k_features:
        raise ValueError(
            f"Requested {config.k_features} features but only "
            f"{len(candidate_features)} candidates are available: {candidate_features}"
        )

    row_count = len(data)
    test_start = int(row_count * (1.0 - config.holdout_ratio))
    train_end = int(test_start * (1.0 - config.validation_ratio))
    validation_end = test_start
    if train_end <= config.lookback:
        raise ValueError("Training partition is too short for the requested lookback.")

    features = data[candidate_features]
    target = data["close"].to_numpy(dtype=np.float64).reshape(-1, 1)
    selector = SelectKBest(score_func=f_regression, k=config.k_features)
    selected_train = selector.fit_transform(
        features.iloc[:train_end],
        target[:train_end].ravel(),
    )
    selected_all = selector.transform(features)
    selected_features = features.columns[selector.get_support()].tolist()

    x_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    y_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    x_scaler.fit(selected_train)
    y_scaler.fit(target[:train_end])
    X_scaled = x_scaler.transform(selected_all)
    y_scaled = y_scaler.transform(target)

    buckets: Dict[str, Dict[str, List[Any]]] = {
        "train": {"X": [], "y": [], "timestamps": [], "naive": []},
        "validation": {"X": [], "y": [], "timestamps": [], "naive": []},
        "test": {"X": [], "y": [], "timestamps": [], "naive": []},
    }
    close_values = data["close"].to_numpy(dtype=np.float64)
    index_values = data.index
    for target_position in range(config.lookback, row_count):
        if target_position < train_end:
            bucket_name = "train"
        elif target_position < validation_end:
            bucket_name = "validation"
        else:
            bucket_name = "test"

        bucket = buckets[bucket_name]
        bucket["X"].append(X_scaled[target_position - config.lookback:target_position])
        bucket["y"].append(y_scaled[target_position])
        bucket["timestamps"].append(str(index_values[target_position]))
        bucket["naive"].append(float(close_values[target_position - 1]))

    train = _cap_recent_partition(
        _as_partition(
            buckets["train"]["X"],
            buckets["train"]["y"],
            buckets["train"]["timestamps"],
            buckets["train"]["naive"],
            config.lookback,
            config.k_features,
            "train",
        ),
        config.max_train_sequences,
    )
    validation = _cap_recent_partition(
        _as_partition(
            buckets["validation"]["X"],
            buckets["validation"]["y"],
            buckets["validation"]["timestamps"],
            buckets["validation"]["naive"],
            config.lookback,
            config.k_features,
            "validation",
        ),
        config.max_validation_sequences,
    )
    test = _cap_recent_partition(
        _as_partition(
            buckets["test"]["X"],
            buckets["test"]["y"],
            buckets["test"]["timestamps"],
            buckets["test"]["naive"],
            config.lookback,
            config.k_features,
            "test",
        ),
        config.max_test_sequences,
    )

    logger.info(
        "Prepared %s rows into %s/%s/%s train/validation/test sequences.",
        row_count,
        len(train.X),
        len(validation.X),
        len(test.X),
    )
    logger.info("Selected features: %s", selected_features)
    return PreparedData(
        train=train,
        validation=validation,
        test=test,
        selector=selector,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        selected_features=selected_features,
        candidate_features=candidate_features,
        raw_rows=row_count,
        train_row_count=train_end,
        validation_row_count=validation_end - train_end,
        test_row_count=row_count - validation_end,
        date_start=str(index_values[0]),
        date_end=str(index_values[-1]),
    )


def ordered_batches(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield ordered mini-batches without randomization across time."""

    for start in range(0, len(X), batch_size):
        yield X[start:start + batch_size], y[start:start + batch_size]


def _copy_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _to_tensors(
    partition: SequencePartition,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(partition.X, dtype=torch.float32, device=device),
        torch.as_tensor(partition.y_scaled.reshape(-1), dtype=torch.float32, device=device),
    )


def _validation_rmse(
    model: HybridQNN1Regressor,
    partition: SequencePartition,
    device: torch.device,
    use_extractor_head: bool,
) -> float:
    X, y = _to_tensors(partition, device)
    model.eval()
    with torch.no_grad():
        if use_extractor_head:
            _, prediction = model.extractor(X)
        else:
            prediction = model(X)
        return float(torch.sqrt(nn.functional.mse_loss(prediction, y)).cpu())


def fit_stage1_with_validation(
    model: HybridQNN1Regressor,
    train: SequencePartition,
    validation: SequencePartition,
    config: TrainingConfig,
    device: torch.device,
    logger: logging.Logger,
) -> StageFitReport:
    """Pretrain the LSTM forecast head and restore its best validation state."""

    X_train, y_train = _to_tensors(train, device)
    optimizer = Adam(model.extractor.parameters(), lr=config.pretrain_lr)
    loss_fn = nn.MSELoss()
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.pretrain_epochs + 1):
        model.extractor.train()
        total_loss = 0.0
        total_count = 0
        for X_batch, y_batch in ordered_batches(X_train, y_train, config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            _, prediction = model.extractor(X_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)

        train_rmse = float(np.sqrt(total_loss / total_count))
        validation_rmse = _validation_rmse(
            model,
            validation,
            device,
            use_extractor_head=True,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_rmse": train_rmse,
                "validation_rmse": validation_rmse,
            }
        )

        validation_loss = validation_rmse ** 2
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_state = _copy_state_dict(model.extractor)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                logger.info("Stage 1 early stopped at epoch %s.", epoch)
                break

    if best_state is None:
        raise RuntimeError("Stage 1 did not produce a valid state.")
    model.extractor.load_state_dict(best_state)
    logger.info(
        "Stage 1 selected epoch %s with validation RMSE %.6f.",
        best_epoch,
        np.sqrt(best_loss),
    )
    return StageFitReport(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_validation_rmse=float(np.sqrt(best_loss)),
        history=history,
    )


def fit_stage2_with_validation(
    model: HybridQNN1Regressor,
    train: SequencePartition,
    validation: SequencePartition,
    config: TrainingConfig,
    device: torch.device,
    logger: logging.Logger,
) -> StageFitReport:
    """Train only quantum/readout parameters and restore the best validation state."""

    X_train, y_train = _to_tensors(train, device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable Stage 2 parameters remain.")

    optimizer = Adam(trainable_parameters, lr=config.quantum_lr)
    loss_fn = nn.MSELoss()
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    quantum_gradient_seen = False
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.quantum_epochs + 1):
        model.train()
        model.extractor.eval()
        total_loss = 0.0
        total_count = 0
        for X_batch, y_batch in ordered_batches(X_train, y_train, config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(X_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()

            if model.q_weights.grad is not None:
                quantum_gradient_seen = quantum_gradient_seen or bool(
                    torch.isfinite(model.q_weights.grad).all().item()
                )
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)

        train_rmse = float(np.sqrt(total_loss / total_count))
        validation_rmse = _validation_rmse(
            model,
            validation,
            device,
            use_extractor_head=False,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_rmse": train_rmse,
                "validation_rmse": validation_rmse,
            }
        )

        validation_loss = validation_rmse ** 2
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_state = _copy_state_dict(model)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                logger.info("Stage 2 early stopped at epoch %s.", epoch)
                break

    if best_state is None:
        raise RuntimeError("Stage 2 did not produce a valid state.")
    if not quantum_gradient_seen:
        raise RuntimeError("No finite gradient reached the quantum weights.")
    model.load_state_dict(best_state)
    logger.info(
        "Stage 2 selected epoch %s with validation RMSE %.6f.",
        best_epoch,
        np.sqrt(best_loss),
    )
    return StageFitReport(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_validation_rmse=float(np.sqrt(best_loss)),
        history=history,
        quantum_gradient_seen=quantum_gradient_seen,
    )


def fit_stage1_fixed_epochs(
    model: HybridQNN1Regressor,
    partition: SequencePartition,
    config: TrainingConfig,
    device: torch.device,
    epochs: int,
) -> List[float]:
    """Refit Stage 1 on train+validation for the epoch count selected above."""

    X_train, y_train = _to_tensors(partition, device)
    optimizer = Adam(model.extractor.parameters(), lr=config.pretrain_lr)
    loss_fn = nn.MSELoss()
    history: List[float] = []
    for _ in range(max(1, epochs)):
        model.extractor.train()
        total_loss = 0.0
        total_count = 0
        for X_batch, y_batch in ordered_batches(X_train, y_train, config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            _, prediction = model.extractor(X_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)
        history.append(float(np.sqrt(total_loss / total_count)))
    return history


def fit_stage2_fixed_epochs(
    model: HybridQNN1Regressor,
    partition: SequencePartition,
    config: TrainingConfig,
    device: torch.device,
    epochs: int,
) -> Tuple[List[float], bool]:
    """Refit Stage 2 on train+validation for the selected epoch count."""

    X_train, y_train = _to_tensors(partition, device)
    optimizer = Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.quantum_lr,
    )
    loss_fn = nn.MSELoss()
    quantum_gradient_seen = False
    history: List[float] = []
    for _ in range(max(1, epochs)):
        model.train()
        model.extractor.eval()
        total_loss = 0.0
        total_count = 0
        for X_batch, y_batch in ordered_batches(X_train, y_train, config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(X_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            if model.q_weights.grad is not None:
                quantum_gradient_seen = quantum_gradient_seen or bool(
                    torch.isfinite(model.q_weights.grad).all().item()
                )
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)
        history.append(float(np.sqrt(total_loss / total_count)))
    return history, quantum_gradient_seen


def concatenate_partitions(
    first: SequencePartition,
    second: SequencePartition,
) -> SequencePartition:
    """Combine adjacent chronological train and validation partitions."""

    return SequencePartition(
        X=np.concatenate([first.X, second.X], axis=0),
        y_scaled=np.concatenate([first.y_scaled, second.y_scaled], axis=0),
        timestamps=first.timestamps + second.timestamps,
        naive_close=np.concatenate([first.naive_close, second.naive_close], axis=0),
    )


def predict_scaled(
    model: HybridQNN1Regressor,
    partition: SequencePartition,
    device: torch.device,
) -> np.ndarray:
    X, _ = _to_tensors(partition, device)
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy().reshape(-1, 1)


def calculate_metrics(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_scaler: MinMaxScaler,
) -> Dict[str, float]:
    """Return scaled and original-price metrics for reporting and release gates."""

    y_true_scaled = y_true_scaled.reshape(-1, 1)
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_true_price = y_scaler.inverse_transform(y_true_scaled)
    y_pred_price = y_scaler.inverse_transform(y_pred_scaled)
    return {
        "scaled_rmse": float(np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))),
        "scaled_mae": float(mean_absolute_error(y_true_scaled, y_pred_scaled)),
        "price_rmse": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
        "price_mae": float(mean_absolute_error(y_true_price, y_pred_price)),
        "price_r2": float(r2_score(y_true_price, y_pred_price)),
    }


def calculate_naive_price_metrics(
    y_true_scaled: np.ndarray,
    naive_close: np.ndarray,
    y_scaler: MinMaxScaler,
) -> Dict[str, float]:
    """Evaluate a last-close baseline on the same holdout rows."""

    y_true_price = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(-1)
    return {
        "price_rmse": float(np.sqrt(mean_squared_error(y_true_price, naive_close))),
        "price_mae": float(mean_absolute_error(y_true_price, naive_close)),
        "price_r2": float(r2_score(y_true_price, naive_close)),
    }


def release_gate(
    test_metrics: Dict[str, float],
    naive_metrics: Dict[str, float],
    config: TrainingConfig,
) -> Dict[str, Any]:
    """Require the model to beat a naive baseline before staging promotion."""

    rejection_reasons: List[str] = []
    if (
        config.require_beat_naive
        and test_metrics["price_rmse"] >= naive_metrics["price_rmse"]
    ):
        rejection_reasons.append(
            "Model price RMSE did not beat the last-close naive baseline."
        )
    if (
        config.max_test_rmse is not None
        and test_metrics["price_rmse"] > config.max_test_rmse
    ):
        rejection_reasons.append(
            f"Model price RMSE exceeded the configured limit ({config.max_test_rmse})."
        )
    return {
        "status": "accepted" if not rejection_reasons else "rejected",
        "reasons": rejection_reasons,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def render_model_card(
    config: TrainingConfig,
    prepared: PreparedData,
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    naive_metrics: Dict[str, float],
    gate: Dict[str, Any],
) -> str:
    """Create human-readable artifact documentation for review."""

    gate_reasons = "\n".join(
        f"- {reason}" for reason in gate["reasons"]
    ) or "- All configured staging gates passed."
    return f"""# HybridQNN1 model card

## Status
`{gate["status"]}`

## Intended use
One-step close-price regression for research/staging workflows. This bundle is
not a trading recommendation and must not be deployed without independent
backtesting, monitoring, governance review, and an approved promotion gate.

## Architecture
Sequential LSTM feature extraction followed by bounded RY/RZ angle encoding,
a shallow RY/RZ + nearest-neighbour CNOT/CZ QNN, and a linear readout.
The source paper does not define Fig. 6's `Ppr` blocks, so this is a documented
Fig. 6-inspired approximation rather than exact gate-level reproduction.

## Data
- Ticker label: `{config.ticker}`
- Source: `{config.data_path}`
- Data range: `{prepared.date_start}` to `{prepared.date_end}`
- Selected features: `{", ".join(prepared.selected_features)}`
- Lookback: `{config.lookback}`

## Evaluation
- Validation price RMSE: `{validation_metrics["price_rmse"]:.6f}`
- Holdout price RMSE: `{test_metrics["price_rmse"]:.6f}`
- Last-close holdout price RMSE: `{naive_metrics["price_rmse"]:.6f}`
- Holdout price R²: `{test_metrics["price_r2"]:.6f}`

## Staging gate
{gate_reasons}
"""


def artifact_manifest(directory: Path) -> Dict[str, str]:
    """Hash all artifact payload files except the manifest currently being built."""

    return {
        file_path.name: sha256_file(file_path)
        for file_path in sorted(directory.iterdir())
        if file_path.is_file() and file_path.name != "manifest.json"
    }


def package_artifact_bundle(
    final_directory: Path,
    temporary_directory: Path,
    model: HybridQNN1Regressor,
    config: TrainingConfig,
    prepared: PreparedData,
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    naive_metrics: Dict[str, float],
    selection_report: Dict[str, Any],
    refit_report: Dict[str, Any],
    gate: Dict[str, Any],
    data_sha256: str,
) -> Path:
    """Write an atomic, reproducible model bundle and move it into place."""

    # The source is included so the architecture can be reconstructed later.
    shutil.copy2(Path(__file__), temporary_directory / "train_hybridqnn1.py")
    preprocessor_bundle = {
        "selector": prepared.selector,
        "x_scaler": prepared.x_scaler,
        "y_scaler": prepared.y_scaler,
        "selected_features": prepared.selected_features,
        "candidate_features": prepared.candidate_features,
        "lookback": config.lookback,
        "target_column": "close",
        "feature_range": [-1.0, 1.0],
    }
    joblib.dump(preprocessor_bundle, temporary_directory / "preprocessor.joblib")
    torch.save(
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "model_class": "HybridQNN1Regressor",
            "n_features": config.k_features,
            "model_hyperparameters": {
                "n_qubits": config.n_qubits,
                "q_layers": config.q_layers,
                "lstm_hidden_size": config.lstm_hidden_size,
                "lstm_layers": config.lstm_layers,
                "lstm_dropout": config.lstm_dropout,
            },
            "state_dict": model.state_dict(),
        },
        temporary_directory / "model_state.pt",
    )

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "test_last_close_baseline": naive_metrics,
        "release_gate": gate,
        "selection_training": selection_report,
        "refit_training": refit_report,
    }
    write_json(temporary_directory / "metrics.json", metrics)
    write_json(
        temporary_directory / "training_config.json",
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "config": config.to_jsonable(),
            "data": {
                "sha256": data_sha256,
                "raw_rows_after_indicators": prepared.raw_rows,
                "raw_train_rows": prepared.train_row_count,
                "raw_validation_rows": prepared.validation_row_count,
                "raw_test_rows": prepared.test_row_count,
                "date_start": prepared.date_start,
                "date_end": prepared.date_end,
            },
            "runtime": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "pennylane": qml.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )

    test_prediction_scaled = predict_scaled(
        model,
        prepared.test,
        torch.device("cpu"),
    )
    test_true_price = prepared.y_scaler.inverse_transform(prepared.test.y_scaled)
    test_prediction_price = prepared.y_scaler.inverse_transform(test_prediction_scaled)
    pd.DataFrame(
        {
            "timestamp": prepared.test.timestamps,
            "actual_close": test_true_price.reshape(-1),
            "predicted_close": test_prediction_price.reshape(-1),
            "last_close_baseline": prepared.test.naive_close,
            "residual": (
                test_true_price.reshape(-1) - test_prediction_price.reshape(-1)
            ),
        }
    ).to_csv(temporary_directory / "test_predictions.csv", index=False)

    (temporary_directory / "model_card.md").write_text(
        render_model_card(
            config,
            prepared,
            validation_metrics,
            test_metrics,
            naive_metrics,
            gate,
        ),
        encoding="utf-8",
    )
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


def profile_defaults(profile: str) -> Dict[str, Any]:
    """Expose a bounded smoke profile and a meaningful staging default."""

    defaults: Dict[str, Dict[str, Any]] = {
        "staging": {
            "lookback": 2,
            "k_features": 3,
            "n_qubits": 3,
            "q_layers": 1,
            "lstm_hidden_size": 8,
            "lstm_layers": 1,
            "lstm_dropout": 0.0,
            "holdout_ratio": 0.20,
            "validation_ratio": 0.20,
            "batch_size": 4,
            "pretrain_epochs": 20,
            "quantum_epochs": 20,
            "pretrain_lr": 5e-3,
            "quantum_lr": 5e-3,
            "patience": 5,
            "max_train_sequences": 0,
            "max_validation_sequences": 0,
            "max_test_sequences": 0,
            "cpu_threads": 4,
        },
        "smoke": {
            "lookback": 2,
            "k_features": 3,
            "n_qubits": 3,
            "q_layers": 1,
            "lstm_hidden_size": 6,
            "lstm_layers": 1,
            "lstm_dropout": 0.0,
            "holdout_ratio": 0.20,
            "validation_ratio": 0.25,
            "batch_size": 4,
            "pretrain_epochs": 2,
            "quantum_epochs": 2,
            "pretrain_lr": 1e-2,
            "quantum_lr": 1e-2,
            "patience": 1,
            "max_train_sequences": 64,
            "max_validation_sequences": 16,
            "max_test_sequences": 16,
            "cpu_threads": 2,
        },
    }
    return defaults[profile].copy()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and package a reproducible, staged HybridQNN1 forecasting model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        choices=("staging", "smoke"),
        default="staging",
        help="staging uses all available sequences; smoke verifies the pipeline cheaply",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--k-features", type=int)
    parser.add_argument("--n-qubits", type=int)
    parser.add_argument("--q-layers", type=int)
    parser.add_argument("--lstm-hidden-size", type=int)
    parser.add_argument("--lstm-layers", type=int)
    parser.add_argument("--lstm-dropout", type=float)
    parser.add_argument("--holdout-ratio", type=float)
    parser.add_argument("--validation-ratio", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--quantum-epochs", type=int)
    parser.add_argument("--pretrain-lr", type=float)
    parser.add_argument("--quantum-lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-train-sequences", type=int)
    parser.add_argument("--max-validation-sequences", type=int)
    parser.add_argument("--max-test-sequences", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-refit-on-train-validation",
        action="store_true",
        help="save the validation-selected model rather than refitting it on train+validation",
    )
    parser.add_argument(
        "--allow-underperforming",
        action="store_true",
        help="mark the artifact accepted even if it fails the last-close baseline gate",
    )
    parser.add_argument(
        "--max-test-rmse",
        type=float,
        default=None,
        help="optional original-price RMSE release threshold",
    )
    parser.add_argument(
        "--allow-large-circuits",
        action="store_true",
        help="allow more than five exact-simulator qubits after manual resource review",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    values = profile_defaults(args.profile)
    for field_name in tuple(values):
        cli_name = field_name.replace("_", "-")
        arg_value = getattr(args, field_name, None)
        if arg_value is not None:
            values[field_name] = arg_value
    return TrainingConfig(
        data_path=args.data_path.resolve(),
        output_dir=args.output_dir.resolve(),
        ticker=args.ticker.upper(),
        profile=args.profile,
        seed=args.seed,
        refit_on_train_validation=not args.no_refit_on_train_validation,
        require_beat_naive=not args.allow_underperforming,
        max_test_rmse=args.max_test_rmse,
        allow_large_circuits=args.allow_large_circuits,
        **values,
    )


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
    logger.info("Starting HybridQNN1 %s profile.", config.profile)
    logger.info("Exact PennyLane default.qubit runs on CPU; CUDA is intentionally unused.")
    logger.info("Configuration: %s", json.dumps(config.to_jsonable(), default=str))

    started_at = time.perf_counter()
    try:
        prepared = prepare_data(config, logger)
        data_sha256 = sha256_file(config.data_path)
        device = torch.device("cpu")

        selection_model = HybridQNN1Regressor(config.k_features, config).to(device)
        stage_1_report = fit_stage1_with_validation(
            selection_model,
            prepared.train,
            prepared.validation,
            config,
            device,
            logger,
        )
        selection_model.freeze_feature_extractor()
        stage_2_report = fit_stage2_with_validation(
            selection_model,
            prepared.train,
            prepared.validation,
            config,
            device,
            logger,
        )
        validation_prediction = predict_scaled(
            selection_model,
            prepared.validation,
            device,
        )
        validation_metrics = calculate_metrics(
            prepared.validation.y_scaled,
            validation_prediction,
            prepared.y_scaler,
        )

        refit_report: Dict[str, Any] = {"used": config.refit_on_train_validation}
        model_to_package = selection_model
        if config.refit_on_train_validation:
            set_reproducible_seed(config.seed, config.cpu_threads)
            refit_partition = concatenate_partitions(
                prepared.train,
                prepared.validation,
            )
            model_to_package = HybridQNN1Regressor(
                config.k_features,
                config,
            ).to(device)
            refit_stage_1_rmse = fit_stage1_fixed_epochs(
                model_to_package,
                refit_partition,
                config,
                device,
                stage_1_report.best_epoch,
            )
            model_to_package.freeze_feature_extractor()
            refit_stage_2_rmse, refit_gradient_seen = fit_stage2_fixed_epochs(
                model_to_package,
                refit_partition,
                config,
                device,
                stage_2_report.best_epoch,
            )
            if not refit_gradient_seen:
                raise RuntimeError("No finite quantum gradient reached the refit model.")
            refit_report.update(
                {
                    "stage_1_epochs": stage_1_report.best_epoch,
                    "stage_1_final_train_rmse": refit_stage_1_rmse[-1],
                    "stage_2_epochs": stage_2_report.best_epoch,
                    "stage_2_final_train_rmse": refit_stage_2_rmse[-1],
                    "quantum_gradient_seen": refit_gradient_seen,
                }
            )

        test_prediction = predict_scaled(model_to_package, prepared.test, device)
        test_metrics = calculate_metrics(
            prepared.test.y_scaled,
            test_prediction,
            prepared.y_scaler,
        )
        naive_metrics = calculate_naive_price_metrics(
            prepared.test.y_scaled,
            prepared.test.naive_close,
            prepared.y_scaler,
        )
        gate = release_gate(test_metrics, naive_metrics, config)
        selection_report = {
            "stage_1": stage_1_report.to_jsonable(),
            "stage_2": stage_2_report.to_jsonable(),
            "elapsed_seconds_before_packaging": time.perf_counter() - started_at,
        }

        close_artifact_file_handlers(logger)
        artifact_path = package_artifact_bundle(
            final_directory,
            temporary_directory,
            model_to_package,
            config,
            prepared,
            validation_metrics,
            test_metrics,
            naive_metrics,
            selection_report,
            refit_report,
            gate,
            data_sha256,
        )
        logger.info("Artifact bundle saved to %s", artifact_path)
        logger.info(
            "Validation/test price RMSE: %.6f / %.6f",
            validation_metrics["price_rmse"],
            test_metrics["price_rmse"],
        )
        logger.info(
            "Naive last-close test price RMSE: %.6f",
            naive_metrics["price_rmse"],
        )
        logger.info("Staging gate: %s", gate["status"])
        print(f"Artifact saved: {artifact_path}")
        print(f"Staging gate: {gate['status']}")
        if gate["reasons"]:
            print("Gate reasons:")
            for reason in gate["reasons"]:
                print(f"- {reason}")
        return 0 if gate["status"] == "accepted" else 2
    except Exception:
        logger.exception(
            "Training failed. The temporary artifact directory is retained for diagnosis: %s",
            temporary_directory,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
