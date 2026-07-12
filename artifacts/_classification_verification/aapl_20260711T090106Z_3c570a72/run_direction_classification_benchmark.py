#!/usr/bin/env python
"""Run a fixed binary next-bar direction benchmark for LSTM, HybridQNN1, and HybridQNN2.

The target is 1 (up) when the close at the prediction timestamp is greater
than the close at the final timestep in the input sequence; otherwise it is 0
(down or unchanged). All models use the same chronological split, selected
features, and test partition. This script intentionally does not perform a
hyperparameter sweep.

Example
-------
    conda run -n env_qubit python scripts/run_direction_classification_benchmark.py
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
import sklearn
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam

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
    sha256_file,
    write_json,
)
from train_hybridqnn2 import HybridQNN2Regressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QNN1_SOURCE_PATH = Path(__file__).resolve().with_name("train_hybridqnn1.py")
QNN2_SOURCE_PATH = Path(__file__).resolve().with_name("train_hybridqnn2.py")
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
class ClassificationConfig:
    """Fixed, resource-bounded settings shared by all classification models."""

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
    max_train_sequences: int
    max_validation_sequences: int
    max_test_sequences: int
    cpu_threads: int
    seed: int
    allow_large_circuits: bool

    def validate(self) -> None:
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Data file was not found: {self.data_path}")
        if self.k_features != self.n_qubits:
            raise ValueError("k_features must equal n_qubits for angle encoding.")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive.")
        if self.n_qubits > 5 and not self.allow_large_circuits:
            raise ValueError(
                "Refusing more than five exact-simulator qubits without "
                "--allow-large-circuits."
            )
        if self.lookback < 1 or self.batch_size < 1:
            raise ValueError("lookback and batch_size must be positive.")
        if (
            self.baseline_epochs < 1
            or self.hqnn1_pretrain_epochs < 1
            or self.hqnn1_quantum_epochs < 1
            or self.hqnn2_joint_epochs < 1
        ):
            raise ValueError("Every model stage needs at least one epoch.")
        if self.patience < 1 or self.cpu_threads < 1:
            raise ValueError("patience and cpu_threads must be positive.")
        if not 0.0 < self.holdout_ratio < 0.5:
            raise ValueError("holdout_ratio must be between 0 and 0.5.")
        if not 0.0 < self.validation_ratio < 0.5:
            raise ValueError("validation_ratio must be between 0 and 0.5.")

    def to_jsonable(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass
class DirectionPartition:
    X: np.ndarray
    y: np.ndarray
    timestamps: List[str]
    prior_close: np.ndarray


@dataclass
class DirectionData:
    train: DirectionPartition
    validation: DirectionPartition
    test: DirectionPartition
    selector: SelectKBest
    x_scaler: MinMaxScaler
    selected_features: List[str]
    candidate_features: List[str]
    raw_rows: int
    date_start: str
    date_end: str


@dataclass
class FitReport:
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


class LSTMClassifier(nn.Module):
    """Classical recurrent baseline with the same temporal input shape."""

    def __init__(self, n_features: int, config: ClassificationConfig) -> None:
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


def _as_partition(
    sequences: List[np.ndarray],
    labels: List[int],
    timestamps: List[str],
    prior_close: List[float],
    lookback: int,
    n_features: int,
    name: str,
) -> DirectionPartition:
    if not sequences:
        raise ValueError(
            f"{name} contains no sequences. Provide more data or reduce split sizes."
        )
    return DirectionPartition(
        X=np.asarray(sequences, dtype=np.float32).reshape(-1, lookback, n_features),
        y=np.asarray(labels, dtype=np.float32).reshape(-1),
        timestamps=timestamps,
        prior_close=np.asarray(prior_close, dtype=np.float64),
    )


def _cap_recent(
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


def prepare_direction_data(
    config: ClassificationConfig,
    logger: logging.Logger,
) -> DirectionData:
    """Construct next-bar direction labels and leakage-safe feature sequences."""

    raw = pd.read_parquet(config.data_path).dropna()
    data = add_technical_indicators(canonicalize_ohlcv_columns(raw))
    row_count = len(data)
    if row_count <= config.lookback + 20:
        raise ValueError("Not enough post-indicator rows for classification.")

    candidate_features = [
        feature for feature in FEATURE_CANDIDATES if feature in data.columns
    ]
    if len(candidate_features) < config.k_features:
        raise ValueError(
            f"Requested {config.k_features} features but found {candidate_features}."
        )

    test_start = int(row_count * (1.0 - config.holdout_ratio))
    train_end = int(test_start * (1.0 - config.validation_ratio))
    if train_end <= config.lookback:
        raise ValueError("Training partition is too short for the requested lookback.")

    close = data["close"].to_numpy(dtype=np.float64)
    # Label at row i predicts whether close[i] rose over close[i - 1].
    directions = (close[1:] > close[:-1]).astype(np.int64)
    features = data[candidate_features]

    # Select features at time i using the direction at time i+1: no label leakage.
    selector = SelectKBest(score_func=f_classif, k=config.k_features)
    selector.fit(
        features.iloc[:train_end - 1],
        directions[:train_end - 1],
    )
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
        if target_index < train_end:
            bucket_name = "train"
        elif target_index < test_start:
            bucket_name = "validation"
        else:
            bucket_name = "test"

        bucket = buckets[bucket_name]
        bucket["X"].append(X_scaled[target_index - config.lookback:target_index])
        bucket["y"].append(int(close[target_index] > close[target_index - 1]))
        bucket["timestamps"].append(str(data.index[target_index]))
        bucket["prior"].append(float(close[target_index - 1]))

    train = _cap_recent(
        _as_partition(
            buckets["train"]["X"],
            buckets["train"]["y"],
            buckets["train"]["timestamps"],
            buckets["train"]["prior"],
            config.lookback,
            config.k_features,
            "train",
        ),
        config.max_train_sequences,
    )
    validation = _cap_recent(
        _as_partition(
            buckets["validation"]["X"],
            buckets["validation"]["y"],
            buckets["validation"]["timestamps"],
            buckets["validation"]["prior"],
            config.lookback,
            config.k_features,
            "validation",
        ),
        config.max_validation_sequences,
    )
    test = _cap_recent(
        _as_partition(
            buckets["test"]["X"],
            buckets["test"]["y"],
            buckets["test"]["timestamps"],
            buckets["test"]["prior"],
            config.lookback,
            config.k_features,
            "test",
        ),
        config.max_test_sequences,
    )
    logger.info(
        "Prepared %s rows into %s/%s/%s train/validation/test direction sequences.",
        row_count,
        len(train.X),
        len(validation.X),
        len(test.X),
    )
    logger.info(
        "Up rates (train/validation/test): %.3f / %.3f / %.3f",
        float(train.y.mean()),
        float(validation.y.mean()),
        float(test.y.mean()),
    )
    logger.info("Selected features: %s", selected_features)
    return DirectionData(
        train=train,
        validation=validation,
        test=test,
        selector=selector,
        x_scaler=x_scaler,
        selected_features=selected_features,
        candidate_features=candidate_features,
        raw_rows=row_count,
        date_start=str(data.index[0]),
        date_end=str(data.index[-1]),
    )


def _to_tensors(
    partition: DirectionPartition,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(partition.X, dtype=torch.float32, device=device),
        torch.as_tensor(partition.y, dtype=torch.float32, device=device),
    )


def ordered_batches(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for start in range(0, len(X), batch_size):
        yield X[start:start + batch_size], y[start:start + batch_size]


def class_weight(train: DirectionPartition, device: torch.device) -> torch.Tensor:
    positives = float(train.y.sum())
    negatives = float(len(train.y) - positives)
    if positives == 0.0 or negatives == 0.0:
        raise ValueError("Training partition contains only one class.")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def validation_loss(
    model: nn.Module,
    partition: DirectionPartition,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    X, y = _to_tensors(partition, device)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        return float(loss_fn(logits, y).cpu())


def fit_classifier(
    model: nn.Module,
    train: DirectionPartition,
    validation: DirectionPartition,
    epochs: int,
    learning_rate: float,
    config: ClassificationConfig,
    device: torch.device,
    logger: logging.Logger,
    model_name: str,
) -> FitReport:
    """Fit a logit-producing model with early stopping on validation BCE loss."""

    X_train, y_train = _to_tensors(train, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight(train, device))
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    quantum_gradient_seen = False

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for X_batch, y_batch in ordered_batches(X_train, y_train, config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            quantum_weights = getattr(model, "q_weights", None)
            if quantum_weights is not None and quantum_weights.grad is not None:
                quantum_gradient_seen = quantum_gradient_seen or bool(
                    torch.isfinite(quantum_weights.grad).all().item()
                )
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)

        train_bce = total_loss / total_count
        current_validation_loss = validation_loss(model, validation, loss_fn, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_bce": float(train_bce),
                "validation_bce": current_validation_loss,
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
                logger.info("%s early stopped at epoch %s.", model_name, epoch)
                break

    if best_state is None:
        raise RuntimeError(f"{model_name} did not produce a valid state.")
    model.load_state_dict(best_state)
    logger.info(
        "%s selected epoch %s with validation BCE %.6f.",
        model_name,
        best_epoch,
        best_loss,
    )
    return FitReport(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_validation_loss=best_loss,
        history=history,
        quantum_gradient_seen=quantum_gradient_seen,
    )


def fit_hqnn1_pretrain(
    model: HybridQNN1Regressor,
    train: DirectionPartition,
    validation: DirectionPartition,
    config: ClassificationConfig,
    device: torch.device,
    logger: logging.Logger,
) -> FitReport:
    """Train the HybridQNN1 LSTM classifier before freezing it for Stage 2."""

    X_train, y_train = _to_tensors(train, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight(train, device))
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
            _, logits = model.extractor(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)

        X_validation, y_validation = _to_tensors(validation, device)
        model.extractor.eval()
        with torch.no_grad():
            _, validation_logits = model.extractor(X_validation)
            current_validation_loss = float(loss_fn(validation_logits, y_validation).cpu())
        history.append(
            {
                "epoch": float(epoch),
                "train_bce": float(total_loss / total_count),
                "validation_bce": current_validation_loss,
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
                logger.info("HybridQNN1 Stage 1 early stopped at epoch %s.", epoch)
                break

    if best_state is None:
        raise RuntimeError("HybridQNN1 Stage 1 did not produce a valid state.")
    model.extractor.load_state_dict(best_state)
    logger.info(
        "HybridQNN1 Stage 1 selected epoch %s with validation BCE %.6f.",
        best_epoch,
        best_loss,
    )
    return FitReport(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_validation_loss=best_loss,
        history=history,
    )


def predict_probabilities(
    model: nn.Module,
    partition: DirectionPartition,
    device: torch.device,
) -> np.ndarray:
    X, _ = _to_tensors(partition, device)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).cpu().numpy().reshape(-1)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    try:
        roc_auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        roc_auc = None
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_up": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_up": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_up": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": matrix,
        "predicted_up_rate": float(predictions.mean()),
        "actual_up_rate": float(y_true.mean()),
    }


def render_model_card(
    config: ClassificationConfig,
    data: DirectionData,
    result_rows: List[Dict[str, Any]],
) -> str:
    rows = "\n".join(
        (
            f"- {row['model']}: accuracy {row['test']['accuracy']:.4f}, "
            f"F1-up {row['test']['f1_up']:.4f}, "
            f"ROC-AUC {row['test']['roc_auc'] if row['test']['roc_auc'] is not None else 'n/a'}, "
            f"training time {row['training_seconds']:.2f}s"
        )
        for row in result_rows
    )
    return f"""# Binary direction-classification benchmark

## Target
Up is 1 when the close at the prediction timestamp exceeds the close at the
last timestep in the input sequence. Down or unchanged is 0.

## Data and split
- Ticker label: `{config.ticker}`
- Source: `{config.data_path}`
- Range: `{data.date_start}` to `{data.date_end}`
- Selected features: `{", ".join(data.selected_features)}`
- Lookback: `{config.lookback}`
- Chronological split with no shuffle; hyperparameters were fixed before this run.

## Test results
{rows}

## Interpretation
This is a single-split research benchmark, not a production promotion result.
Use walk-forward folds and transaction-cost-aware evaluation before making any
trading or deployment decision.
"""


def package_artifact_bundle(
    final_directory: Path,
    temporary_directory: Path,
    config: ClassificationConfig,
    data: DirectionData,
    results: List[Dict[str, Any]],
    models: Dict[str, nn.Module],
) -> Path:
    """Save states, preprocessing, predictions, and reproducible benchmark metrics."""

    shutil.copy2(
        Path(__file__),
        temporary_directory / "run_direction_classification_benchmark.py",
    )
    shutil.copy2(QNN1_SOURCE_PATH, temporary_directory / "train_hybridqnn1.py")
    shutil.copy2(QNN2_SOURCE_PATH, temporary_directory / "train_hybridqnn2.py")
    joblib.dump(
        {
            "selector": data.selector,
            "x_scaler": data.x_scaler,
            "selected_features": data.selected_features,
            "candidate_features": data.candidate_features,
            "lookback": config.lookback,
            "target_definition": "close[t] > close[t - 1]",
        },
        temporary_directory / "preprocessor.joblib",
    )
    for model_name, model in models.items():
        torch.save(
            {
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "model_name": model_name,
                "state_dict": model.state_dict(),
            },
            temporary_directory / f"{model_name.lower().replace(' ', '_')}_state.pt",
        )

    write_json(
        temporary_directory / "classification_metrics.json",
        {"results": results},
    )
    write_json(
        temporary_directory / "benchmark_config.json",
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "config": config.to_jsonable(),
            "data": {
                "raw_rows_after_indicators": data.raw_rows,
                "date_start": data.date_start,
                "date_end": data.date_end,
                "train_sequences": len(data.train.X),
                "validation_sequences": len(data.validation.X),
                "test_sequences": len(data.test.X),
            },
            "runtime": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )

    result_frame = pd.DataFrame(
        [
            {
                "Model": row["model"],
                "Accuracy": row["test"]["accuracy"],
                "BalancedAccuracy": row["test"]["balanced_accuracy"],
                "F1Up": row["test"]["f1_up"],
                "ROCAUC": row["test"]["roc_auc"],
                "TrainingSeconds": row["training_seconds"],
                "BestEpoch": row["best_epoch"],
            }
            for row in results
        ]
    )
    result_frame.to_csv(temporary_directory / "classification_results.csv", index=False)

    prediction_frame: Dict[str, Any] = {
        "timestamp": data.test.timestamps,
        "actual_direction_up": data.test.y.astype(int),
    }
    for row in results:
        model_key = row["model"].lower().replace(" ", "_")
        prediction_frame[f"{model_key}_probability_up"] = row["test_probabilities"]
        prediction_frame[f"{model_key}_prediction_up"] = (
            np.asarray(row["test_probabilities"]) >= 0.5
        ).astype(int)
    pd.DataFrame(prediction_frame).to_csv(
        temporary_directory / "test_direction_predictions.csv",
        index=False,
    )

    (temporary_directory / "model_card.md").write_text(
        render_model_card(config, data, results),
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
    defaults: Dict[str, Dict[str, Any]] = {
        "benchmark": {
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
            "batch_size": 4,
            "baseline_epochs": 20,
            "hqnn1_pretrain_epochs": 10,
            "hqnn1_quantum_epochs": 10,
            "hqnn2_joint_epochs": 20,
            "baseline_lr": 1e-3,
            "hqnn1_lr": 5e-3,
            "hqnn2_lr": 1e-3,
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
            "fusion_hidden_size": 6,
            "holdout_ratio": 0.20,
            "validation_ratio": 0.25,
            "batch_size": 4,
            "baseline_epochs": 2,
            "hqnn1_pretrain_epochs": 2,
            "hqnn1_quantum_epochs": 2,
            "hqnn2_joint_epochs": 2,
            "baseline_lr": 1e-2,
            "hqnn1_lr": 1e-2,
            "hqnn2_lr": 1e-2,
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
        description="Run a fixed binary direction-classification benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=("benchmark", "smoke"), default="benchmark")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR.parent / "direction_classification",
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--k-features", type=int)
    parser.add_argument("--n-qubits", type=int)
    parser.add_argument("--q-layers", type=int)
    parser.add_argument("--lstm-hidden-size", type=int)
    parser.add_argument("--lstm-layers", type=int)
    parser.add_argument("--lstm-dropout", type=float)
    parser.add_argument("--fusion-hidden-size", type=int)
    parser.add_argument("--holdout-ratio", type=float)
    parser.add_argument("--validation-ratio", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--baseline-epochs", type=int)
    parser.add_argument("--hqnn1-pretrain-epochs", type=int)
    parser.add_argument("--hqnn1-quantum-epochs", type=int)
    parser.add_argument("--hqnn2-joint-epochs", type=int)
    parser.add_argument("--baseline-lr", type=float)
    parser.add_argument("--hqnn1-lr", type=float)
    parser.add_argument("--hqnn2-lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-train-sequences", type=int)
    parser.add_argument("--max-validation-sequences", type=int)
    parser.add_argument("--max-test-sequences", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-large-circuits", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ClassificationConfig:
    values = profile_defaults(args.profile)
    for field_name in tuple(values):
        value = getattr(args, field_name, None)
        if value is not None:
            values[field_name] = value
    return ClassificationConfig(
        data_path=args.data_path.resolve(),
        output_dir=args.output_dir.resolve(),
        ticker=args.ticker.upper(),
        profile=args.profile,
        seed=args.seed,
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
    logger.info("Starting %s direction-classification benchmark.", config.profile)
    logger.info("Exact PennyLane default.qubit runs on CPU; CUDA is intentionally unused.")
    logger.info("Configuration: %s", json.dumps(config.to_jsonable(), default=str))

    try:
        data = prepare_direction_data(config, logger)
        device = torch.device("cpu")
        results: List[Dict[str, Any]] = []
        models: Dict[str, nn.Module] = {}

        # Classical baseline
        set_reproducible_seed(config.seed, config.cpu_threads)
        baseline = LSTMClassifier(config.k_features, config).to(device)
        started = time.perf_counter()
        baseline_report = fit_classifier(
            baseline,
            data.train,
            data.validation,
            config.baseline_epochs,
            config.baseline_lr,
            config,
            device,
            logger,
            "LSTM baseline",
        )
        baseline_seconds = time.perf_counter() - started
        baseline_probabilities = predict_probabilities(baseline, data.test, device)
        results.append(
            {
                "model": "LSTM baseline",
                "training_seconds": baseline_seconds,
                "best_epoch": baseline_report.best_epoch,
                "training": baseline_report.to_jsonable(),
                "test": classification_metrics(data.test.y.astype(int), baseline_probabilities),
                "test_probabilities": baseline_probabilities.tolist(),
            }
        )
        models["LSTM baseline"] = baseline

        # HybridQNN1: sequential classical pretraining followed by frozen QNN stage.
        set_reproducible_seed(config.seed, config.cpu_threads)
        hqnn1 = HybridQNN1Regressor(config.k_features, config).to(device)
        started = time.perf_counter()
        hqnn1_stage1 = fit_hqnn1_pretrain(
            hqnn1,
            data.train,
            data.validation,
            config,
            device,
            logger,
        )
        hqnn1.freeze_feature_extractor()
        hqnn1_stage2 = fit_classifier(
            hqnn1,
            data.train,
            data.validation,
            config.hqnn1_quantum_epochs,
            config.hqnn1_lr,
            config,
            device,
            logger,
            "HybridQNN1 Stage 2",
        )
        hqnn1_seconds = time.perf_counter() - started
        hqnn1_probabilities = predict_probabilities(hqnn1, data.test, device)
        results.append(
            {
                "model": "HybridQNN1",
                "training_seconds": hqnn1_seconds,
                "best_epoch": hqnn1_stage2.best_epoch,
                "training": {
                    "stage_1": hqnn1_stage1.to_jsonable(),
                    "stage_2": hqnn1_stage2.to_jsonable(),
                },
                "test": classification_metrics(data.test.y.astype(int), hqnn1_probabilities),
                "test_probabilities": hqnn1_probabilities.tolist(),
            }
        )
        models["HybridQNN1"] = hqnn1

        # HybridQNN2: jointly train LSTM, quantum, and fusion parameters.
        set_reproducible_seed(config.seed, config.cpu_threads)
        hqnn2 = HybridQNN2Regressor(config.k_features, config).to(device)
        started = time.perf_counter()
        hqnn2_report = fit_classifier(
            hqnn2,
            data.train,
            data.validation,
            config.hqnn2_joint_epochs,
            config.hqnn2_lr,
            config,
            device,
            logger,
            "HybridQNN2 joint",
        )
        hqnn2_seconds = time.perf_counter() - started
        hqnn2_probabilities = predict_probabilities(hqnn2, data.test, device)
        results.append(
            {
                "model": "HybridQNN2",
                "training_seconds": hqnn2_seconds,
                "best_epoch": hqnn2_report.best_epoch,
                "training": hqnn2_report.to_jsonable(),
                "test": classification_metrics(data.test.y.astype(int), hqnn2_probabilities),
                "test_probabilities": hqnn2_probabilities.tolist(),
            }
        )
        models["HybridQNN2"] = hqnn2

        close_artifact_file_handlers(logger)
        artifact_path = package_artifact_bundle(
            final_directory,
            temporary_directory,
            config,
            data,
            results,
            models,
        )
        logger.info("Benchmark artifact saved to %s", artifact_path)
        for result in results:
            logger.info(
                "%s: accuracy %.4f | F1-up %.4f | ROC-AUC %s | %.2fs",
                result["model"],
                result["test"]["accuracy"],
                result["test"]["f1_up"],
                (
                    f"{result['test']['roc_auc']:.4f}"
                    if result["test"]["roc_auc"] is not None
                    else "n/a"
                ),
                result["training_seconds"],
            )
        print(f"Benchmark artifact saved: {artifact_path}")
        return 0
    except Exception:
        logger.exception(
            "Benchmark failed. The temporary directory is retained for diagnosis: %s",
            temporary_directory,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
