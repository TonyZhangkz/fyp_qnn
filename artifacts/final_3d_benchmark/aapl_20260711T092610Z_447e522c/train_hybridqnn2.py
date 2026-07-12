#!/usr/bin/env python
"""Train and package the paper-inspired joint HybridQNN2 architecture.

HybridQNN2 differs from HybridQNN1 by optimizing the classical LSTM branch,
the quantum branch, and the fusion head together. The script reuses the
leakage-safe preprocessing and release gate from ``train_hybridqnn1.py`` while
writing a distinct, self-describing artifact bundle.

Examples
--------
Meaningful local staging run:
    conda run -n env_qubit python scripts/train_hybridqnn2.py

Fast end-to-end verification:
    conda run -n env_qubit python scripts/train_hybridqnn2.py --profile smoke
"""

from __future__ import annotations

import argparse
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
from torch.optim import Adam

from train_hybridqnn1 import (
    ARTIFACT_FORMAT_VERSION,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    PreparedData,
    SequencePartition,
    artifact_manifest,
    calculate_metrics,
    calculate_naive_price_metrics,
    close_artifact_file_handlers,
    concatenate_partitions,
    configure_logging,
    prepare_data,
    release_gate,
    set_reproducible_seed,
    sha256_file,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QNN1_SOURCE_PATH = Path(__file__).resolve().with_name("train_hybridqnn1.py")


@dataclass
class HybridQNN2Config:
    """All settings for a reproducible joint HybridQNN2 run."""

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
    joint_epochs: int
    joint_lr: float
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
            raise ValueError("k_features must equal n_qubits for angle encoding.")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive.")
        if self.n_qubits > 5 and not self.allow_large_circuits:
            raise ValueError(
                "Refusing to run more than five exact-simulator qubits. "
                "Pass --allow-large-circuits only after a resource review."
            )
        if not 0.0 < self.holdout_ratio < 0.5:
            raise ValueError("holdout_ratio must be between 0 and 0.5.")
        if not 0.0 < self.validation_ratio < 0.5:
            raise ValueError("validation_ratio must be between 0 and 0.5.")
        if self.lookback < 1:
            raise ValueError("lookback must be at least one.")
        if self.batch_size < 1 or self.joint_epochs < 1:
            raise ValueError("batch_size and joint_epochs must be positive.")
        if self.fusion_hidden_size < 1 or self.lstm_hidden_size < 1:
            raise ValueError("hidden sizes must be positive.")
        if self.patience < 1 or self.cpu_threads < 1:
            raise ValueError("patience and cpu_threads must be positive.")

    def to_jsonable(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass
class JointFitReport:
    """Joint-training validation outcome and full loss history."""

    best_epoch: int
    epochs_ran: int
    best_validation_rmse: float
    history: List[Dict[str, float]]
    quantum_gradient_seen: bool

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "best_validation_rmse": self.best_validation_rmse,
            "history": self.history,
            "quantum_gradient_seen": self.quantum_gradient_seen,
        }


class HybridQNN2Regressor(nn.Module):
    """Joint LSTM and QNN branches fused into one trainable prediction head.

    The paper's Fig. 7 describes concurrent LSTM/QNN processing followed by a
    learned fusion layer. Fig. 5's `Ppr` blocks are undefined, so the QNN
    branch uses the explicitly described RY/RZ rotations and CNOT/CZ
    entanglement instead of inventing a `Ppr` unitary.
    """

    def __init__(self, n_features: int, config: HybridQNN2Config) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
        )
        self.quantum_projection = nn.Linear(
            config.lookback * n_features,
            config.n_qubits,
        )
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(config.q_layers, config.n_qubits, 2)
        )
        self.output_head = nn.Sequential(
            nn.Linear(
                config.lstm_hidden_size + config.n_qubits,
                config.fusion_hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_size, 1),
        )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        classical_features = hidden[-1]

        flattened = x.reshape(x.size(0), -1)
        quantum_inputs = torch.tanh(self.quantum_projection(flattened))
        quantum_outputs = []
        for sample in quantum_inputs:
            q_result = self.qnode(sample, self.q_weights)
            quantum_outputs.append(torch.stack(list(q_result)))
        quantum_features = torch.stack(quantum_outputs).to(
            dtype=classical_features.dtype
        )

        fused_features = torch.cat([classical_features, quantum_features], dim=1)
        return self.output_head(fused_features).squeeze(-1)


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
        torch.as_tensor(
            partition.y_scaled.reshape(-1),
            dtype=torch.float32,
            device=device,
        ),
    )


def ordered_batches(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield chronological mini-batches without randomizing the time series."""

    for start in range(0, len(X), batch_size):
        yield X[start:start + batch_size], y[start:start + batch_size]


def validation_rmse(
    model: HybridQNN2Regressor,
    partition: SequencePartition,
    device: torch.device,
) -> float:
    X_validation, y_validation = _to_tensors(partition, device)
    model.eval()
    with torch.no_grad():
        prediction = model(X_validation)
        return float(torch.sqrt(nn.functional.mse_loss(prediction, y_validation)).cpu())


def fit_joint_with_validation(
    model: HybridQNN2Regressor,
    train: SequencePartition,
    validation: SequencePartition,
    config: HybridQNN2Config,
    device: torch.device,
    logger: logging.Logger,
) -> JointFitReport:
    """Train both branches and the fusion head under a shared validation loss."""

    X_train, y_train = _to_tensors(train, device)
    optimizer = Adam(model.parameters(), lr=config.joint_lr)
    loss_fn = nn.MSELoss()
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    quantum_gradient_seen = False
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.joint_epochs + 1):
        model.train()
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
        current_validation_rmse = validation_rmse(model, validation, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_rmse": train_rmse,
                "validation_rmse": current_validation_rmse,
            }
        )
        current_validation_loss = current_validation_rmse ** 2

        if current_validation_loss < best_loss - 1e-12:
            best_loss = current_validation_loss
            best_state = _copy_state_dict(model)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                logger.info("Joint training early stopped at epoch %s.", epoch)
                break

    if best_state is None:
        raise RuntimeError("Joint training did not produce a valid state.")
    if not quantum_gradient_seen:
        raise RuntimeError("No finite gradient reached the quantum weights.")

    model.load_state_dict(best_state)
    logger.info(
        "Joint training selected epoch %s with validation RMSE %.6f.",
        best_epoch,
        np.sqrt(best_loss),
    )
    return JointFitReport(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_validation_rmse=float(np.sqrt(best_loss)),
        history=history,
        quantum_gradient_seen=quantum_gradient_seen,
    )


def fit_joint_fixed_epochs(
    model: HybridQNN2Regressor,
    partition: SequencePartition,
    config: HybridQNN2Config,
    device: torch.device,
    epochs: int,
) -> Tuple[List[float], bool]:
    """Refit all QNN2 parameters on train plus validation after selection."""

    X_train, y_train = _to_tensors(partition, device)
    optimizer = Adam(model.parameters(), lr=config.joint_lr)
    loss_fn = nn.MSELoss()
    history: List[float] = []
    quantum_gradient_seen = False

    for _ in range(max(1, epochs)):
        model.train()
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


def predict_scaled(
    model: HybridQNN2Regressor,
    partition: SequencePartition,
    device: torch.device,
) -> np.ndarray:
    X, _ = _to_tensors(partition, device)
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy().reshape(-1, 1)


def render_model_card(
    config: HybridQNN2Config,
    prepared: PreparedData,
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    naive_metrics: Dict[str, float],
    gate: Dict[str, Any],
) -> str:
    gate_reasons = "\n".join(
        f"- {reason}" for reason in gate["reasons"]
    ) or "- All configured staging gates passed."
    return f"""# HybridQNN2 model card

## Status
`{gate["status"]}`

## Intended use
One-step close-price regression for research and staging. This bundle is not a
trading recommendation and must not be deployed without independent
backtesting, monitoring, governance review, and an approved promotion gate.

## Architecture
HybridQNN2 jointly optimizes a parallel LSTM branch and quantum branch. Their
representations are concatenated and passed through a learned fusion head.
The article's `Ppr` blocks are undefined, so the QNN branch uses the explicit
RY/RZ and CNOT/CZ operations described in the text.

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


def package_artifact_bundle(
    final_directory: Path,
    temporary_directory: Path,
    model: HybridQNN2Regressor,
    config: HybridQNN2Config,
    prepared: PreparedData,
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    naive_metrics: Dict[str, float],
    joint_report: Dict[str, Any],
    refit_report: Dict[str, Any],
    gate: Dict[str, Any],
    data_sha256: str,
) -> Path:
    """Write an atomic QNN2 artifact bundle with preprocessing and audit data."""

    shutil.copy2(Path(__file__), temporary_directory / "train_hybridqnn2.py")
    shutil.copy2(QNN1_SOURCE_PATH, temporary_directory / "train_hybridqnn1.py")
    joblib.dump(
        {
            "selector": prepared.selector,
            "x_scaler": prepared.x_scaler,
            "y_scaler": prepared.y_scaler,
            "selected_features": prepared.selected_features,
            "candidate_features": prepared.candidate_features,
            "lookback": config.lookback,
            "target_column": "close",
            "feature_range": [-1.0, 1.0],
        },
        temporary_directory / "preprocessor.joblib",
    )
    torch.save(
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "model_class": "HybridQNN2Regressor",
            "n_features": config.k_features,
            "model_hyperparameters": {
                "n_qubits": config.n_qubits,
                "q_layers": config.q_layers,
                "lstm_hidden_size": config.lstm_hidden_size,
                "lstm_layers": config.lstm_layers,
                "lstm_dropout": config.lstm_dropout,
                "fusion_hidden_size": config.fusion_hidden_size,
                "lookback": config.lookback,
            },
            "state_dict": model.state_dict(),
        },
        temporary_directory / "model_state.pt",
    )

    write_json(
        temporary_directory / "metrics.json",
        {
            "validation": validation_metrics,
            "test": test_metrics,
            "test_last_close_baseline": naive_metrics,
            "release_gate": gate,
            "joint_training": joint_report,
            "refit_training": refit_report,
        },
    )
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

    prediction_scaled = predict_scaled(model, prepared.test, torch.device("cpu"))
    actual_price = prepared.y_scaler.inverse_transform(prepared.test.y_scaled)
    prediction_price = prepared.y_scaler.inverse_transform(prediction_scaled)
    pd.DataFrame(
        {
            "timestamp": prepared.test.timestamps,
            "actual_close": actual_price.reshape(-1),
            "predicted_close": prediction_price.reshape(-1),
            "last_close_baseline": prepared.test.naive_close,
            "residual": actual_price.reshape(-1) - prediction_price.reshape(-1),
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
    defaults: Dict[str, Dict[str, Any]] = {
        "staging": {
            # Best resource-bounded configuration from the 2026-07-11 sweep.
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
            "joint_epochs": 40,
            "joint_lr": 1e-3,
            "patience": 10,
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
            "joint_epochs": 2,
            "joint_lr": 1e-2,
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
            "Train and package a reproducible joint HybridQNN2 forecasting model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=("staging", "smoke"), default="staging")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR.parent / "hybridqnn2",
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
    parser.add_argument("--joint-epochs", type=int)
    parser.add_argument("--joint-lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-train-sequences", type=int)
    parser.add_argument("--max-validation-sequences", type=int)
    parser.add_argument("--max-test-sequences", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-refit-on-train-validation", action="store_true")
    parser.add_argument("--allow-underperforming", action="store_true")
    parser.add_argument("--max-test-rmse", type=float, default=None)
    parser.add_argument("--allow-large-circuits", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> HybridQNN2Config:
    values = profile_defaults(args.profile)
    for field_name in tuple(values):
        value = getattr(args, field_name, None)
        if value is not None:
            values[field_name] = value
    return HybridQNN2Config(
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
    logger.info("Starting HybridQNN2 %s profile.", config.profile)
    logger.info("Exact PennyLane default.qubit runs on CPU; CUDA is intentionally unused.")
    logger.info("Configuration: %s", json.dumps(config.to_jsonable(), default=str))

    started_at = time.perf_counter()
    try:
        prepared = prepare_data(config, logger)
        data_sha256 = sha256_file(config.data_path)
        device = torch.device("cpu")

        selection_model = HybridQNN2Regressor(config.k_features, config).to(device)
        joint_report = fit_joint_with_validation(
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

        model_to_package = selection_model
        refit_report: Dict[str, Any] = {"used": config.refit_on_train_validation}
        if config.refit_on_train_validation:
            set_reproducible_seed(config.seed, config.cpu_threads)
            refit_partition = concatenate_partitions(
                prepared.train,
                prepared.validation,
            )
            model_to_package = HybridQNN2Regressor(
                config.k_features,
                config,
            ).to(device)
            refit_history, refit_gradient_seen = fit_joint_fixed_epochs(
                model_to_package,
                refit_partition,
                config,
                device,
                joint_report.best_epoch,
            )
            if not refit_gradient_seen:
                raise RuntimeError("No finite quantum gradient reached the refit model.")
            refit_report.update(
                {
                    "joint_epochs": joint_report.best_epoch,
                    "joint_final_train_rmse": refit_history[-1],
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
        report = {
            "joint": joint_report.to_jsonable(),
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
            report,
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
