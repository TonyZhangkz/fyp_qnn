#!/usr/bin/env python
"""Tune causal feature filters before model input and write fixed configurations.

Available filters:
1. Scalar causal Kalman filter with tuned process/measurement variance ratios.
2. Causal endpoint Savitzky-Golay filter with tuned window/polyorder.
3. Causal ten-trading-day rolling average.

The tuning objective is validation ROC-AUC from a fixed classical LSTM
direction-classification proxy. Test metrics are recorded only after selection
and are never used to choose a filter configuration.

Example
-------
    conda run -n env_qubit python scripts/tune_feature_filters.py
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
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler

from run_direction_classification_benchmark import (
    DirectionPartition,
    LSTMClassifier,
    classification_metrics,
    fit_classifier,
    predict_probabilities,
)
from train_hybridqnn1 import (
    ARTIFACT_FORMAT_VERSION,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    add_technical_indicators,
    artifact_manifest,
    canonicalize_ohlcv_columns,
    close_artifact_file_handlers,
    configure_logging,
    set_reproducible_seed,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
class FilterTuningConfig:
    data_path: Path
    output_dir: Path
    fixed_config_path: Path
    ticker: str
    lookback: int
    k_features: int
    lstm_hidden_size: int
    holdout_ratio: float
    validation_ratio: float
    batch_size: int
    proxy_epochs: int
    proxy_lr: float
    patience: int
    cpu_threads: int
    seed: int
    max_candidates: int

    def validate(self) -> None:
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Data file was not found: {self.data_path}")
        if self.lookback < 1 or self.k_features < 1:
            raise ValueError("lookback and k_features must be positive.")
        if not 0.0 < self.holdout_ratio < 0.5:
            raise ValueError("holdout_ratio must be between 0 and 0.5.")
        if not 0.0 < self.validation_ratio < 0.5:
            raise ValueError("validation_ratio must be between 0 and 0.5.")
        if self.batch_size < 1 or self.proxy_epochs < 1:
            raise ValueError("batch_size and proxy_epochs must be positive.")
        if self.patience < 1 or self.cpu_threads < 1:
            raise ValueError("patience and cpu_threads must be positive.")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least one.")

    def to_jsonable(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        payload["output_dir"] = str(self.output_dir)
        payload["fixed_config_path"] = str(self.fixed_config_path)
        return payload


@dataclass
class FilterData:
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


def causal_kalman_filter(
    values: np.ndarray,
    calibration_values: np.ndarray,
    process_variance_ratio: float,
    measurement_variance_ratio: float,
) -> np.ndarray:
    """Run one scalar Kalman filter per feature without future observations."""

    calibration_variance = float(np.var(calibration_values))
    calibration_variance = max(calibration_variance, 1e-12)
    process_variance = max(process_variance_ratio * calibration_variance, 1e-12)
    measurement_variance = max(
        measurement_variance_ratio * calibration_variance,
        1e-12,
    )
    filtered = np.empty_like(values, dtype=np.float64)
    state = float(values[0])
    covariance = measurement_variance
    for index, measurement in enumerate(values):
        prediction_covariance = covariance + process_variance
        gain = prediction_covariance / (prediction_covariance + measurement_variance)
        state = state + gain * (float(measurement) - state)
        covariance = (1.0 - gain) * prediction_covariance
        filtered[index] = state
    return filtered


def causal_savgol_filter(
    values: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    """Causal Savitzky-Golay smoothing evaluated at each local window endpoint."""

    if window_length % 2 == 0:
        raise ValueError("Savitzky-Golay window_length must be odd.")
    if polyorder >= window_length:
        raise ValueError("Savitzky-Golay polyorder must be less than window_length.")

    filtered = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window_length + 1)
        local_values = values[start:index + 1]
        effective_order = min(polyorder, len(local_values) - 1)
        if effective_order < 1:
            filtered[index] = values[index]
            continue
        local_time = np.arange(len(local_values), dtype=np.float64)
        coefficients = np.polyfit(local_time, local_values, effective_order)
        filtered[index] = np.polyval(coefficients, local_time[-1])
    return filtered


def causal_rolling_average(values: np.ndarray, window_days: int) -> np.ndarray:
    """Causal rolling mean; in daily equities data this represents trading days."""

    return (
        pd.Series(values)
        .rolling(window=window_days, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def apply_feature_filter(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    filter_name: str,
    filter_config: Dict[str, Any],
    calibration_rows: int,
) -> pd.DataFrame:
    """Filter input features causally while preserving the unfiltered target."""

    output = frame.copy()
    for column in feature_columns:
        values = output[column].to_numpy(dtype=np.float64)
        calibration_values = values[:calibration_rows]
        if filter_name == "none":
            filtered = values
        elif filter_name == "kalman":
            filtered = causal_kalman_filter(
                values,
                calibration_values,
                filter_config["process_variance_ratio"],
                filter_config["measurement_variance_ratio"],
            )
        elif filter_name == "savgol":
            filtered = causal_savgol_filter(
                values,
                filter_config["window_length"],
                filter_config["polyorder"],
            )
        elif filter_name == "rolling":
            filtered = causal_rolling_average(
                values,
                filter_config["window_days"],
            )
        else:
            raise ValueError(f"Unsupported filter: {filter_name}")
        output[column] = filtered
    return output


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
        raise ValueError(f"{name} has no valid classification sequences.")
    return DirectionPartition(
        X=np.asarray(sequences, dtype=np.float32).reshape(-1, lookback, n_features),
        y=np.asarray(labels, dtype=np.float32).reshape(-1),
        timestamps=timestamps,
        prior_close=np.asarray(prior_close, dtype=np.float64),
    )


def prepare_filtered_data(
    raw_data: pd.DataFrame,
    config: FilterTuningConfig,
    filter_name: str,
    filter_config: Dict[str, Any],
) -> FilterData:
    """Apply a candidate feature filter, then fit selector/scaler on training only."""

    row_count = len(raw_data)
    test_start = int(row_count * (1.0 - config.holdout_ratio))
    train_end = int(test_start * (1.0 - config.validation_ratio))
    if train_end <= config.lookback:
        raise ValueError("Training partition is too short for the selected lookback.")

    candidate_features = [
        feature for feature in FEATURE_CANDIDATES if feature in raw_data.columns
    ]
    if len(candidate_features) < config.k_features:
        raise ValueError(
            f"Requested {config.k_features} features but found {candidate_features}."
        )

    filtered_data = apply_feature_filter(
        raw_data,
        candidate_features,
        filter_name,
        filter_config,
        train_end,
    )
    close = raw_data["close"].to_numpy(dtype=np.float64)
    directions = (close[1:] > close[:-1]).astype(np.int64)
    features = filtered_data[candidate_features]

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
        bucket["timestamps"].append(str(raw_data.index[target_index]))
        bucket["prior"].append(float(close[target_index - 1]))

    return FilterData(
        train=_as_partition(
            buckets["train"]["X"],
            buckets["train"]["y"],
            buckets["train"]["timestamps"],
            buckets["train"]["prior"],
            config.lookback,
            config.k_features,
            "train",
        ),
        validation=_as_partition(
            buckets["validation"]["X"],
            buckets["validation"]["y"],
            buckets["validation"]["timestamps"],
            buckets["validation"]["prior"],
            config.lookback,
            config.k_features,
            "validation",
        ),
        test=_as_partition(
            buckets["test"]["X"],
            buckets["test"]["y"],
            buckets["test"]["timestamps"],
            buckets["test"]["prior"],
            config.lookback,
            config.k_features,
            "test",
        ),
        selector=selector,
        x_scaler=x_scaler,
        selected_features=selected_features,
        candidate_features=candidate_features,
        raw_rows=row_count,
        date_start=str(raw_data.index[0]),
        date_end=str(raw_data.index[-1]),
    )


def proxy_config(config: FilterTuningConfig) -> SimpleNamespace:
    """Supply the fields shared by the reusable LSTM benchmark trainer."""

    return SimpleNamespace(
        lstm_hidden_size=config.lstm_hidden_size,
        lstm_layers=1,
        lstm_dropout=0.0,
        batch_size=config.batch_size,
        patience=config.patience,
    )


def candidate_grid() -> List[Tuple[str, Dict[str, Any]]]:
    """Return fixed candidate configurations, including the unfiltered reference."""

    candidates: List[Tuple[str, Dict[str, Any]]] = [("none", {})]
    for process_ratio in (1e-4, 1e-3, 1e-2):
        for measurement_ratio in (1e-2, 1e-1, 1.0):
            candidates.append(
                (
                    "kalman",
                    {
                        "process_variance_ratio": process_ratio,
                        "measurement_variance_ratio": measurement_ratio,
                    },
                )
            )
    for window_length in (5, 7, 11, 15):
        for polyorder in (2, 3):
            candidates.append(
                (
                    "savgol",
                    {
                        "window_length": window_length,
                        "polyorder": polyorder,
                    },
                )
            )
    candidates.append(("rolling", {"window_days": 10}))
    return candidates


def ranking_key(result: Dict[str, Any]) -> Tuple[float, float]:
    """Higher validation ROC-AUC wins; lower validation BCE breaks ties."""

    validation_auc = result["validation"]["roc_auc"]
    if validation_auc is None:
        validation_auc = -float("inf")
    return float(validation_auc), -float(result["fit"]["best_validation_loss"])


def select_fixed_configs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose one configuration per filter family exclusively by validation metrics."""

    fixed: Dict[str, Any] = {
        "selection_objective": "maximize validation ROC-AUC; tie-break lower validation BCE",
        "causality": (
            "All transforms use observations available at each timestamp only; "
            "no centered windows or future values are used."
        ),
        "no_filter": {},
        "rolling": {"window_days": 10},
    }
    for filter_name in ("kalman", "savgol"):
        filter_results = [
            result for result in results if result["filter"] == filter_name
        ]
        fixed[filter_name] = max(filter_results, key=ranking_key)["config"]
    return fixed


def render_model_card(
    config: FilterTuningConfig,
    fixed_configs: Dict[str, Any],
    selected_results: List[Dict[str, Any]],
) -> str:
    rows = "\n".join(
        (
            f"- {result['filter']}: validation ROC-AUC "
            f"{result['validation']['roc_auc']:.4f}, test ROC-AUC "
            f"{result['test']['roc_auc']:.4f}, test balanced accuracy "
            f"{result['test']['balanced_accuracy']:.4f}"
        )
        for result in selected_results
    )
    return f"""# Causal feature-filter tuning

## Target and objective
Filters are tuned with validation ROC-AUC from a fixed LSTM next-bar
direction-classification proxy. Test metrics are reported only after the
configuration is selected.

## Data
- Ticker label: `{config.ticker}`
- Source: `{config.data_path}`
- Lookback: `{config.lookback}`
- Features selected after filtering: `{config.k_features}`

## Fixed configurations
```json
{json.dumps(fixed_configs, indent=2)}
```

## Selected per-filter results
{rows}

## Leakage control
Kalman, Savitzky-Golay, and rolling-average calculations are causal. Each
value uses only its current and prior feature observations.
"""


def package_artifact_bundle(
    final_directory: Path,
    temporary_directory: Path,
    config: FilterTuningConfig,
    raw_data: pd.DataFrame,
    results: List[Dict[str, Any]],
    fixed_configs: Dict[str, Any],
) -> Path:
    """Write all tuning candidates and fixed filter configurations atomically."""

    shutil.copy2(Path(__file__), temporary_directory / "tune_feature_filters.py")
    shutil.copy2(
        Path(__file__).resolve().with_name("run_direction_classification_benchmark.py"),
        temporary_directory / "run_direction_classification_benchmark.py",
    )
    write_json(
        temporary_directory / "filter_tuning_results.json",
        {"results": results},
    )
    write_json(temporary_directory / "fixed_feature_filter_configs.json", fixed_configs)
    pd.DataFrame(
        [
            {
                "Filter": result["filter"],
                "Config": json.dumps(result["config"], sort_keys=True),
                "ValidationROCAUC": result["validation"]["roc_auc"],
                "ValidationBalancedAccuracy": result["validation"]["balanced_accuracy"],
                "TestROCAUC": result["test"]["roc_auc"],
                "TestBalancedAccuracy": result["test"]["balanced_accuracy"],
                "TestAccuracy": result["test"]["accuracy"],
                "TrainingSeconds": result["training_seconds"],
                "SelectedFeatures": ", ".join(result["selected_features"]),
            }
            for result in results
        ]
    ).to_csv(temporary_directory / "filter_tuning_results.csv", index=False)
    write_json(
        temporary_directory / "filter_tuning_config.json",
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "config": config.to_jsonable(),
            "data": {
                "raw_rows_after_indicators": len(raw_data),
                "date_start": str(raw_data.index[0]),
                "date_end": str(raw_data.index[-1]),
            },
            "runtime": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "torch": torch.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )
    selected_results = [
        result
        for result in results
        if (
            (result["filter"] == "none")
            or (
                result["filter"] in ("kalman", "savgol", "rolling")
                and result["config"] == fixed_configs[result["filter"]]
            )
        )
    ]
    (temporary_directory / "model_card.md").write_text(
        render_model_card(config, fixed_configs, selected_results),
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune causal Kalman, Savitzky-Golay, and rolling feature filters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR.parent / "feature_filter_tuning",
    )
    parser.add_argument(
        "--fixed-config-path",
        type=Path,
        default=PROJECT_ROOT / "config" / "fixed_feature_filter_configs.json",
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--k-features", type=int, default=3)
    parser.add_argument("--lstm-hidden-size", type=int, default=16)
    parser.add_argument("--holdout-ratio", type=float, default=0.20)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--proxy-epochs", type=int, default=12)
    parser.add_argument("--proxy-lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=19,
        help="Cap sequential candidates for a resource-bounded partial sweep.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FilterTuningConfig:
    return FilterTuningConfig(
        data_path=args.data_path.resolve(),
        output_dir=args.output_dir.resolve(),
        fixed_config_path=args.fixed_config_path.resolve(),
        ticker=args.ticker.upper(),
        lookback=args.lookback,
        k_features=args.k_features,
        lstm_hidden_size=args.lstm_hidden_size,
        holdout_ratio=args.holdout_ratio,
        validation_ratio=args.validation_ratio,
        batch_size=args.batch_size,
        proxy_epochs=args.proxy_epochs,
        proxy_lr=args.proxy_lr,
        patience=args.patience,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
        max_candidates=args.max_candidates,
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
    logger.info("Starting causal feature-filter tuning.")
    logger.info("Configuration: %s", json.dumps(config.to_jsonable(), default=str))

    try:
        raw = pd.read_parquet(config.data_path).dropna()
        raw_data = add_technical_indicators(canonicalize_ohlcv_columns(raw))
        candidates = candidate_grid()[:config.max_candidates]
        logger.info("Evaluating %s filter candidates sequentially.", len(candidates))

        results: List[Dict[str, Any]] = []
        device = torch.device("cpu")
        proxy = proxy_config(config)
        for candidate_index, (filter_name, filter_config) in enumerate(candidates):
            # Hold model initialization constant so validation differences come
            # from the filter configuration rather than random-seed variation.
            set_reproducible_seed(config.seed, config.cpu_threads)
            data = prepare_filtered_data(
                raw_data,
                config,
                filter_name,
                filter_config,
            )
            model = LSTMClassifier(config.k_features, proxy).to(device)
            started = time.perf_counter()
            fit = fit_classifier(
                model,
                data.train,
                data.validation,
                config.proxy_epochs,
                config.proxy_lr,
                proxy,
                device,
                logger,
                f"{filter_name} candidate {candidate_index + 1}",
            )
            training_seconds = time.perf_counter() - started
            validation_probabilities = predict_probabilities(model, data.validation, device)
            test_probabilities = predict_probabilities(model, data.test, device)
            validation_metrics = classification_metrics(
                data.validation.y.astype(int),
                validation_probabilities,
            )
            test_metrics = classification_metrics(data.test.y.astype(int), test_probabilities)
            result = {
                "filter": filter_name,
                "config": filter_config,
                "validation": validation_metrics,
                "test": test_metrics,
                "fit": fit.to_jsonable(),
                "training_seconds": training_seconds,
                "selected_features": data.selected_features,
            }
            results.append(result)
            logger.info(
                "%s %s: validation ROC-AUC %.4f | test ROC-AUC %.4f | %.2fs",
                filter_name,
                filter_config,
                validation_metrics["roc_auc"],
                test_metrics["roc_auc"],
                training_seconds,
            )

        expected_filters = {"none", "kalman", "savgol", "rolling"}
        completed_filters = {result["filter"] for result in results}
        if not expected_filters.issubset(completed_filters):
            raise RuntimeError(
                "Candidate cap excluded one or more required filters. "
                "Increase --max-candidates to at least 19."
            )
        fixed_configs = select_fixed_configs(results)
        config.fixed_config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(config.fixed_config_path, fixed_configs)

        close_artifact_file_handlers(logger)
        artifact_path = package_artifact_bundle(
            final_directory,
            temporary_directory,
            config,
            raw_data,
            results,
            fixed_configs,
        )
        logger.info("Feature-filter tuning artifact saved to %s", artifact_path)
        logger.info("Fixed configurations written to %s", config.fixed_config_path)
        print(f"Tuning artifact saved: {artifact_path}")
        print(f"Fixed configurations: {config.fixed_config_path}")
        return 0
    except Exception:
        logger.exception(
            "Filter tuning failed. The temporary directory is retained for diagnosis: %s",
            temporary_directory,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
