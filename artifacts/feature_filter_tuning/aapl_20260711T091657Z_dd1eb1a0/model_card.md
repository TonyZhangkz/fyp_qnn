# Causal feature-filter tuning

## Target and objective
Filters are tuned with validation ROC-AUC from a fixed LSTM next-bar
direction-classification proxy. Test metrics are reported only after the
configuration is selected.

## Data
- Ticker label: `AAPL`
- Source: `E:\fyp_qnn\data\yfinance\AAPL.parquet`
- Lookback: `5`
- Features selected after filtering: `3`

## Fixed configurations
```json
{
  "selection_objective": "maximize validation ROC-AUC; tie-break lower validation BCE",
  "causality": "All transforms use observations available at each timestamp only; no centered windows or future values are used.",
  "no_filter": {},
  "rolling": {
    "window_days": 10
  },
  "kalman": {
    "process_variance_ratio": 0.001,
    "measurement_variance_ratio": 1.0
  },
  "savgol": {
    "window_length": 5,
    "polyorder": 3
  }
}
```

## Selected per-filter results
- none: validation ROC-AUC 0.4994, test ROC-AUC 0.5717, test balanced accuracy 0.5255
- kalman: validation ROC-AUC 0.5553, test ROC-AUC 0.5184, test balanced accuracy 0.5017
- savgol: validation ROC-AUC 0.5322, test ROC-AUC 0.5195, test balanced accuracy 0.5000
- rolling: validation ROC-AUC 0.4966, test ROC-AUC 0.5833, test balanced accuracy 0.5347

## Leakage control
Kalman, Savitzky-Golay, and rolling-average calculations are causal. Each
value uses only its current and prior feature observations.
