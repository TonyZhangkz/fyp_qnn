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
    "measurement_variance_ratio": 0.01
  },
  "savgol": {
    "window_length": 15,
    "polyorder": 2
  }
}
```

## Selected per-filter results
- none: validation ROC-AUC 0.4994, test ROC-AUC 0.5717, test balanced accuracy 0.5255
- kalman: validation ROC-AUC 0.5307, test ROC-AUC 0.5255, test balanced accuracy 0.5084
- savgol: validation ROC-AUC 0.5070, test ROC-AUC 0.5434, test balanced accuracy 0.5255
- rolling: validation ROC-AUC 0.5147, test ROC-AUC 0.5770, test balanced accuracy 0.5173

## Leakage control
Kalman, Savitzky-Golay, and rolling-average calculations are causal. Each
value uses only its current and prior feature observations.
