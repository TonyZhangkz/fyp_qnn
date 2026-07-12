# Binary direction-classification benchmark

## Target
Up is 1 when the close at the prediction timestamp exceeds the close at the
last timestep in the input sequence. Down or unchanged is 0.

## Data and split
- Ticker label: `AAPL`
- Source: `E:\fyp_qnn\data\yfinance\AAPL.parquet`
- Range: `2022-02-09 00:00:00` to `2025-12-30 00:00:00`
- Selected features: `open, high, low`
- Lookback: `2`
- Chronological split with no shuffle; hyperparameters were fixed before this run.

## Test results
- LSTM baseline: accuracy 0.5000, F1-up 0.6667, ROC-AUC 0.5, training time 1.07s
- HybridQNN1: accuracy 0.5000, F1-up 0.0000, ROC-AUC 0.875, training time 1.18s
- HybridQNN2: accuracy 0.5000, F1-up 0.6667, ROC-AUC 0.59375, training time 1.31s

## Interpretation
This is a single-split research benchmark, not a production promotion result.
Use walk-forward folds and transaction-cost-aware evaluation before making any
trading or deployment decision.
