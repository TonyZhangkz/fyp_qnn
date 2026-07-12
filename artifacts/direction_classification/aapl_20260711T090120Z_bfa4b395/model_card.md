# Binary direction-classification benchmark

## Target
Up is 1 when the close at the prediction timestamp exceeds the close at the
last timestep in the input sequence. Down or unchanged is 0.

## Data and split
- Ticker label: `AAPL`
- Source: `E:\fyp_qnn\data\yfinance\AAPL.parquet`
- Range: `2022-02-09 00:00:00` to `2025-12-30 00:00:00`
- Selected features: `volume, macd, adx_14`
- Lookback: `5`
- Chronological split with no shuffle; hyperparameters were fixed before this run.

## Test results
- LSTM baseline: accuracy 0.4949, F1-up 0.2774, ROC-AUC 0.5701809764309764, training time 3.18s
- HybridQNN1: accuracy 0.5510, F1-up 0.7105, ROC-AUC 0.4312920875420875, training time 39.29s
- HybridQNN2: accuracy 0.5153, F1-up 0.3624, ROC-AUC 0.5647095959595959, training time 45.03s

## Interpretation
This is a single-split research benchmark, not a production promotion result.
Use walk-forward folds and transaction-cost-aware evaluation before making any
trading or deployment decision.
