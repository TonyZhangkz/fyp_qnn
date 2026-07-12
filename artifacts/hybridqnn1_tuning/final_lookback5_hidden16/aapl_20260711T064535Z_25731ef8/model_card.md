# HybridQNN1 model card

## Status
`rejected`

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
- Ticker label: `AAPL`
- Source: `E:\fyp_qnn\data\yfinance\AAPL.parquet`
- Data range: `2022-02-09 00:00:00` to `2025-12-30 00:00:00`
- Selected features: `open, high, low`
- Lookback: `5`

## Evaluation
- Validation price RMSE: `41.941880`
- Holdout price RMSE: `30.794014`
- Last-close holdout price RMSE: `4.328472`
- Holdout price R²: `-0.098737`

## Staging gate
- Model price RMSE did not beat the last-close naive baseline.
