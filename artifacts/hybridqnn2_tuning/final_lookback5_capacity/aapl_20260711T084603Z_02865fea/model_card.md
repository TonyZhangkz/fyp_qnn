# HybridQNN2 model card

## Status
`rejected`

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
- Ticker label: `AAPL`
- Source: `E:\fyp_qnn\data\yfinance\AAPL.parquet`
- Data range: `2022-02-09 00:00:00` to `2025-12-30 00:00:00`
- Selected features: `open, high, low`
- Lookback: `5`

## Evaluation
- Validation price RMSE: `4.814062`
- Holdout price RMSE: `12.167298`
- Last-close holdout price RMSE: `4.328472`
- Holdout price R²: `0.828466`

## Staging gate
- Model price RMSE did not beat the last-close naive baseline.
