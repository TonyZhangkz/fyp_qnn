# Final 3D model × filter × target benchmark

Run date: 2026-07-11  
Data: AAPL parquet, 976 rows after technical-indicator preparation  
Dimensions: 3 models × 3 causal filters × 2 prediction targets = 18 runs  
Split: chronological 619 / 156 / 196 train / validation / test sequences  
Fixed filters: vanilla, causal Savitzky-Golay (window 15, order 2), and causal
rolling average (10 trading days)

## Binary next-bar direction grid

Target: `up = 1` when `close[t] > close[t - 1]`; otherwise `down = 0`.  
Metrics are computed on the same 196-observation test partition.

```text
┌────────────────┬──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────────────────────────┐
│ Model          │ Vanilla              │ Savitzky-Golay       │ Rolling 10-day       │ Training Time (seconds)                  │
│                │ AUC / Bal. Accuracy │ AUC / Bal. Accuracy  │ AUC / Bal. Accuracy  │ Vanilla / SavGol / Rolling               │
├────────────────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────────────────────────┤
│ LSTM baseline  │ 0.5717 / 0.5255      │ 0.5003 / 0.4987      │ 0.5838 / 0.5116      │ 2.07 / 0.64 / 0.93                       │
│ HybridQNN1     │ 0.4394 / 0.5000      │ 0.4614 / 0.5000      │ 0.5633 / 0.5000      │ 33.97 / 36.68 / 22.48                    │
│ HybridQNN2     │ 0.5317 / 0.5328      │ 0.4631 / 0.5179      │ 0.5638 / 0.5000      │ 51.06 / 40.35 / 26.35                    │
└────────────────┴──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────────────────────────┘
```

Best binary ROC-AUC: **LSTM baseline + rolling average = 0.5838**.

HybridQNN1 reported 55.10% accuracy for each filter, but its balanced accuracy
was exactly 50.00% because it classified every test observation as up. Its
accuracy must not be interpreted as directional skill.

## Continuous next-bar close grid

Target: the unfiltered close price at the next timestamp after the input
window. Lower RMSE and MAE are better; higher R² is better.

```text
┌────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┬──────────────────────────────────────────┐
│ Model          │ Vanilla                 │ Savitzky-Golay          │ Rolling 10-day          │ Training Time (seconds)                  │
│                │ RMSE / R²               │ RMSE / R²               │ RMSE / R²               │ Vanilla / SavGol / Rolling               │
├────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼──────────────────────────────────────────┤
│ LSTM baseline  │ 14.5994 / 0.7530        │ 13.0651 / 0.8022        │ 18.5822 / 0.5999        │ 1.97 / 1.94 / 0.88                       │
│ HybridQNN1     │ 51.2372 / -2.0418       │ 51.2128 / -2.0389       │ 50.0699 / -1.9048       │ 22.67 / 30.85 / 28.45                    │
│ HybridQNN2     │ 22.2356 / 0.4271        │ 16.2804 / 0.6929        │ 21.6792 / 0.4554        │ 83.27 / 78.52 / 78.16                    │
└────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┴──────────────────────────────────────────┘
```

Best continuous result: **LSTM baseline + Savitzky-Golay = 13.0651 RMSE and
0.8022 R²**.

## Findings

1. The LSTM baseline outperformed both hybrid models on both targets while
   requiring a small fraction of the training time.
2. Causal Savitzky-Golay filtering helped continuous price prediction for the
   LSTM and HybridQNN2. It did not help binary classification.
3. The 10-day rolling filter produced the best binary ROC-AUC for the LSTM,
   but reduced continuous-regression performance.
4. HybridQNN2 was consistently stronger than HybridQNN1 for continuous
   prediction, but it remained slower and less accurate than the LSTM.
5. HybridQNN1 is not useful for this benchmark's binary task because its
   decision threshold collapsed to the majority up class.

## Scope and next step

This final run is a single chronological split with fixed configurations, not
an estimate of production performance. The next defensible experiment is
walk-forward cross-validation with the same 18-combination grid, then a
statistical comparison against the LSTM baseline.

## Reproducibility

- Runner: `scripts/run_final_3d_benchmark.py`
- Fixed filters: `config/fixed_feature_filter_configs.json`
- Saved artifact:
  `artifacts/final_3d_benchmark/aapl_20260711T092610Z_447e522c`
- Machine-readable grids: `binary_grid.csv` and `continuous_grid.csv` in the
  saved artifact
