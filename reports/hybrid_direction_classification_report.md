# Binary up/down direction-classification benchmark

Run date: 2026-07-11  
Data: AAPL parquet, 976 rows after technical-indicator preparation  
Target: `up = 1` when `close[t] > close[t - 1]`; otherwise `down = 0`  
Evaluation: one fixed chronological split with 619 / 156 / 196 train / validation / test sequences

## Experimental setup

All models used the same three selected features (`volume`, `macd`, and
`adx_14`), lookback 5, three qubits where applicable, and a 0.5 classification
threshold. The models were not tuned during this benchmark.

- **LSTM baseline:** 16 hidden units, maximum 20 epochs.
- **HybridQNN1:** LSTM pretraining followed by a frozen classical extractor
  and a one-layer QNN classifier.
- **HybridQNN2:** jointly trained LSTM and one-layer QNN branches with a
  16-unit fusion layer.

The test partition contained 55.10% up observations. Therefore an always-up
classifier would achieve 55.10% accuracy but only 50.00% balanced accuracy.

## Paper-style test-results grid

```text
┌────────────────┬──────────┬───────────────────┬──────────┬─────────┬──────────────────────┐
│ Model          │ Accuracy │ Balanced Accuracy │ F1 (Up)  │ ROC-AUC │ Training Time (sec)  │
├────────────────┼──────────┼───────────────────┼──────────┼─────────┼──────────────────────┤
│ LSTM baseline  │ 0.4949   │ 0.5311            │ 0.2774   │ 0.5702  │ 3.18                 │
│ HybridQNN1     │ 0.5510   │ 0.5000            │ 0.7105   │ 0.4313  │ 39.29                │
│ HybridQNN2     │ 0.5153   │ 0.5455            │ 0.3624   │ 0.5647  │ 45.03                │
└────────────────┴──────────┴───────────────────┴──────────┴─────────┴──────────────────────┘
```

## Interpretation

1. **HybridQNN1 is not a useful classifier in this run.** It predicted up for
   all 196 test observations. Its 55.10% accuracy equals the test up-rate,
   while its balanced accuracy is 50.00% and ROC-AUC is 0.4313.
2. **HybridQNN2 has the highest balanced accuracy** at 0.5455, but its
   ROC-AUC (0.5647) is slightly below the classical LSTM baseline (0.5702).
   This is insufficient evidence of a quantum advantage.
3. **The LSTM baseline is much cheaper.** It trained in 3.18 seconds,
   compared with 39.29 seconds for HybridQNN1 and 45.03 seconds for
   HybridQNN2.
4. This is a single chronological split, not a repeated or walk-forward
   statistical evaluation. The results should be treated as preliminary.

## Conclusion

No hybrid classifier should be promoted from this benchmark. The useful next
step is a walk-forward classification study with per-fold feature selection,
threshold selection only on validation data, and a simple majority-direction
baseline alongside the LSTM, HybridQNN1, and HybridQNN2 models.

## Reproducibility

- Benchmark script: `scripts/run_direction_classification_benchmark.py`
- Saved artifact:
  `artifacts/direction_classification/aapl_20260711T090120Z_bfa4b395`
- Machine-readable grid:
  `classification_results.csv` inside the saved artifact
