# Causal feature-filter tuning report

Run date: 2026-07-11  
Data: AAPL parquet, 976 rows after technical-indicator preparation  
Proxy objective: validation ROC-AUC from a fixed LSTM direction-classification proxy  
Leakage control: all filters are causal and use observations available at the
current timestamp or earlier only

## Fixed configurations

```json
{
  "kalman": {
    "process_variance_ratio": 0.001,
    "measurement_variance_ratio": 0.01
  },
  "savgol": {
    "window_length": 15,
    "polyorder": 2
  },
  "rolling": {
    "window_days": 10
  }
}
```

The fixed configuration is saved in
`config/fixed_feature_filter_configs.json`. The rolling window represents ten
trading observations in this daily market dataset.

## Selection grid

```text
┌───────────┬───────────────────────────────────────┬────────────┬──────────┬────────────────┬──────────────────────┐
│ Filter    │ Fixed configuration                   │ Val ROC-AUC│ Test AUC │ Test Bal. Acc. │ Proxy Training (sec) │
├───────────┼───────────────────────────────────────┼────────────┼──────────┼────────────────┼──────────────────────┤
│ None      │ no smoothing                          │ 0.4994     │ 0.5717   │ 0.5255         │ 2.07                 │
│ Kalman    │ q=0.001, r=0.01 (variance ratios)     │ 0.5307     │ 0.5255   │ 0.5084         │ 1.00                 │
│ SavGol    │ causal window=15, polynomial order=2  │ 0.5070     │ 0.5434   │ 0.5255         │ 0.95                 │
│ Rolling   │ causal 10-trading-day mean            │ 0.5147     │ 0.5770   │ 0.5173         │ 1.07                 │
└───────────┴───────────────────────────────────────┴────────────┴──────────┴────────────────┴──────────────────────┘
```

## Method

1. Candidate filters were applied to all numeric candidate features after
   technical-indicator construction and before feature selection/scaling.
2. Feature selection (`SelectKBest(f_classif)`) and MinMax scaling were fit
   using training rows only.
3. Every candidate used the same LSTM initialization seed, data split, and
   proxy hyperparameters. This isolates filter configuration differences from
   random initialization.
4. Kalman candidates tuned dimensionless process and measurement variance
   ratios against each feature's training variance.
5. Savitzky-Golay candidates used a causal endpoint polynomial fit rather than
   the usual centered implementation, avoiding future-data leakage.
6. The test partition was not used to choose a configuration.

## Findings

1. The fixed Kalman configuration achieved the highest validation ROC-AUC
   (0.5307), so it is the correct validation-selected Kalman configuration.
   Its test AUC did not generalize (0.5255), which shows why the selection must
   not use test metrics.
2. The fixed Savitzky-Golay configuration was window 15 and polynomial order
   2. Its validation AUC was modest (0.5070), and it did not outperform the
   no-filter test reference.
3. The required rolling 10-day average had the highest observed test AUC
   (0.5770), slightly above the unfiltered proxy (0.5717), but it was not
   selected on test data and needs confirmation with walk-forward folds.
4. Effects are small relative to the uncertainty of a single temporal split.
   These values are fixed preprocessing candidates, not evidence that any
   filter is universally superior.

## Use

Run the tuner again with:

```powershell
conda run -n env_qubit python scripts\tune_feature_filters.py
```

The reusable implementations and causal filtering entry point are in
`scripts/tune_feature_filters.py`. Apply one fixed configuration before
feature selection and scaling in later model benchmarks.

## Reproducibility

- Tuning script: `scripts/tune_feature_filters.py`
- Tuning artifact:
  `artifacts/feature_filter_tuning/aapl_20260711T091753Z_0a8cfbcc`
- Full candidate grid:
  `filter_tuning_results.csv` inside the saved artifact
