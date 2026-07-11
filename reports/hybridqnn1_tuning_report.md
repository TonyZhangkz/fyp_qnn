# HybridQNN1 tuning report

Run date: 2026-07-11  
Data: AAPL parquet, 976 rows after technical-indicator preparation  
Evaluation: chronological train/validation/test split; 3 selected features and 3 exact-simulator qubits

## Decision

Do not promote any HybridQNN1 artifact from this sweep. Both strict staging
runs were rejected because they did not beat the last-close baseline on the
untouched test partition.

## Baseline

The initial staging configuration used lookback 2, LSTM width 8, one quantum
layer, quantum learning rate 0.005, maximum 20 epochs per stage, and
train-plus-validation refitting.

- Stage 1 LSTM validation RMSE: 0.046265 scaled, selected at epoch 5.
- Stage 2 QNN validation RMSE: 0.390845 scaled, selected at epoch 5.
- Holdout test price RMSE: 30.100916.
- Last-close holdout price RMSE: 4.328472.
- Relative error: HybridQNN1 was 6.96 times the naive baseline RMSE.
- Staging gate: rejected.

Artifact: `artifacts/hybridqnn1/aapl_20260711T063415Z_85a35bff`

## Tuning runs

The comparison-only runs used `--no-refit-on-train-validation` and
`--allow-underperforming` so their metrics could be collected without falsely
promoting them. Their acceptance status is not a production decision.

### Longer one-layer QNN

Configuration: one quantum layer, quantum learning rate 0.001, maximum 60
quantum epochs, patience 15.

- Stage 2 selected epoch 30 of 45 executed epochs.
- Stage 2 validation RMSE: 0.500484 scaled.
- Test price RMSE: 62.355941.
- Selection-training time: 250.64 seconds.
- Result: worse than the baseline and 14.41 times the naive baseline RMSE.

Artifact: `artifacts/hybridqnn1_tuning/longer_q1/aapl_20260711T063720Z_7fd84793`

### Two-layer QNN

Configuration: two quantum layers, quantum learning rate 0.003, maximum 20
quantum epochs, patience 5.

- Stage 2 selected epoch 4 of 9 executed epochs.
- Stage 2 validation RMSE: 0.448641 scaled.
- Test price RMSE: 57.242965.
- Selection-training time: 76.71 seconds.
- Result: worse than the one-layer baseline and 13.22 times the naive baseline RMSE.

Artifact: `artifacts/hybridqnn1_tuning/depth2_q003/aapl_20260711T064151Z_8b823758`

### Longer lookback and wider LSTM

Configuration: lookback 5, LSTM width 16, one quantum layer, quantum learning
rate 0.005, maximum 30 pretraining epochs and 20 quantum epochs.

Comparison-only result:

- Stage 1 validation RMSE: 0.039664 scaled, selected at epoch 2.
- Stage 2 validation RMSE: 0.381390 scaled, selected at epoch 2.
- Test price RMSE without refitting: 50.871696.
- Result: the best screened validation score, but still 11.75 times the naive
  baseline RMSE on the comparison-only test.

Artifact: `artifacts/hybridqnn1_tuning/lookback5_hidden16/aapl_20260711T064327Z_0c031cf4`

### Final strict refit of the best screened configuration

The lookback-5 / hidden-16 candidate was retrained on train plus validation
using its selected two epochs per stage, then evaluated with the strict
promotion gate.

- Holdout test price RMSE: 30.794014.
- Last-close holdout price RMSE: 4.328472.
- Relative error: 7.11 times the naive baseline RMSE.
- Staging gate: rejected.

Artifact: `artifacts/hybridqnn1_tuning/final_lookback5_hidden16/aapl_20260711T064535Z_25731ef8`

## Findings

1. More epochs did not improve this model. The longest run selected epoch 30,
   then degraded; it also consumed more than four minutes of selection
   training.
2. Increasing circuit depth from one to two layers degraded validation and
   test performance.
3. A longer lookback and wider LSTM improved the classical Stage 1 validation
   representation, but the frozen quantum Stage 2 still discarded most of
   that advantage.
4. Across the strongest configurations, Stage 1 validation RMSE was roughly
   0.04 scaled while Stage 2 was roughly 0.38 to 0.39 scaled. The sequential
   QNN regressor is the observed bottleneck, not a lack of LSTM training.
5. The last-close baseline is exceptionally strong for this price-level
   target. A production candidate must beat it under walk-forward validation,
   not merely fit the training series.

## Recommended next work

- Keep the staging gate enabled and do not deploy any bundle from this sweep.
- Establish a classical LSTM-only benchmark using identical leakage-safe,
  walk-forward folds before continuing QNN tuning.
- Treat a residual or fusion connection that preserves the LSTM prediction,
  or joint end-to-end training, as a separate architecture change. Those
  changes are closer to HybridQNN2 than the sequential HybridQNN1 defined
  here.
- Consider forecasting returns or deltas rather than raw close prices; keep
  the last-close baseline as the required comparator.
- After an architecture change, run rolling TimeSeriesSplit evaluation with
  per-fold selection/scaling and a fixed promotion threshold before producing
  another production candidate.
