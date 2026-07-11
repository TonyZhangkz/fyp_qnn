# HybridQNN2 tuning report

Run date: 2026-07-11  
Data: AAPL parquet, 976 rows after technical-indicator preparation  
Evaluation: chronological train/validation/test split; three selected features and three exact-simulator qubits

## Decision

Do not promote a HybridQNN2 artifact from this sweep. Joint optimization
substantially improved on HybridQNN1, but the best strict result did not beat
the 4.328472 USD last-close baseline on the untouched test partition.

## Baseline joint model

The initial staging configuration used lookback 2, LSTM width 8, fusion width
8, one quantum layer, joint learning rate 0.005, and at most 20 epochs.

- Joint training selected epoch 2 of 7 executed epochs.
- Validation price RMSE: 7.632899.
- Holdout test price RMSE: 23.046721.
- Last-close holdout price RMSE: 4.328472.
- Relative error: 5.32 times the naive baseline RMSE.
- Staging gate: rejected.

Artifact: `artifacts/hybridqnn2/aapl_20260711T083522Z_aafb6e99`

## Tuning runs

The comparison-only runs used `--no-refit-on-train-validation` and
`--allow-underperforming` to collect comparable metrics without treating a
failed run as promotable.

### Lower joint learning rate with longer training

Configuration: lookback 2, LSTM width 8, fusion width 8, one quantum layer,
joint learning rate 0.001, 40 epochs, and patience 10.

- Joint training selected epoch 40 of 40.
- Validation price RMSE: 5.054970.
- Test price RMSE without refitting: 13.928912.
- Selection-training time: 257.69 seconds.
- Result: a large improvement over the initial joint model, but still 3.22
  times the naive baseline RMSE.

Artifact: `artifacts/hybridqnn2_tuning/low_lr_long/aapl_20260711T083641Z_96c59cf8`

### Longer temporal context and larger fusion capacity

Configuration: lookback 5, LSTM width 16, fusion width 16, one quantum layer,
joint learning rate 0.001, 40 epochs, and patience 10.

- Joint training selected epoch 40 of 40.
- Validation price RMSE: 4.814062.
- Test price RMSE without refitting: 12.015996.
- Selection-training time: 262.02 seconds.
- Result: the best comparison-only setting, but still 2.78 times the naive
  baseline RMSE.

Artifact: `artifacts/hybridqnn2_tuning/lookback5_capacity/aapl_20260711T084123Z_6f0b6770`

### Final strict refit of the best screened configuration

The lookback-5 / hidden-16 / fusion-16 setting was retrained jointly on train
plus validation, then evaluated with the strict promotion gate.

- Holdout test price RMSE: 12.167298.
- Last-close holdout price RMSE: 4.328472.
- Relative error: 2.81 times the naive baseline RMSE.
- Staging gate: rejected.
- Total training and refit time before packaging: 544.91 seconds.

Artifact: `artifacts/hybridqnn2_tuning/final_lookback5_capacity/aapl_20260711T084603Z_02865fea`

## Findings

1. Joint HybridQNN2 is materially better than the sequential HybridQNN1
   reproduction. The best strict QNN2 test RMSE of 12.167298 is about 59.6%
   lower than HybridQNN1's best strict RMSE of 30.100916.
2. Learning rate was the main useful tuning lever. At 0.005, validation
   performance peaked at epoch 2. At 0.001, validation performance continued
   improving through epoch 40.
3. Increasing lookback from 2 to 5 and increasing LSTM/fusion width from
   8 to 16 improved screening RMSE from 13.928912 to 12.015996.
4. Train-plus-validation refitting changed the best setting's test RMSE only
   slightly, from 12.015996 to 12.167298. It did not close the gap to the
   naive baseline.
5. Each full 40-epoch screening run used roughly 4.3 minutes of CPU time;
   the strict selected-model refit used about 9.1 minutes. Further
   hyperparameter-only sweeps should not be prioritized until the modeling
   target and validation design are improved.

## Recommended next work

- Keep the promotion gate enabled. No QNN2 artifact from this sweep is fit for
  production deployment.
- Establish a classical LSTM-only benchmark using the same split and
  walk-forward folds. The last-close baseline must remain the required
  comparator.
- Test returns or price deltas rather than raw close prices. Raw price levels
  make persistence exceptionally strong and can obscure incremental value.
- Add a proper rolling TimeSeriesSplit evaluation that refits selection and
  scalers inside each fold before considering another artifact for promotion.
- Only after the classical benchmark is established, consider feature
  enrichment or a residual connection between the LSTM forecast and the
  quantum branch. Those are architecture changes, not more epoch tuning.
