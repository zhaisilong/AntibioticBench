# Test-only diagnostics update

Split-specific output policy:

- train: per-molecule predictions and overall metrics only
- validation: per-molecule predictions and overall metrics only
- test: per-molecule predictions, overall metrics, nearest-neighbor diagnostics, and activity-cliff diagnostics

Implementation changes:

1. `predictions_train.csv` and `predictions_validation.csv` no longer calculate or store `nn_*` or `ac_*` fields.
2. `predictions_test.csv` retains train-reference NN fields and AC fields.
3. `diagnostics_metrics.csv` and `diagnostic_subgroup_metrics.csv` contain test-only NN and AC results.
4. Pairwise AC postprocessing is restricted to `test_test`.
5. Automatic landscape-roughness postprocessing was removed from this GINE workflow.
6. The validator rejects train/validation outputs containing `nn_*` or `ac_*` fields.
7. `strict_perimeter` remains restricted to `seed_0`.
8. Fixed GINE hyperparameters remain unchanged: 5 layers, hidden/MLP dimension 300, dropout 0.20, learning rate 5e-4, weight decay 1e-5, mean pooling, 300 epochs, patience 40.
