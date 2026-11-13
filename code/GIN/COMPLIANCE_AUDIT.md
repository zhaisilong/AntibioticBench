# GINE output policy audit

Current policy:

- train: per-molecule predictions and overall metrics only
- validation: per-molecule predictions and overall metrics only
- test: per-molecule predictions, overall metrics, train-reference nearest-neighbor diagnostics, and activity-cliff diagnostics

`validate_gine_outputs.py --strict` checks that train/validation do not contain `nn_*` or `ac_*` columns and that test contains the required diagnostic columns. Pairwise AC postprocessing is restricted to `test_test`.
