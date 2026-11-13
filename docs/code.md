# Code notes

The `code/` directory contains the research script packages supplied with the project, organized by model. Model-specific documentation and configuration files remain alongside their scripts. The `GIN` package contains GINE experiments; the `ML` package includes classical machine-learning and similarity baselines. Scripts for classification, regression, diagnostics, and result summaries are present in different packages.

These files are a research snapshot, not a verified installation or execution guide. The scripts require user-configured paths, external repositories, checkpoints, schedulers, GPUs, and conda environments. Set `DATA_ROOT` to prepared splits where required; consult each model's documentation and review scheduler settings before use. Outputs and internal execution manifests may include local paths and should be reviewed before sharing. Some models or checkpoints may have separate access requirements or licenses.

To reduce accidental disclosure, some scripts omit machine paths, host names, and detailed exception messages from their logs and metadata. This can make troubleshooting less specific. Third-party packages and scheduler output may still expose local information; inspect files before posting them publicly.

Several workflows expect separately prepared datasets with classification and regression directories, split methods, seed directories, and `train.csv`, `validation.csv`, and `test.csv` files. The supplied full genus-level datasets are not a replacement for that layout. No prepared split files, trained models, or third-party model implementations are supplied here. Consult the package-specific notes for the inputs expected by each script; they have not been validated against the current repository.

No execution or reproduction of the manuscript results is guaranteed. For problems, please [open an issue](https://github.com/zhaisilong/AntibioticBench/issues) or email zhaisilong@outlook.com. We will do our best to help.
