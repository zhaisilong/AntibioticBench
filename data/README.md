# Benchmark data

The complete benchmark is available as [`AntibioticBench-data.zip`](AntibioticBench-data.zip). It contains 12 genus-level CSV files with 269,072 rows in total, a file inventory, and checksums. The archive is approximately 24 MiB; the CSV files occupy approximately 274 MiB when extracted.

To unpack it and verify the contents:

```bash
unzip data/AntibioticBench-data.zip -d antibioticbench-data
cd antibioticbench-data
sha256sum -c SHA256SUMS
```

The manuscript describes genus-level minimum inhibitory concentration (MIC) records curated from ChEMBL, PubChem, SPARK, CO-ADD, NPASS, and ChemDiv. Records were aggregated at the compound–genus level and may combine measurements from different strains or assay conditions. The CSV files contain source and identifier columns alongside chemical structures, MIC values, benchmark labels, molecular descriptors, and data-quality annotations. The `source` field should not be assumed to be an exhaustive record of the provenance of every aggregated measurement.

The data are subject to the applicable terms of their original sources. The repository's academic license applies to project-authored code and documentation and does not replace third-party data terms; consult the original providers for reuse conditions.

These genus-level tables are not the prepared training, validation, and test splits expected by many of the scripts. Values derived using the original full dataset, including nearest-neighbor indices and similarities, should not be assumed to describe a filtered subset or a new train/test split.
