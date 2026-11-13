# Shared-cache revision summary

Compared with the previous package:

1. Added one-time per-dataset 3D preprocessing:
   - `precompute_unimol_shared_3d_cache.py`
   - `build_unimol_cache_manifest.py`
   - `submit_unimol_3d_cache_slurm.sh`
   - `unimol_cache_job.slurm`
   - `validate_unimol_shared_cache.py`
   - `run_single_cache_test.sh`

2. Removed all per-run conformer generation from classification/regression training.

3. Cache contents:
   - exact SMILES union from classification and regression splits;
   - Uni-Mol v1 no-H inputs;
   - 11 deterministic conformers;
   - base conformer seed 0, derived RDKit seeds 0–10.

4. Cache reuse:
   - classification and regression;
   - random/scaffold/repeated_stratified_perimeter/strict_perimeter;
   - all training seeds.

5. Multi-conformer policy:
   - train: one cached conformer per molecule per epoch, deterministic cycling;
   - validation: mean prediction over all 11 conformers for checkpoint selection;
   - final train/validation/test: mean prediction over all 11 conformers.

6. Embeddings are not cached. Every run recomputes Uni-Mol embeddings and
   updates the full backbone during end-to-end fine-tuning.

7. CPU controls:
   - cache preprocessing: 16 Slurm CPUs and 15 RDKit worker processes;
   - training: 12 Slurm CPUs, 4 DataLoader workers, one thread per process;
   - oversubscription checks added.

8. Training submission fails fast unless all 12 dataset caches are complete.
