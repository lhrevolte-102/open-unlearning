# Scripts Layout

Scripts are grouped by purpose:

- `scripts/baselines/`
  - Baseline shell entrypoints shipped with the repo, such as `tofu_unlearn.sh`, `tofu_finetune.sh`, and `muse_unlearn.sh`.
- `scripts/tofu/train/`
  - TOFU training entrypoints maintained in this workspace.
  - Currently includes staged original baselines: `tofu_original_npo.py` and `tofu_original_dpo.py`.
- `scripts/tofu/eval/`
  - Standalone TOFU evaluation entrypoints.
  - Includes `tofu_eval_common.py` plus eval launchers for original, selective, and `MRD-NPO`.
- `scripts/tofu/slurm/`
  - Cluster submission wrappers for TOFU experiments.
  - Currently includes `submit_selective_npo.sh` and `submit_selective_dpo.sh`.
- `scripts/maintenance/`
  - One-off migration or cleanup utilities.
  - Currently includes `migrate_tofu_staging_artifacts.py`.

Typical commands now look like:

```bash
python scripts/tofu/train/tofu_original_npo.py
python scripts/tofu/eval/tofu_eval_original_npo.py
bash scripts/tofu/slurm/submit_selective_npo.sh
bash scripts/baselines/tofu_unlearn.sh
python scripts/maintenance/migrate_tofu_staging_artifacts.py
```
