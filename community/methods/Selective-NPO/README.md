# Selective-NPO

- Repository-local adaptation of the selective curriculum idea applied to NPO.
- Method base: OpenUnlearning TOFU unlearning with a difficulty-aware, staged forget schedule.
- Selective data ordering is built from held-out reference folds, then used to train staged unlearning runs from easy to hard.
- Optional control: set `INTRA_STAGE_ORDER=difficulty_strict` to keep stage coverage unchanged while forcing each stage epoch to traverse forget samples in easy-to-hard difficulty order.

# Setup

- Hyperparameters:
  - TOFU `forget10` / `retain90`
  - `beta=0.1`
  - `stage_percentiles=[0.3, 0.6, 1.0]`
  - `stage_epoch_ratios=[0.3, 0.3, 0.4]`
  - `num_folds=4`
  - `per_device_train_batch_size=16`
  - `gradient_accumulation_steps=4`
- Computation setup:
  - Designed for a single GPU in the provided script.
- Workflow:
  - build reference folds
  - train reference models
  - score difficulty
  - stage forget samples
  - run selective unlearning
  - optional strict-order control inside each stage via `INTRA_STAGE_ORDER`

# Results

Run [`run.sh`](./run.sh). Leave `INTRA_STAGE_ORDER=random` for the current baseline, or set `INTRA_STAGE_ORDER=difficulty_strict` for the strict intra-stage ordering control.

- Resume behavior:
  - `RESUME=true` by default and skips completed reference fold training, difficulty scoring, finished stage training, and final evals.
  - If a stage was interrupted after checkpoints were written, rerunning `run.sh` resumes that stage from its latest checkpoint automatically.
  - Set `RESUME=false` to rebuild the workflow from scratch with the current script settings.

# Citation

This is a repository-local adaptation rather than a separate paper artifact. If you use it, please cite the OpenUnlearning technical report and the Selective DPO paper that motivates the curriculum idea.
