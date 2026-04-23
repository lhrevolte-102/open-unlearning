# Selective-DPO

- Selective curriculum variant of IdkDPO on TOFU.
- Method base: OpenUnlearning TOFU unlearning with a difficulty-aware, staged forget schedule.
- Selective data ordering is built from held-out reference splits, then used to train staged unlearning runs from easy to hard.
- Optional control: set `INTRA_STAGE_ORDER=strict` to keep stage coverage unchanged while forcing each stage epoch to traverse forget samples in easy-to-hard difficulty order.

# Setup

- Hyperparameters:
  - TOFU `forget10` / `retain90`
  - `beta=0.1`
  - `stage_percentiles=[0.3, 0.6, 1.0]`
  - `stage_epoch_ratios=[0.3, 0.3, 0.4]`
  - stage unlearning logs to TensorBoard every optimizer step by default via `STAGE_LOGGING_STEPS=1`
  - repeated random halving for reference difficulty estimation
  - `NUM_REFERENCE_REPEATS=3` by default, corresponding to 3 random partitions and 6 reference models
  - `per_device_train_batch_size=16`
  - `gradient_accumulation_steps=4`
- Computation setup:
  - Designed for a single GPU in the provided script.
- Workflow:
  - build reference splits
  - train reference models
  - score difficulty
  - stage forget samples
  - run selective unlearning
  - optional strict-order control inside each stage via `INTRA_STAGE_ORDER`

# Results

Run [`run.py`](./run.py). The script supports a single intra-stage ordering mode at a time and defaults to `INTRA_STAGE_ORDER=random`. Set `INTRA_STAGE_ORDER=strict` to run the strict intra-stage ordering control.

- Stage subsets:
  - `STAGE_PERCENTILES` are interpreted as cumulative difficulty thresholds, e.g. `[0.3, 0.6, 1.0]`.

- Output layout:
  - Reference artifacts live under `saves/selective/reference/${REFERENCE_TASK_NAME}`.
  - Difficulty scores live under `saves/selective/prepare/${TASK_PREFIX}_prepare`.
  - Stage manifests live under `saves/selective/stage/${TASK_PREFIX}_${INTRA_STAGE_ORDER}_stages`.
  - Stage checkpoints and final evals are written under `saves/unlearn/${task_name}`.
  - Final evals use the full TOFU metric suite, including exact memorization and the MIA metrics.
  - TensorBoard logs are written under each run's `${output_dir}/logs`.

- Reference split setup:
  - The script uses repeated random 50/50 partitions and trains one reference model on each half, so every repeat yields two cross-evaluating reference models.
  - `NUM_REFERENCE_REPEATS` controls how many random partitions are used. For example, `NUM_REFERENCE_REPEATS=3` yields 3 random partitions and 6 reference models.

- Resume behavior:
  - `RESUME=true` by default and skips completed reference split training, difficulty scoring, finished stage training, and final evals.
  - If a stage was interrupted after checkpoints were written, rerunning `run.py` resumes that stage from its latest checkpoint automatically.
  - Set `RESUME=false` to rebuild the workflow from scratch with the current script settings.

# Citation

Selective-DPO is a TOFU variant of the selective curriculum idea. If you use it, cite the OpenUnlearning technical report and the Selective DPO paper that motivates the curriculum idea.
