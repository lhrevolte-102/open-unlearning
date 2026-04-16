# Selective-DPO

- Repository-local adaptation of the selective curriculum idea applied to IdkDPO.
- Method base: OpenUnlearning TOFU unlearning with a difficulty-aware, staged forget schedule.
- Selective data ordering is built from held-out reference folds, then used to train staged unlearning runs from easy to hard.

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

# Results

Run [`run.sh`](./run.sh).

# Citation

This is a repository-local adaptation rather than a separate paper artifact. If you use it, please cite the OpenUnlearning technical report and the Selective DPO paper that motivates the curriculum idea.
