# MRD-NPO

- MRD-weighted NPO for TOFU, based on *A Neuro-inspired Interpretation of Unlearning in Large Language Models through Sample-level Unlearning Difficulty*.
- Uses OpenUnlearning's existing NPO trainer, data preprocessing, and TOFU evaluation.
- The archived `mrd/` directory is not used directly by this method.

# Setup

- Workflow:
  - score current forget samples with MRD
  - convert MRD scores into weighted sampling probabilities
  - run one NPO round
  - refresh MRD from the latest checkpoint and continue
- Runtime assumptions:
  - single process
  - single visible GPU
  - `trainer.train_sampler=weighted`
- Key environment variables:
  - `MODEL`, `FORGET_SPLIT`, `RETAIN_SPLIT`
  - `TOTAL_EPOCHS`, `REFRESH_EPOCHS`
  - `LEARNING_RATE`, `BETA`, `ALPHA`
  - `MRD_SIGMA`, `MRD_NUM_MC_SAMPLES`, `MRD_BATCH_SIZE`, `MRD_EPS`
- Default optimization settings:
  - `LEARNING_RATE=1e-5`
  - `BETA=0.1`
  - `ALPHA=1`

# Results

Run [`run.py`](./run.py).

- MRD artifacts are written under `saves/mrd/${task_name}`.
- Round checkpoints are written under `saves/unlearn/${task_name}`.
- Final TOFU evaluation is written under the last round directory's `evals/`.
- Naming follows the same pattern as `Selective-DPO` / `Selective-NPO`: a stable base `task_prefix` plus a role suffix. The `task_prefix` encodes the main run-affecting hyperparameters: `lr`, `beta`, `alpha`, `epoch`, and `refresh`.
- Example training task name: `tofu_${MODEL}_${FORGET_SPLIT}_MRD-NPO_lr1e-5_alpha1_beta0.1_epoch5_refresh1_round01`.
- Example MRD task name: `tofu_${MODEL}_${FORGET_SPLIT}_MRD-NPO_lr1e-5_alpha1_beta0.1_epoch5_refresh1_mrd_round01`.

# Citation

```bibtex
@article{feng2025neuro,
  title={A Neuro-inspired Interpretation of Unlearning in Large Language Models through Sample-level Unlearning Difficulty},
  author={Feng, Xiaohua and Li, Yuyuan and Wang, Chengye and Liu, Junlin and Zhang, Li and Chen, Chaochao},
  journal={arXiv preprint arXiv:2504.06658},
  year={2025}
}
```
