# MRD-NPO

- Repository-local implementation of the MRD-weighted NPO variant from *A Neuro-inspired Interpretation of Unlearning in Large Language Models through Sample-level Unlearning Difficulty*.
- Scope of this integration: TOFU only, using OpenUnlearning's existing NPO trainer, data preprocessing, and TOFU evaluation.
- The `mrd/` directory in this repository is treated as archival experiment code and is not used directly.

# Setup

- Method flow:
  - score current forget samples with MRD
  - convert MRD scores into weighted sampling probabilities
  - run one NPO round
  - refresh MRD from the latest checkpoint and continue
- Default runtime assumptions:
  - single process
  - single visible GPU
  - `trainer.train_sampler=weighted`
- Key environment variables:
  - `MODEL`, `FORGET_SPLIT`, `RETAIN_SPLIT`
  - `TOTAL_EPOCHS`, `REFRESH_EPOCHS`
  - `LEARNING_RATE`, `BETA`
  - `MRD_SIGMA`, `MRD_NUM_MC_SAMPLES`, `MRD_BATCH_SIZE`, `MRD_EPS`

# Results

Run [`run.py`](./run.py).

- MRD artifacts are written under `saves/mrd/${task_name}`.
- Round checkpoints are written under `saves/unlearn/${task_name}`.
- Final TOFU evaluation is written under the last round directory's `evals/`.

# Citation

```bibtex
@article{feng2025neuro,
  title={A Neuro-inspired Interpretation of Unlearning in Large Language Models through Sample-level Unlearning Difficulty},
  author={Feng, Xiaohua and Li, Yuyuan and Wang, Chengye and Liu, Junlin and Zhang, Li and Chen, Chaochao},
  journal={arXiv preprint arXiv:2504.06658},
  year={2025}
}
```
