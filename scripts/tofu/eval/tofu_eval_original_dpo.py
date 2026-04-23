#!/usr/bin/env python3

import math
from dataclasses import dataclass
from pathlib import Path

from tofu_eval_common import (
    TofuEvalConfig,
    UNLEARN_ROOT,
    checkpoint_dir_for_epoch,
    env_value,
    format_float_list_suffix,
    is_truthy,
    parse_float_list,
    run_tofu_eval,
)


ENV_DEFAULTS = {
    "MODEL": "Llama-3.2-3B-Instruct",
    "FORGET_SPLIT": "forget10",
    "RETAIN_SPLIT": "retain90",
    "TOTAL_EPOCHS": "5",
    "STAGE_EPOCH_RATIOS": "[0.2,0.4,0.4]",
    "MAX_STAGE_ID": "0",
    "GPU_ID": "0",
    "RESUME": "true",
    "BETA": "0.1",
}


@dataclass(frozen=True)
class RunConfig(TofuEvalConfig):
    total_epochs_token: str
    total_epochs: float
    stage_epoch_ratios_raw: str
    max_stage_id: int
    beta: str

    @property
    def base_task_name(self) -> str:
        return f"tofu_{self.model}_{self.forget_split}_DPO"


def load_config() -> RunConfig:
    return RunConfig(
        model=env_value("MODEL", ENV_DEFAULTS["MODEL"]),
        forget_split=env_value("FORGET_SPLIT", ENV_DEFAULTS["FORGET_SPLIT"]),
        retain_split=env_value("RETAIN_SPLIT", ENV_DEFAULTS["RETAIN_SPLIT"]),
        gpu_id=env_value("GPU_ID", ENV_DEFAULTS["GPU_ID"]),
        resume=is_truthy(env_value("RESUME", ENV_DEFAULTS["RESUME"])),
        total_epochs_token=env_value("TOTAL_EPOCHS", ENV_DEFAULTS["TOTAL_EPOCHS"]),
        total_epochs=float(env_value("TOTAL_EPOCHS", ENV_DEFAULTS["TOTAL_EPOCHS"])),
        stage_epoch_ratios_raw=env_value("STAGE_EPOCH_RATIOS", ENV_DEFAULTS["STAGE_EPOCH_RATIOS"]),
        max_stage_id=int(env_value("MAX_STAGE_ID", ENV_DEFAULTS["MAX_STAGE_ID"])),
        beta=env_value("BETA", ENV_DEFAULTS["BETA"]),
    )


def total_epoch_count(cfg: RunConfig) -> int:
    rounded = int(round(cfg.total_epochs))
    if not math.isclose(cfg.total_epochs, rounded):
        raise SystemExit("Staged original eval currently requires integer TOTAL_EPOCHS.")
    return rounded


def stage_cumulative_epochs(cfg: RunConfig, epoch_ratios: list[float]) -> list[int]:
    total_epochs = total_epoch_count(cfg)
    cumulative_epochs: list[int] = []
    ratio_sum = 0.0
    previous_epoch = 0

    for stage_id, ratio in enumerate(epoch_ratios, start=1):
        ratio_sum += ratio
        target_epoch = total_epochs if stage_id == len(epoch_ratios) else int(round(ratio_sum * total_epochs))
        target_epoch = max(stage_id, target_epoch)
        if target_epoch <= previous_epoch:
            target_epoch = previous_epoch + 1
        cumulative_epochs.append(target_epoch)
        previous_epoch = target_epoch

    if cumulative_epochs[-1] != total_epochs:
        cumulative_epochs[-1] = total_epochs
    return cumulative_epochs


def resolve_eval_targets(cfg: RunConfig) -> list[tuple[str, Path | None]]:
    if not cfg.stage_epoch_ratios_raw.strip():
        return [(cfg.base_task_name, None)]

    epoch_ratios = parse_float_list(cfg.stage_epoch_ratios_raw, "STAGE_EPOCH_RATIOS")
    run_task_name = (
        f"{cfg.base_task_name}_beta{cfg.beta}_epoch{cfg.total_epochs_token}_"
        f"{format_float_list_suffix(epoch_ratios, 'ratio')}"
    )
    run_output_dir = UNLEARN_ROOT / run_task_name
    cumulative_epochs = stage_cumulative_epochs(cfg, epoch_ratios)
    stage_count = len(cumulative_epochs) if cfg.max_stage_id <= 0 else min(cfg.max_stage_id, len(cumulative_epochs))
    return [
        (f"{run_task_name}_stage{stage_id}", checkpoint_dir_for_epoch(run_output_dir, cumulative_epochs[stage_id - 1]))
        for stage_id in range(1, stage_count + 1)
    ]


def main() -> None:
    cfg = load_config()
    for task_name, model_path in resolve_eval_targets(cfg):
        run_tofu_eval(cfg, task_name, "Eval-Original-DPO", model_path=model_path)


if __name__ == "__main__":
    main()
