#!/usr/bin/env python3

from dataclasses import dataclass

from tofu_eval_common import (
    TofuEvalConfig,
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
    "STAGE_PERCENTILES": "[0.3,0.6,1.0]",
    "STAGE_EPOCH_RATIOS": "[0.2,0.4,0.4]",
    "INTRA_STAGE_ORDER": "random",
    "LEARNING_RATE": "1e-5",
    "ALPHA": "1",
    "BETA": "0.1",
    "NUM_REFERENCE_REPEATS": "3",
    "MAX_STAGE_ID": "0",
    "GPU_ID": "0",
    "RESUME": "true",
}


@dataclass(frozen=True)
class RunConfig(TofuEvalConfig):
    total_epochs_token: str
    stage_percentiles_raw: str
    stage_epoch_ratios_raw: str
    intra_stage_order: str
    learning_rate: str
    alpha: str
    beta: str
    num_reference_repeats: str
    max_stage_id: int

    @property
    def stage_run_prefix(self) -> str:
        stage_percentiles = parse_float_list(self.stage_percentiles_raw, "STAGE_PERCENTILES")
        stage_epoch_ratios = parse_float_list(self.stage_epoch_ratios_raw, "STAGE_EPOCH_RATIOS")
        return (
            f"tofu_{self.model}_{self.forget_split}_Selective-NPO_"
            f"lr{self.learning_rate}_alpha{self.alpha}_beta{self.beta}_"
            f"epoch{self.total_epochs_token}_refs{self.num_reference_repeats}_"
            f"{format_float_list_suffix(stage_percentiles, 'pct')}_"
            f"{format_float_list_suffix(stage_epoch_ratios, 'ratio')}_"
            f"{self.intra_stage_order}"
        )


def load_config() -> RunConfig:
    return RunConfig(
        model=env_value("MODEL", ENV_DEFAULTS["MODEL"]),
        forget_split=env_value("FORGET_SPLIT", ENV_DEFAULTS["FORGET_SPLIT"]),
        retain_split=env_value("RETAIN_SPLIT", ENV_DEFAULTS["RETAIN_SPLIT"]),
        gpu_id=env_value("GPU_ID", ENV_DEFAULTS["GPU_ID"]),
        resume=is_truthy(env_value("RESUME", ENV_DEFAULTS["RESUME"])),
        total_epochs_token=env_value("TOTAL_EPOCHS", ENV_DEFAULTS["TOTAL_EPOCHS"]),
        stage_percentiles_raw=env_value("STAGE_PERCENTILES", ENV_DEFAULTS["STAGE_PERCENTILES"]),
        stage_epoch_ratios_raw=env_value("STAGE_EPOCH_RATIOS", ENV_DEFAULTS["STAGE_EPOCH_RATIOS"]),
        intra_stage_order=env_value("INTRA_STAGE_ORDER", ENV_DEFAULTS["INTRA_STAGE_ORDER"]),
        learning_rate=env_value("LEARNING_RATE", ENV_DEFAULTS["LEARNING_RATE"]),
        alpha=env_value("ALPHA", ENV_DEFAULTS["ALPHA"]),
        beta=env_value("BETA", ENV_DEFAULTS["BETA"]),
        num_reference_repeats=env_value("NUM_REFERENCE_REPEATS", ENV_DEFAULTS["NUM_REFERENCE_REPEATS"]),
        max_stage_id=int(env_value("MAX_STAGE_ID", ENV_DEFAULTS["MAX_STAGE_ID"])),
    )


def resolve_task_names(cfg: RunConfig) -> list[str]:
    stage_count = len(parse_float_list(cfg.stage_epoch_ratios_raw, "STAGE_EPOCH_RATIOS"))
    if cfg.max_stage_id > 0:
        stage_count = min(stage_count, cfg.max_stage_id)
    return [f"{cfg.stage_run_prefix}_stage{stage_id}" for stage_id in range(1, stage_count + 1)]


def main() -> None:
    cfg = load_config()
    for task_name in resolve_task_names(cfg):
        run_tofu_eval(cfg, task_name, "Eval-Selective-NPO")


if __name__ == "__main__":
    main()
