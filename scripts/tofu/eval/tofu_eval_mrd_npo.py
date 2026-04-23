#!/usr/bin/env python3

import math
from dataclasses import dataclass

from tofu_eval_common import TofuEvalConfig, env_value, is_truthy, run_tofu_eval


ENV_DEFAULTS = {
    "MODEL": "Llama-3.2-3B-Instruct",
    "FORGET_SPLIT": "forget10",
    "RETAIN_SPLIT": "retain90",
    "TOTAL_EPOCHS": "5",
    "REFRESH_EPOCHS": "1",
    "LEARNING_RATE": "1e-5",
    "ALPHA": "1",
    "BETA": "0.1",
    "GPU_ID": "0",
    "RESUME": "true",
}


@dataclass(frozen=True)
class RunConfig(TofuEvalConfig):
    total_epochs_token: str
    total_epochs: float
    refresh_epochs_token: str
    refresh_epochs: float
    learning_rate: str
    alpha: str
    beta: str

    @property
    def num_rounds(self) -> int:
        return int(math.ceil(self.total_epochs / self.refresh_epochs))

    @property
    def task_prefix(self) -> str:
        return (
            f"tofu_{self.model}_{self.forget_split}_MRD-NPO_"
            f"lr{self.learning_rate}_alpha{self.alpha}_beta{self.beta}_"
            f"epoch{self.total_epochs_token}_"
            f"refresh{self.refresh_epochs_token}"
        )

    def round_task_name(self, round_id: int) -> str:
        return f"{self.task_prefix}_round{round_id:02d}"


def load_config() -> RunConfig:
    total_epochs_token = env_value("TOTAL_EPOCHS", ENV_DEFAULTS["TOTAL_EPOCHS"])
    refresh_epochs_token = env_value("REFRESH_EPOCHS", ENV_DEFAULTS["REFRESH_EPOCHS"])
    return RunConfig(
        model=env_value("MODEL", ENV_DEFAULTS["MODEL"]),
        forget_split=env_value("FORGET_SPLIT", ENV_DEFAULTS["FORGET_SPLIT"]),
        retain_split=env_value("RETAIN_SPLIT", ENV_DEFAULTS["RETAIN_SPLIT"]),
        gpu_id=env_value("GPU_ID", ENV_DEFAULTS["GPU_ID"]),
        resume=is_truthy(env_value("RESUME", ENV_DEFAULTS["RESUME"])),
        total_epochs_token=total_epochs_token,
        total_epochs=float(total_epochs_token),
        refresh_epochs_token=refresh_epochs_token,
        refresh_epochs=float(refresh_epochs_token),
        learning_rate=env_value("LEARNING_RATE", ENV_DEFAULTS["LEARNING_RATE"]),
        alpha=env_value("ALPHA", ENV_DEFAULTS["ALPHA"]),
        beta=env_value("BETA", ENV_DEFAULTS["BETA"]),
    )


def main() -> None:
    cfg = load_config()
    if cfg.refresh_epochs <= 0:
        raise SystemExit("REFRESH_EPOCHS must be positive.")

    target_rounds = [round_id for round_id in (1, 3, 5) if round_id <= cfg.num_rounds]
    if not target_rounds:
        raise SystemExit("No MRD rounds are available to evaluate.")

    for round_id in target_rounds:
        run_tofu_eval(cfg, cfg.round_task_name(round_id), "Eval-MRD-NPO")


if __name__ == "__main__":
    main()
