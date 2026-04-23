#!/usr/bin/env python3

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
UNLEARN_ROOT = ROOT_DIR / "saves" / "unlearn"
EVAL_ROOT = ROOT_DIR / "saves" / "eval"

REQUIRED_EVAL_METRICS = {
    "exact_memorization",
    "mia_gradnorm",
    "mia_loss",
    "mia_min_k",
    "mia_min_k_plus_plus",
    "mia_reference",
    "mia_zlib",
}


@dataclass(frozen=True)
class TofuEvalConfig:
    model: str
    forget_split: str
    retain_split: str
    gpu_id: str
    resume: bool

    @property
    def base_model_path(self) -> str:
        return f"open-unlearning/tofu_{self.model}_full"

    @property
    def retain_model_path(self) -> str:
        return f"open-unlearning/tofu_{self.model}_{self.retain_split}"

    @property
    def retain_logs_path(self) -> str:
        return f"saves/eval/tofu_{self.model}_{self.retain_split}/TOFU_EVAL.json"


def env_value(name: str, default: str) -> str:
    return os.environ.get(name, default)


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_number_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), "g").replace(".", "p")


def format_float_list_suffix(values: list[float], prefix: str) -> str:
    return prefix + "-".join(compact_number_token(value) for value in values)


def parse_float_list(raw: str, name: str) -> list[float]:
    values = [float(value) for value in json.loads(raw)]
    if not values:
        raise SystemExit(f"{name} must not be empty.")
    return values


def model_dir_complete(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").is_file():
        return False
    if any(model_dir.glob("model-*.safetensors")):
        return True
    return any(
        (model_dir / name).is_file()
        for name in ("model.safetensors", "model.safetensors.index.json")
    )


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    if match is None:
        raise ValueError(f"Invalid checkpoint directory name: {path}")
    return int(match.group(1))


def sorted_checkpoint_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    return sorted(
        (path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")),
        key=checkpoint_step,
    )


def checkpoint_dir_for_epoch(output_dir: Path, epoch_index: int) -> Path:
    checkpoints = sorted_checkpoint_dirs(output_dir)
    if epoch_index < 1:
        raise SystemExit(f"Epoch index must be positive, got {epoch_index}.")
    if len(checkpoints) < epoch_index:
        raise SystemExit(
            f"Expected at least {epoch_index} epoch checkpoints under {output_dir}, found {len(checkpoints)}."
        )
    return checkpoints[epoch_index - 1]


def tofu_eval_has_full_metrics(eval_file: Path) -> bool:
    if not eval_file.is_file():
        return False
    return REQUIRED_EVAL_METRICS.issubset(load_json(eval_file))


def run_repo_python(args: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run([str(VENV_PYTHON), *args], cwd=ROOT_DIR, env=env, check=True)


def run_tofu_eval(
    cfg: TofuEvalConfig,
    task_name: str,
    prefix: str,
    model_path: Path | None = None,
) -> None:
    output_dir = model_path if model_path is not None else UNLEARN_ROOT / task_name
    if not model_dir_complete(output_dir):
        raise SystemExit(f"Model output is missing or incomplete: {output_dir}")

    eval_dir = EVAL_ROOT / task_name
    eval_file = eval_dir / "TOFU_EVAL.json"
    if cfg.resume and tofu_eval_has_full_metrics(eval_file):
        print(f"[{prefix}] Skipping eval for {task_name}; found {eval_file}.", flush=True)
        return

    print(f"[{prefix}] Running eval for {task_name}.", flush=True)
    run_repo_python(
        [
            "src/eval.py",
            "experiment=eval/tofu/full",
            f"forget_split={cfg.forget_split}",
            f"model={cfg.model}",
            f"task_name={task_name}",
            f"model.model_args.pretrained_model_name_or_path={output_dir}",
            f"model.tokenizer_args.pretrained_model_name_or_path={cfg.base_model_path}",
            f"paths.output_dir={eval_dir}",
            f"retain_logs_path={cfg.retain_logs_path}",
            f"reference_model_path={cfg.retain_model_path}",
        ],
        extra_env={"CUDA_VISIBLE_DEVICES": cfg.gpu_id},
    )
