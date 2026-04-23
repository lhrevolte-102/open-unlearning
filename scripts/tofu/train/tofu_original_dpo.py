#!/usr/bin/env python3

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "src"))

from selective.utils import save_json  # noqa: E402


VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
ORIGINAL_ROOT = ROOT_DIR / "saves" / "original"
UNLEARN_ROOT = ROOT_DIR / "saves" / "unlearn"

ENV_DEFAULTS = {
    "MODEL": "Llama-3.2-3B-Instruct",
    "FORGET_SPLIT": "forget10",
    "RETAIN_SPLIT": "retain90",
    "TOTAL_EPOCHS": "5",
    "STAGE_EPOCH_RATIOS": "[0.2,0.4,0.4]",
    "MAX_STAGE_ID": "0",
    "LEARNING_RATE": "1e-5",
    "BETA": "0.1",
    "ALPHA": "1",
    "PER_DEVICE_TRAIN_BATCH_SIZE": "16",
    "GRADIENT_ACCUMULATION_STEPS": "4",
    "TRAIN_LOGGING_STEPS": "1",
    "GPU_ID": "0",
    "RESUME": "true",
}

@dataclass(frozen=True)
class MethodSpec:
    label: str
    experiment: str
    trainer_name: str
    task_suffix: str
    forget_dataset_name: str


METHOD = MethodSpec(
    label="IdkDPO",
    experiment="unlearn/tofu/idk.yaml",
    trainer_name="DPO",
    task_suffix="DPO",
    forget_dataset_name="TOFU_QA_forget_idk",
)


@dataclass(frozen=True)
class RunConfig:
    model: str
    forget_split: str
    retain_split: str
    total_epochs_token: str
    total_epochs: float
    stage_epoch_ratios_raw: str
    max_stage_id: int
    learning_rate: str
    beta: str
    alpha: str
    per_device_train_batch_size: str
    gradient_accumulation_steps: str
    train_logging_steps: str
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

    @property
    def base_task_name(self) -> str:
        return f"tofu_{self.model}_{self.forget_split}_{METHOD.task_suffix}"


@dataclass(frozen=True)
class StagePlan:
    epoch_ratios: list[float]
    staged: bool


def env_value(name: str) -> str:
    return os.environ.get(name, ENV_DEFAULTS[name])


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def compact_number_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), "g").replace(".", "p")


def format_float_list_suffix(values: list[float], prefix: str) -> str:
    return prefix + "-".join(compact_number_token(value) for value in values)


def load_config() -> RunConfig:
    total_epochs_token = env_value("TOTAL_EPOCHS")
    return RunConfig(
        model=env_value("MODEL"),
        forget_split=env_value("FORGET_SPLIT"),
        retain_split=env_value("RETAIN_SPLIT"),
        total_epochs_token=total_epochs_token,
        total_epochs=float(total_epochs_token),
        stage_epoch_ratios_raw=env_value("STAGE_EPOCH_RATIOS"),
        max_stage_id=int(env_value("MAX_STAGE_ID")),
        learning_rate=env_value("LEARNING_RATE"),
        beta=env_value("BETA"),
        alpha=env_value("ALPHA"),
        per_device_train_batch_size=env_value("PER_DEVICE_TRAIN_BATCH_SIZE"),
        gradient_accumulation_steps=env_value("GRADIENT_ACCUMULATION_STEPS"),
        train_logging_steps=env_value("TRAIN_LOGGING_STEPS"),
        gpu_id=env_value("GPU_ID"),
        resume=is_truthy(env_value("RESUME")),
    )


def parse_stage_epoch_ratios(cfg: RunConfig) -> list[float]:
    if not cfg.stage_epoch_ratios_raw.strip():
        return [1.0]
    ratios = [float(value) for value in json.loads(cfg.stage_epoch_ratios_raw)]
    if not ratios:
        raise SystemExit("STAGE_EPOCH_RATIOS must not be empty.")
    if any(value <= 0 for value in ratios):
        raise SystemExit("STAGE_EPOCH_RATIOS must contain only positive values.")
    return ratios


def resolve_stage_plan(cfg: RunConfig) -> StagePlan:
    epoch_ratios = parse_stage_epoch_ratios(cfg)
    staged = cfg.stage_epoch_ratios_raw.strip() != "" and len(epoch_ratios) > 1
    return StagePlan(epoch_ratios=epoch_ratios, staged=staged)


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


def completed_epoch_count(output_dir: Path) -> int:
    if not output_dir.is_dir():
        return 0
    return len([path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")])


def training_output_complete(output_dir: Path, required_epoch_count: int | None = None) -> bool:
    if not (output_dir / "trainer_state.json").is_file() or not model_dir_complete(output_dir):
        return False
    if required_epoch_count is None:
        return True
    return completed_epoch_count(output_dir) >= required_epoch_count


def latest_checkpoint_in_dir(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    checkpoints = sorted(
        path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")
    )
    return checkpoints[-1] if checkpoints else None


def log(message: str) -> None:
    print(f"[Original-{METHOD.label}] {message}", flush=True)


def run_repo_python(args: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run([str(VENV_PYTHON), *args], cwd=ROOT_DIR, env=env, check=True)


def total_epoch_count(cfg: RunConfig) -> int:
    rounded = int(round(cfg.total_epochs))
    if not math.isclose(cfg.total_epochs, rounded):
        raise SystemExit("Staged original training currently requires integer TOTAL_EPOCHS.")
    return rounded


def stage_cumulative_epochs(cfg: RunConfig, plan: StagePlan) -> list[int]:
    total_epochs = total_epoch_count(cfg)
    cumulative_epochs: list[int] = []
    ratio_sum = 0.0
    previous_epoch = 0

    for stage_id, ratio in enumerate(plan.epoch_ratios, start=1):
        ratio_sum += ratio
        target_epoch = total_epochs if stage_id == len(plan.epoch_ratios) else int(round(ratio_sum * total_epochs))
        target_epoch = max(stage_id, target_epoch)
        if target_epoch <= previous_epoch:
            target_epoch = previous_epoch + 1
        cumulative_epochs.append(target_epoch)
        previous_epoch = target_epoch

    if cumulative_epochs[-1] != total_epochs:
        cumulative_epochs[-1] = total_epochs
    return cumulative_epochs


def common_train_args(cfg: RunConfig, pretrained_model_path: str) -> list[str]:
    return [
        f"model={cfg.model}",
        f"forget_split={cfg.forget_split}",
        f"retain_split={cfg.retain_split}",
        f"model.model_args.pretrained_model_name_or_path={pretrained_model_path}",
        f"model.tokenizer_args.pretrained_model_name_or_path={cfg.base_model_path}",
        f"retain_logs_path={cfg.retain_logs_path}",
        f"trainer.method_args.beta={cfg.beta}",
        f"trainer.method_args.alpha={cfg.alpha}",
        f"trainer.args.per_device_train_batch_size={cfg.per_device_train_batch_size}",
        f"trainer.args.gradient_accumulation_steps={cfg.gradient_accumulation_steps}",
        f"trainer.args.learning_rate={cfg.learning_rate}",
        f"trainer.args.logging_steps={cfg.train_logging_steps}",
        "+trainer.args.logging_first_step=true",
        "trainer.args.gradient_checkpointing=true",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        "trainer.args.save_strategy=epoch",
        f"+trainer.args.save_total_limit={total_epoch_count(cfg)}",
        "trainer.args.save_only_model=false",
        "+trainer.args.ignore_data_skip=true",
    ]


def run_train_command(
    cfg: RunConfig,
    task_name: str,
    output_dir: Path,
    num_train_epochs: str,
    pretrained_model_path: str,
    extra_args: list[str] | None = None,
) -> None:
    run_repo_python(
        [
            "src/train.py",
            "--config-name=unlearn.yaml",
            f"experiment={METHOD.experiment}",
            f"trainer={METHOD.trainer_name}",
            f"task_name={task_name}",
            *common_train_args(cfg, pretrained_model_path),
            f"trainer.args.num_train_epochs={num_train_epochs}",
            *(extra_args or []),
            f"paths.output_dir={output_dir}",
        ],
        extra_env={"CUDA_VISIBLE_DEVICES": cfg.gpu_id},
    )


def stage_manifest_suffix(cfg: RunConfig, plan: StagePlan) -> str:
    return format_float_list_suffix(plan.epoch_ratios, "ratio")


def staged_training_suffix(cfg: RunConfig, plan: StagePlan) -> str:
    return f"beta{cfg.beta}_epoch{cfg.total_epochs_token}_{format_float_list_suffix(plan.epoch_ratios, 'ratio')}"


def training_task_prefix(cfg: RunConfig, plan: StagePlan) -> str:
    if not plan.staged:
        return cfg.base_task_name
    return f"{cfg.base_task_name}_{staged_training_suffix(cfg, plan)}"


def stage_manifest_dir(cfg: RunConfig, plan: StagePlan) -> Path:
    return ORIGINAL_ROOT / "stage" / f"{cfg.base_task_name}_{stage_manifest_suffix(cfg, plan)}_stages" / "stages"


def build_original_stage_manifests(cfg: RunConfig, plan: StagePlan) -> list[Path]:
    from datasets import load_dataset

    stage_dir = stage_manifest_dir(cfg, plan)
    stage_dir.mkdir(parents=True, exist_ok=True)
    forget_dataset = load_dataset("locuslab/TOFU", name=cfg.forget_split, split="train")
    allowed_indices = list(range(len(forget_dataset)))
    cumulative_epochs = stage_cumulative_epochs(cfg, plan)
    manifests = []
    previous_cumulative_epoch = 0
    for stage_id, (epoch_ratio, cumulative_epoch) in enumerate(
        zip(plan.epoch_ratios, cumulative_epochs, strict=True),
        start=1,
    ):
        manifests.append(
            {
                "stage_id": stage_id,
                "stage_name": f"stage{stage_id}",
                "epoch_ratio": float(epoch_ratio),
                "stage_epochs": cumulative_epoch - previous_cumulative_epoch,
                "cumulative_epoch": cumulative_epoch,
                "allowed_indices": allowed_indices,
                "num_examples": len(allowed_indices),
                "total_examples": len(allowed_indices),
                "method": METHOD.label,
                "ordering": "full_dataset",
            }
        )
        previous_cumulative_epoch = cumulative_epoch

    summary = {
        "method": METHOD.label,
        "model": cfg.model,
        "forget_split": cfg.forget_split,
        "retain_split": cfg.retain_split,
        "stage_epoch_ratios": plan.epoch_ratios,
        "stages": [],
    }
    for manifest in manifests:
        output_path = stage_dir / f"{manifest['stage_name']}.json"
        save_json(output_path, manifest)
        summary["stages"].append(
            {
                "stage_name": manifest["stage_name"],
                "output_path": str(output_path.resolve()),
                "num_examples": manifest["num_examples"],
                "epoch_ratio": manifest["epoch_ratio"],
                "stage_epochs": manifest["stage_epochs"],
                "cumulative_epoch": manifest["cumulative_epoch"],
            }
        )
    save_json(stage_dir / "stages.json", summary)
    return sorted(stage_dir.glob("stage[0-9]*.json"))


def run_single_stage(cfg: RunConfig, task_name: str) -> str:
    output_dir = UNLEARN_ROOT / task_name
    required_epoch_count = total_epoch_count(cfg)
    if cfg.resume and training_output_complete(output_dir, required_epoch_count):
        log(f"Skipping {METHOD.label}; found completed model output at {output_dir}.")
        return task_name
    extra_args: list[str] = []
    if cfg.resume:
        latest_checkpoint = latest_checkpoint_in_dir(output_dir)
        if latest_checkpoint is not None:
            log(f"Resuming {METHOD.label} from in-progress checkpoint {latest_checkpoint}.")
            extra_args.append(f"resume_from_checkpoint={latest_checkpoint}")
    run_train_command(
        cfg=cfg,
        task_name=task_name,
        output_dir=output_dir,
        num_train_epochs=cfg.total_epochs_token,
        pretrained_model_path=cfg.base_model_path,
        extra_args=extra_args,
    )
    return task_name


def run_staged_training(cfg: RunConfig, task_name: str, plan: StagePlan) -> str:
    output_dir = UNLEARN_ROOT / task_name
    cumulative_epochs = stage_cumulative_epochs(cfg, plan)
    stage_limit = len(cumulative_epochs) if cfg.max_stage_id <= 0 else min(cfg.max_stage_id, len(cumulative_epochs))
    target_epoch_count = cumulative_epochs[stage_limit - 1]
    if cfg.resume and training_output_complete(output_dir, target_epoch_count):
        log(f"Skipping {METHOD.label}; found completed model output at {output_dir}.")
        return task_name

    extra_args: list[str] = []
    if cfg.resume:
        latest_checkpoint = latest_checkpoint_in_dir(output_dir)
        if latest_checkpoint is not None:
            log(f"Resuming {METHOD.label} from in-progress checkpoint {latest_checkpoint}.")
            extra_args.append(f"resume_from_checkpoint={latest_checkpoint}")

    build_original_stage_manifests(cfg, plan)
    run_train_command(
        cfg=cfg,
        task_name=task_name,
        output_dir=output_dir,
        num_train_epochs=str(target_epoch_count),
        pretrained_model_path=cfg.base_model_path,
        extra_args=extra_args,
    )
    return task_name


def run_unlearn(cfg: RunConfig, plan: StagePlan) -> str:
    task_prefix = training_task_prefix(cfg, plan)
    if not plan.staged:
        return run_single_stage(cfg, task_prefix)
    return run_staged_training(cfg, task_prefix, plan)


def main() -> None:
    cfg = load_config()
    plan = resolve_stage_plan(cfg)
    if cfg.resume:
        log("Resume mode enabled. Completed unlearning runs will be skipped.")
    else:
        log("Resume mode disabled. Existing outputs will be reused only when the underlying command does so implicitly.")
    if plan.staged:
        suffix = f" and MAX_STAGE_ID={cfg.max_stage_id}" if cfg.max_stage_id > 0 else ""
        log(f"Staged original training enabled with STAGE_EPOCH_RATIOS={plan.epoch_ratios}{suffix}.")
    final_task_name = run_unlearn(cfg, plan)
    log(f"{METHOD.label} training output: saves/unlearn/{final_task_name}")


if __name__ == "__main__":
    main()
