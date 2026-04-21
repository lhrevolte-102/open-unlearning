#!/usr/bin/env python3

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
UNLEARN_ROOT = ROOT_DIR / "saves" / "unlearn"

ENV_DEFAULTS = {
    "MODEL": "Llama-3.2-3B-Instruct",
    "FORGET_SPLIT": "forget10",
    "RETAIN_SPLIT": "retain90",
    "TOTAL_EPOCHS": "5",
    "STAGE_EPOCH_RATIOS": "",
    "MAX_STAGE_ID": "0",
    "BETA": "0.1",
    "PER_DEVICE_TRAIN_BATCH_SIZE": "16",
    "GRADIENT_ACCUMULATION_STEPS": "4",
    "TRAIN_LOGGING_STEPS": "1",
    "GPU_ID": "0",
    "RESUME": "true",
}

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
class MethodSpec:
    label: str
    experiment: str
    trainer_name: str
    task_suffix: str


METHODS = (
    MethodSpec(
        label="IdkDPO",
        experiment="unlearn/tofu/idk.yaml",
        trainer_name="DPO",
        task_suffix="idkdpo_original",
    ),
    MethodSpec(
        label="NPO",
        experiment="unlearn/tofu/default.yaml",
        trainer_name="NPO",
        task_suffix="npo_original",
    ),
)


def env_value(name: str) -> str:
    return os.environ.get(name, ENV_DEFAULTS[name])


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def training_output_complete(output_dir: Path) -> bool:
    return (output_dir / "trainer_state.json").is_file() and model_dir_complete(output_dir)


def latest_checkpoint_in_dir(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    checkpoints = sorted(
        path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")
    )
    return checkpoints[-1] if checkpoints else None


def tofu_eval_has_full_metrics(eval_file: Path) -> bool:
    if not eval_file.is_file():
        return False
    return REQUIRED_EVAL_METRICS.issubset(load_json(eval_file))


def log(message: str) -> None:
    print(f"[Original-TOFU] {message}", flush=True)


def run_repo_python(args: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run([str(VENV_PYTHON), *args], cwd=ROOT_DIR, env=env, check=True)


@dataclass(frozen=True)
class RunConfig:
    model: str
    forget_split: str
    retain_split: str
    total_epochs_token: str
    total_epochs: float
    stage_epoch_ratios_raw: str
    max_stage_id: int
    beta: str
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

    def task_name_for(self, method: MethodSpec) -> str:
        return f"tofu_{self.model}_{self.forget_split}_{method.task_suffix}"


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
        beta=env_value("BETA"),
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


def resolve_stage_epoch_plan(cfg: RunConfig) -> tuple[list[float], bool]:
    ratios = parse_stage_epoch_ratios(cfg)
    return ratios, cfg.stage_epoch_ratios_raw.strip() != "" and len(ratios) > 1


def common_train_args(cfg: RunConfig, pretrained_model_path: str) -> list[str]:
    return [
        f"model={cfg.model}",
        f"forget_split={cfg.forget_split}",
        f"retain_split={cfg.retain_split}",
        f"model.model_args.pretrained_model_name_or_path={pretrained_model_path}",
        f"model.tokenizer_args.pretrained_model_name_or_path={cfg.base_model_path}",
        f"retain_logs_path={cfg.retain_logs_path}",
        f"trainer.method_args.beta={cfg.beta}",
        f"trainer.args.per_device_train_batch_size={cfg.per_device_train_batch_size}",
        f"trainer.args.gradient_accumulation_steps={cfg.gradient_accumulation_steps}",
        f"trainer.args.logging_steps={cfg.train_logging_steps}",
        "+trainer.args.logging_first_step=true",
        "trainer.args.gradient_checkpointing=true",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        "trainer.args.save_strategy=epoch",
        "+trainer.args.save_total_limit=1",
        "trainer.args.save_only_model=false",
        "+trainer.args.ignore_data_skip=true",
    ]


def run_train_command(
    cfg: RunConfig,
    method: MethodSpec,
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
            f"experiment={method.experiment}",
            f"trainer={method.trainer_name}",
            f"task_name={task_name}",
            *common_train_args(cfg, pretrained_model_path),
            f"trainer.args.num_train_epochs={num_train_epochs}",
            *(extra_args or []),
            f"paths.output_dir={output_dir}",
        ],
        extra_env={"CUDA_VISIBLE_DEVICES": cfg.gpu_id},
    )


def run_single_stage(cfg: RunConfig, method: MethodSpec, task_name: str) -> str:
    output_dir = UNLEARN_ROOT / task_name
    if cfg.resume and training_output_complete(output_dir):
        log(f"Skipping {method.label}; found completed model output at {output_dir}.")
        return task_name

    extra_args: list[str] = []
    if cfg.resume:
        latest_checkpoint = latest_checkpoint_in_dir(output_dir)
        if latest_checkpoint is not None:
            log(f"Resuming {method.label} from in-progress checkpoint {latest_checkpoint}.")
            extra_args.append(f"resume_from_checkpoint={latest_checkpoint}")

    run_train_command(
        cfg=cfg,
        method=method,
        task_name=task_name,
        output_dir=output_dir,
        num_train_epochs=cfg.total_epochs_token,
        pretrained_model_path=cfg.base_model_path,
        extra_args=extra_args,
    )
    return task_name


def run_multi_stage(cfg: RunConfig, method: MethodSpec, task_prefix: str, stage_epoch_ratios: list[float]) -> str:
    stage_limit = len(stage_epoch_ratios) if cfg.max_stage_id <= 0 else min(cfg.max_stage_id, len(stage_epoch_ratios))
    prev_output_dir: Path | None = None
    final_task_name = ""

    for stage_id, epoch_ratio in enumerate(stage_epoch_ratios[:stage_limit], start=1):
        stage_name = f"stage{stage_id}"
        task_name = f"{task_prefix}_{stage_name}"
        output_dir = UNLEARN_ROOT / task_name
        final_task_name = task_name

        if cfg.resume and training_output_complete(output_dir):
            log(f"Skipping {method.label} {stage_name}; found completed model output at {output_dir}.")
            prev_output_dir = output_dir
            continue

        extra_args: list[str] = []
        pretrained_model_path = cfg.base_model_path

        if cfg.resume:
            latest_checkpoint = latest_checkpoint_in_dir(output_dir)
            if latest_checkpoint is not None:
                log(f"Resuming {method.label} {stage_name} from in-progress checkpoint {latest_checkpoint}.")
                extra_args.append(f"resume_from_checkpoint={latest_checkpoint}")

        if not extra_args and prev_output_dir is not None:
            latest_checkpoint = latest_checkpoint_in_dir(prev_output_dir)
            if latest_checkpoint is None:
                raise SystemExit(f"No checkpoint found under {prev_output_dir} for {method.label} {stage_name} resume.")
            log(f"Initializing {method.label} {stage_name} from previous stage model {latest_checkpoint}.")
            pretrained_model_path = str(latest_checkpoint)

        run_train_command(
            cfg=cfg,
            method=method,
            task_name=task_name,
            output_dir=output_dir,
            num_train_epochs=str(max(epoch_ratio * cfg.total_epochs, 1.0)),
            pretrained_model_path=pretrained_model_path,
            extra_args=extra_args,
        )
        prev_output_dir = output_dir

    if not final_task_name:
        raise SystemExit(f"No stages were executed for {method.label}.")
    return final_task_name


def run_unlearn(cfg: RunConfig, method: MethodSpec) -> str:
    task_name = cfg.task_name_for(method)
    stage_epoch_ratios, staged = resolve_stage_epoch_plan(cfg)
    if not staged:
        return run_single_stage(cfg, method, task_name)
    return run_multi_stage(cfg, method, task_name, stage_epoch_ratios)


def run_eval(cfg: RunConfig, method: MethodSpec, task_name: str) -> None:
    output_dir = UNLEARN_ROOT / task_name
    eval_dir = output_dir / "evals"
    eval_file = eval_dir / "TOFU_EVAL.json"

    if cfg.resume and tofu_eval_has_full_metrics(eval_file):
        log(f"Skipping {method.label} eval; found existing full-metric eval logs at {eval_file}.")
        return

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


def main() -> None:
    cfg = load_config()
    stage_epoch_ratios, staged = resolve_stage_epoch_plan(cfg)

    if cfg.resume:
        log("Resume mode enabled. Completed unlearning runs and full-metric evals will be skipped.")
    else:
        log("Resume mode disabled. Existing outputs will be reused only when the underlying command does so implicitly.")

    if staged:
        suffix = f" and MAX_STAGE_ID={cfg.max_stage_id}" if cfg.max_stage_id > 0 else ""
        log(f"Staged original training enabled with STAGE_EPOCH_RATIOS={stage_epoch_ratios}{suffix}.")

    for method in METHODS:
        final_task_name = run_unlearn(cfg, method)
        run_eval(cfg, method, final_task_name)
        log(f"{method.label} output: saves/unlearn/{final_task_name}")

    log("Finished original TOFU runs.")


if __name__ == "__main__":
    main()
