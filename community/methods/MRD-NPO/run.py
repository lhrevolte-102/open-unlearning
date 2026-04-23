#!/usr/bin/env python3

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
MRD_ROOT = ROOT_DIR / "saves" / "mrd"
UNLEARN_ROOT = ROOT_DIR / "saves" / "unlearn"

ENV_DEFAULTS = {
    "MODEL": "Llama-3.2-3B-Instruct",
    "FORGET_SPLIT": "forget10",
    "RETAIN_SPLIT": "retain90",
    "TOTAL_EPOCHS": "5",
    "REFRESH_EPOCHS": "1",
    "LEARNING_RATE": "1e-5",
    "BETA": "0.1",
    "ALPHA": "1",
    "PER_DEVICE_TRAIN_BATCH_SIZE": "16",
    "GRADIENT_ACCUMULATION_STEPS": "4",
    "TRAIN_LOGGING_STEPS": "1",
    "MRD_SIGMA": "1e-5",
    "MRD_NUM_MC_SAMPLES": "8",
    "MRD_BATCH_SIZE": "4",
    "MRD_EPS": "1e-6",
    "GPU_ID": "0",
    "RESUME": "true",
}


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
    return (output_dir / "trainer_state.json").is_file() and model_dir_complete(
        output_dir
    )


def latest_checkpoint_in_dir(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    checkpoints = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    )
    return checkpoints[-1] if checkpoints else None


def log(message: str) -> None:
    print(f"[MRD-NPO] {message}", flush=True)


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
    refresh_epochs_token: str
    refresh_epochs: float
    learning_rate: str
    beta: str
    alpha: str
    per_device_train_batch_size: str
    gradient_accumulation_steps: str
    train_logging_steps: str
    mrd_sigma: str
    mrd_num_mc_samples: str
    mrd_batch_size: str
    mrd_eps: str
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
    def run_config_suffix(self) -> str:
        return (
            f"MRD-NPO_lr{self.learning_rate}_alpha{self.alpha}_beta{self.beta}_"
            f"epoch{self.total_epochs_token}_refresh{self.refresh_epochs_token}"
        )

    @property
    def task_prefix(self) -> str:
        return f"tofu_{self.model}_{self.forget_split}_{self.run_config_suffix}"

    @property
    def mrd_task_prefix(self) -> str:
        return f"{self.task_prefix}_mrd"

    @property
    def num_rounds(self) -> int:
        return int(math.ceil(self.total_epochs / self.refresh_epochs))

    def round_epochs(self, round_id: int) -> str:
        consumed = round_id * self.refresh_epochs
        remaining = max(self.total_epochs - consumed, 0.0)
        return str(min(self.refresh_epochs, remaining))

    def round_task_name(self, round_id: int) -> str:
        return f"{self.task_prefix}_round{round_id + 1:02d}"

    def round_mrd_task_name(self, round_id: int) -> str:
        return f"{self.mrd_task_prefix}_round{round_id + 1:02d}"

    def round_output_dir(self, round_id: int) -> Path:
        return UNLEARN_ROOT / self.round_task_name(round_id)

    def round_mrd_dir(self, round_id: int) -> Path:
        return MRD_ROOT / self.round_mrd_task_name(round_id)

    def round_weights_path(self, round_id: int) -> Path:
        return self.round_mrd_dir(round_id) / "difficulty.json"


def load_config() -> RunConfig:
    total_epochs_token = env_value("TOTAL_EPOCHS")
    refresh_epochs_token = env_value("REFRESH_EPOCHS")
    return RunConfig(
        model=env_value("MODEL"),
        forget_split=env_value("FORGET_SPLIT"),
        retain_split=env_value("RETAIN_SPLIT"),
        total_epochs_token=total_epochs_token,
        total_epochs=float(total_epochs_token),
        refresh_epochs_token=refresh_epochs_token,
        refresh_epochs=float(refresh_epochs_token),
        learning_rate=env_value("LEARNING_RATE"),
        beta=env_value("BETA"),
        alpha=env_value("ALPHA"),
        per_device_train_batch_size=env_value("PER_DEVICE_TRAIN_BATCH_SIZE"),
        gradient_accumulation_steps=env_value("GRADIENT_ACCUMULATION_STEPS"),
        train_logging_steps=env_value("TRAIN_LOGGING_STEPS"),
        mrd_sigma=env_value("MRD_SIGMA"),
        mrd_num_mc_samples=env_value("MRD_NUM_MC_SAMPLES"),
        mrd_batch_size=env_value("MRD_BATCH_SIZE"),
        mrd_eps=env_value("MRD_EPS"),
        gpu_id=env_value("GPU_ID"),
        resume=is_truthy(env_value("RESUME")),
    )


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
        "+trainer.args.save_total_limit=1",
        "trainer.args.save_only_model=false",
        "+trainer.args.ignore_data_skip=true",
    ]


def run_mrd_scoring(cfg: RunConfig, round_id: int, model_path: str) -> None:
    weights_path = cfg.round_weights_path(round_id)
    if cfg.resume and weights_path.is_file():
        log(f"Skipping MRD scoring for round {round_id + 1}; found {weights_path}.")
        return

    run_repo_python(
        [
            "src/mrd.py",
            "experiment=mrd/tofu/npo",
            f"task_name={cfg.round_mrd_task_name(round_id)}",
            f"paths.output_dir={cfg.round_mrd_dir(round_id)}",
            f"model={cfg.model}",
            f"forget_split={cfg.forget_split}",
            f"retain_split={cfg.retain_split}",
            f"model.model_args.pretrained_model_name_or_path={model_path}",
            f"model.tokenizer_args.pretrained_model_name_or_path={cfg.base_model_path}",
            f"mrd.sigma={cfg.mrd_sigma}",
            f"mrd.num_mc_samples={cfg.mrd_num_mc_samples}",
            f"mrd.batch_size={cfg.mrd_batch_size}",
            f"mrd.eps={cfg.mrd_eps}",
        ],
        extra_env={"CUDA_VISIBLE_DEVICES": cfg.gpu_id},
    )


def run_training_round(
    cfg: RunConfig,
    round_id: int,
    model_path: str,
    resume_checkpoint: Path | None = None,
) -> None:
    output_dir = cfg.round_output_dir(round_id)
    weights_path = cfg.round_weights_path(round_id)
    extra_args = []
    if resume_checkpoint is not None:
        extra_args.append(f"resume_from_checkpoint={resume_checkpoint}")

    run_repo_python(
        [
            "src/train.py",
            "--config-name=unlearn.yaml",
            "experiment=unlearn/tofu/mrd_npo",
            "trainer=NPO",
            f"task_name={cfg.round_task_name(round_id)}",
            *common_train_args(cfg, model_path),
            f"trainer.args.num_train_epochs={cfg.round_epochs(round_id)}",
            f"trainer.train_sampler_args.weights_path={weights_path}",
            "trainer.train_sampler_args.replacement=true",
            *extra_args,
            f"paths.output_dir={output_dir}",
        ],
        extra_env={"CUDA_VISIBLE_DEVICES": cfg.gpu_id},
    )


def main() -> None:
    cfg = load_config()
    if cfg.refresh_epochs <= 0:
        raise SystemExit("REFRESH_EPOCHS must be positive.")

    current_model_path = cfg.base_model_path
    final_output_dir: Path | None = None

    for round_id in range(cfg.num_rounds):
        output_dir = cfg.round_output_dir(round_id)
        weights_path = cfg.round_weights_path(round_id)

        if cfg.resume and training_output_complete(output_dir):
            log(
                f"Skipping round {round_id + 1}; found completed training output at {output_dir}."
            )
            current_model_path = str(output_dir)
            final_output_dir = output_dir
            continue

        run_mrd_scoring(cfg, round_id, current_model_path)
        if not weights_path.is_file():
            raise SystemExit(
                f"Expected MRD weights file was not produced: {weights_path}"
            )

        resume_checkpoint = latest_checkpoint_in_dir(output_dir) if cfg.resume else None
        if resume_checkpoint is not None:
            log(f"Resuming round {round_id + 1} from checkpoint {resume_checkpoint}.")

        run_training_round(
            cfg=cfg,
            round_id=round_id,
            model_path=current_model_path,
            resume_checkpoint=resume_checkpoint,
        )
        current_model_path = str(output_dir)
        final_output_dir = output_dir

    if final_output_dir is None:
        raise SystemExit("MRD-NPO did not produce any training output.")

    log(f"MRD-NPO training completed. Final model output: {final_output_dir}")


if __name__ == "__main__":
    main()
