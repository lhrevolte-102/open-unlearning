#!/usr/bin/env python3

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
UNLEARN_ROOT = ROOT_DIR / "saves" / "unlearn"
SELECTIVE_ROOT = ROOT_DIR / "saves" / "selective"


@dataclass(frozen=True)
class MethodSpec:
    label: str
    task_label: str
    selective_experiment: str
    unlearn_experiment: str
    trainer_name: str


METHOD = MethodSpec(
    label="Selective-DPO",
    task_label="Selective-DPO",
    selective_experiment="selective/tofu/idkdpo",
    unlearn_experiment="unlearn/tofu/selective_idk",
    trainer_name="DPO",
)

ENV_DEFAULTS = {
    "MODEL": "Llama-3.2-3B-Instruct",
    "FORGET_SPLIT": "forget10",
    "RETAIN_SPLIT": "retain90",
    "TOTAL_EPOCHS": "5",
    "STAGE_SUBSET_MODE": "cumulative",
    "STAGE_PERCENTILES": "[0.3,0.6,1.0]",
    "STAGE_EPOCH_RATIOS": "[0.2,0.4,0.4]",
    "INTRA_STAGE_ORDER": "random",
    "LEARNING_RATE": "1e-5",
    "BETA": "0.1",
    "ALPHA": "1",
    "PER_DEVICE_TRAIN_BATCH_SIZE": "16",
    "GRADIENT_ACCUMULATION_STEPS": "4",
    "NUM_REFERENCE_REPEATS": "3",
    "REPEAT_SPLIT_SEED": "0",
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


def env_value(name: str) -> str:
    return os.environ.get(name, ENV_DEFAULTS[name])


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_number_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), "g").replace(".", "p")


def format_number_list_suffix(values: str, prefix: str) -> str:
    return prefix + "-".join(compact_number_token(float(value)) for value in json.loads(values))


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


def selective_output_dir(category: str, task_name: str) -> Path:
    return SELECTIVE_ROOT / category / task_name


def tofu_eval_has_full_metrics(eval_file: Path) -> bool:
    if not eval_file.is_file():
        return False
    return REQUIRED_EVAL_METRICS.issubset(load_json(eval_file))


def log(message: str) -> None:
    print(f"[{METHOD.label}] {message}", flush=True)


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
    stage_subset_mode: str
    stage_percentiles: str
    stage_epoch_ratios: str
    intra_stage_order: str
    learning_rate: str
    beta: str
    alpha: str
    per_device_train_batch_size: str
    gradient_accumulation_steps: str
    num_reference_repeats: str
    repeat_split_seed: str
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
    def shared_task_config_suffix(self) -> str:
        return (
            f"{METHOD.task_label}_lr{self.learning_rate}_beta{self.beta}_alpha{self.alpha}_"
            f"epoch{self.total_epochs_token}_"
            f"{format_number_list_suffix(self.stage_percentiles, 'pct')}_"
            f"{format_number_list_suffix(self.stage_epoch_ratios, 'ratio')}"
        )

    @property
    def stage_task_config_suffix(self) -> str:
        return (
            f"{METHOD.task_label}_lr{self.learning_rate}_beta{self.beta}_alpha{self.alpha}_"
            f"epoch{self.total_epochs_token}_{self.stage_subset_mode}_"
            f"{format_number_list_suffix(self.stage_percentiles, 'pct')}_"
            f"{format_number_list_suffix(self.stage_epoch_ratios, 'ratio')}"
        )

    @property
    def task_prefix(self) -> str:
        return f"tofu_{self.model}_{self.forget_split}_{self.shared_task_config_suffix}"

    @property
    def stage_task_base_prefix(self) -> str:
        return f"tofu_{self.model}_{self.forget_split}_{self.stage_task_config_suffix}"

    @property
    def reference_task_prefix(self) -> str:
        return f"tofu_{self.model}_{self.forget_split}_references_{self.shared_task_config_suffix}"

    @property
    def reference_task_name(self) -> str:
        return self.reference_task_prefix

    @property
    def prepare_task_name(self) -> str:
        return f"{self.task_prefix}_prepare"

    @property
    def reference_dir(self) -> Path:
        return selective_output_dir("reference", self.reference_task_name)

    @property
    def reference_splits_dir(self) -> Path:
        return self.reference_dir / "reference_splits"

    @property
    def reference_models_dir(self) -> Path:
        return self.reference_dir / "models"

    @property
    def reference_manifest_path(self) -> Path:
        return self.reference_dir / "reference_models.json"

    @property
    def prepare_dir(self) -> Path:
        return selective_output_dir("prepare", self.prepare_task_name)

    @property
    def difficulty_path(self) -> Path:
        return self.prepare_dir / "difficulty.json"

    @property
    def stage_task_prefix(self) -> str:
        return f"{self.stage_task_base_prefix}_{self.intra_stage_order}"

    @property
    def stage_task_name(self) -> str:
        return f"{self.stage_task_prefix}_stages"

    @property
    def stage_dir(self) -> Path:
        return selective_output_dir("stage", self.stage_task_name) / "stages"


def load_config() -> RunConfig:
    total_epochs_token = env_value("TOTAL_EPOCHS")
    return RunConfig(
        model=env_value("MODEL"),
        forget_split=env_value("FORGET_SPLIT"),
        retain_split=env_value("RETAIN_SPLIT"),
        total_epochs_token=total_epochs_token,
        total_epochs=float(total_epochs_token),
        stage_subset_mode=env_value("STAGE_SUBSET_MODE"),
        stage_percentiles=env_value("STAGE_PERCENTILES"),
        stage_epoch_ratios=env_value("STAGE_EPOCH_RATIOS"),
        intra_stage_order=env_value("INTRA_STAGE_ORDER"),
        learning_rate=env_value("LEARNING_RATE"),
        beta=env_value("BETA"),
        alpha=env_value("ALPHA"),
        per_device_train_batch_size=env_value("PER_DEVICE_TRAIN_BATCH_SIZE"),
        gradient_accumulation_steps=env_value("GRADIENT_ACCUMULATION_STEPS"),
        num_reference_repeats=env_value("NUM_REFERENCE_REPEATS"),
        repeat_split_seed=env_value("REPEAT_SPLIT_SEED"),
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
        "trainer.args.logging_steps=1",
        "trainer.args.gradient_checkpointing=true",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
    ]


def run_selective_reference(cfg: RunConfig, validate_checkpoint_paths: bool) -> None:
    run_repo_python(
        [
            "src/selective.py",
            f"experiment={METHOD.selective_experiment}",
            "pipeline_steps=[reference]",
            f"task_name={cfg.reference_task_name}",
            f"paths.output_dir={cfg.reference_dir}",
            f"model={cfg.model}",
            f"forget_split={cfg.forget_split}",
            f"retain_split={cfg.retain_split}",
            f"model.model_args.pretrained_model_name_or_path={cfg.base_model_path}",
            f"model.tokenizer_args.pretrained_model_name_or_path={cfg.base_model_path}",
            f"reference.reference_splits_output_dir={cfg.reference_splits_dir}",
            f"reference.reference_splits_summary_path={cfg.reference_splits_dir / 'reference_splits.json'}",
            f"reference.checkpoint_root_dir={cfg.reference_models_dir}",
            f"reference.reference_manifest_output_path={cfg.reference_manifest_path}",
            f"num_repeats={cfg.num_reference_repeats}",
            f"repeat_split_seed={cfg.repeat_split_seed}",
            f"reference.validate_checkpoint_paths={'true' if validate_checkpoint_paths else 'false'}",
        ]
    )


def run_reference_training(cfg: RunConfig) -> None:
    train_manifests = sorted(cfg.reference_splits_dir.glob("split*_train.json"))
    if not train_manifests:
        raise SystemExit(f"No reference split manifests were found under {cfg.reference_splits_dir}.")

    for train_manifest in train_manifests:
        split_name = load_json(train_manifest)["split_name"]
        split_task_name = f"{cfg.reference_task_prefix}_{split_name}"
        split_output_dir = cfg.reference_models_dir / split_name

        if cfg.resume and training_output_complete(split_output_dir):
            log(f"Skipping reference {split_name}; found completed model output at {split_output_dir}.")
            continue

        run_repo_python(
            [
                "src/train.py",
                "--config-name=unlearn.yaml",
                f"experiment={METHOD.unlearn_experiment}",
                f"trainer={METHOD.trainer_name}",
                f"task_name={split_task_name}",
                *common_train_args(cfg, cfg.base_model_path),
                f"selective_manifest_path={train_manifest}",
                f"trainer.args.num_train_epochs={cfg.total_epochs_token}",
                "trainer.args.save_strategy=no",
                "trainer.args.save_only_model=true",
                f"paths.output_dir={split_output_dir}",
            ],
            extra_env={"CUDA_VISIBLE_DEVICES": cfg.gpu_id},
        )


def run_difficulty_prepare(cfg: RunConfig) -> None:
    if cfg.resume and cfg.difficulty_path.is_file():
        log(f"Skipping difficulty preparation; found existing score file at {cfg.difficulty_path}.")
        return

    run_repo_python(
        [
            "src/selective.py",
            f"experiment={METHOD.selective_experiment}",
            "pipeline_steps=[prepare]",
            f"task_name={cfg.prepare_task_name}",
            f"paths.output_dir={cfg.prepare_dir}",
            f"model={cfg.model}",
            f"forget_split={cfg.forget_split}",
            f"model.model_args.pretrained_model_name_or_path={cfg.base_model_path}",
            f"model.tokenizer_args.pretrained_model_name_or_path={cfg.base_model_path}",
            f"prepare.reference_manifest_path={cfg.reference_manifest_path}",
            f"prepare.score_output_path={cfg.difficulty_path}",
            f"beta={cfg.beta}",
        ]
    )


def run_stage_manifest_build(cfg: RunConfig) -> list[Path]:
    cfg.stage_dir.mkdir(parents=True, exist_ok=True)
    run_repo_python(
        [
            "src/selective.py",
            "pipeline_steps=[stage]",
            f"task_name={cfg.stage_task_name}",
            f"paths.output_dir={cfg.stage_dir.parent}",
            f"stage.difficulty_path={cfg.difficulty_path}",
            f"stage.output_dir={cfg.stage_dir}",
            f"stage.intra_stage_order={cfg.intra_stage_order}",
            f"stage.stage_subset_mode={cfg.stage_subset_mode}",
            f"stage.stage_percentiles={cfg.stage_percentiles}",
            f"stage.stage_epoch_ratios={cfg.stage_epoch_ratios}",
        ]
    )

    stage_manifests = sorted(cfg.stage_dir.glob("stage[0-9]*.json"))
    if not stage_manifests:
        raise SystemExit(f"No stage manifests were found under {cfg.stage_dir}.")
    return stage_manifests


def run_stage_eval(cfg: RunConfig, stage_task_name: str, stage_output_dir: Path, stage_name: str) -> None:
    eval_dir = stage_output_dir / "evals"
    eval_file = eval_dir / "TOFU_EVAL.json"
    if cfg.resume and tofu_eval_has_full_metrics(eval_file):
        log(
            f"Skipping eval for {stage_name} ({cfg.intra_stage_order}); found existing full-metric eval logs at {eval_file}."
        )
        return

    run_repo_python(
        [
            "src/eval.py",
            "experiment=eval/tofu/full",
            f"forget_split={cfg.forget_split}",
            f"model={cfg.model}",
            f"task_name={stage_task_name}",
            f"model.model_args.pretrained_model_name_or_path={stage_output_dir}",
            f"model.tokenizer_args.pretrained_model_name_or_path={cfg.base_model_path}",
            f"paths.output_dir={eval_dir}",
            f"retain_logs_path={cfg.retain_logs_path}",
            f"reference_model_path={cfg.retain_model_path}",
        ],
        extra_env={"CUDA_VISIBLE_DEVICES": cfg.gpu_id},
    )


def run_stage_training(cfg: RunConfig, stage_manifests: list[Path]) -> None:
    prev_output_dir: Path | None = None

    for stage_manifest in stage_manifests:
        manifest = load_json(stage_manifest)
        stage_name = manifest["stage_name"]
        stage_epochs = str(max(float(manifest["epoch_ratio"]) * cfg.total_epochs, 1.0))
        stage_task_name = f"{cfg.stage_task_prefix}_{stage_name}"
        stage_output_dir = UNLEARN_ROOT / stage_task_name

        if cfg.resume and training_output_complete(stage_output_dir):
            log(
                f"Skipping {stage_name} ({cfg.intra_stage_order}); found completed training output at {stage_output_dir}."
            )
            prev_output_dir = stage_output_dir
            run_stage_eval(cfg, stage_task_name, stage_output_dir, stage_name)
            continue

        extra_args: list[str] = []
        stage_model_path = cfg.base_model_path
        if cfg.resume:
            latest_checkpoint = latest_checkpoint_in_dir(stage_output_dir)
            if latest_checkpoint is not None:
                log(
                    f"Resuming {stage_name} ({cfg.intra_stage_order}) from in-progress checkpoint {latest_checkpoint}."
                )
                extra_args.append(f"resume_from_checkpoint={latest_checkpoint}")

        if not extra_args and prev_output_dir is not None:
            latest_checkpoint = latest_checkpoint_in_dir(prev_output_dir)
            if latest_checkpoint is None:
                raise SystemExit(f"No checkpoint found under {prev_output_dir} for {stage_name} resume.")
            log(f"Initializing {stage_name} ({cfg.intra_stage_order}) from previous stage model {latest_checkpoint}.")
            stage_model_path = str(latest_checkpoint)

        run_repo_python(
            [
                "src/train.py",
                "--config-name=unlearn.yaml",
                f"experiment={METHOD.unlearn_experiment}",
                f"trainer={METHOD.trainer_name}",
                f"task_name={stage_task_name}",
                *common_train_args(cfg, stage_model_path),
                f"intra_stage_order={cfg.intra_stage_order}",
                f"selective_manifest_path={stage_manifest}",
                f"trainer.args.num_train_epochs={stage_epochs}",
                "+trainer.args.logging_first_step=true",
                "trainer.args.save_strategy=epoch",
                "+trainer.args.save_total_limit=1",
                "trainer.args.save_only_model=false",
                "+trainer.args.ignore_data_skip=true",
                *extra_args,
                f"paths.output_dir={stage_output_dir}",
            ],
            extra_env={"CUDA_VISIBLE_DEVICES": cfg.gpu_id},
        )
        prev_output_dir = stage_output_dir
        run_stage_eval(cfg, stage_task_name, stage_output_dir, stage_name)

    log(f"{METHOD.label} ({cfg.intra_stage_order}) training completed. Final stage output: {prev_output_dir}")


def prepare_workspace(cfg: RunConfig) -> None:
    cfg.reference_splits_dir.mkdir(parents=True, exist_ok=True)
    cfg.reference_models_dir.mkdir(parents=True, exist_ok=True)
    cfg.difficulty_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = load_config()
    prepare_workspace(cfg)

    if cfg.resume:
        log(
            "Resume mode enabled. Completed reference split training, difficulty scoring, stage training, and evals will be skipped."
        )
    else:
        log(
            "Resume mode disabled. Existing outputs will be reused only when the underlying command does so implicitly."
        )

    run_selective_reference(cfg, validate_checkpoint_paths=False)
    run_reference_training(cfg)
    run_selective_reference(cfg, validate_checkpoint_paths=True)
    run_difficulty_prepare(cfg)
    run_stage_training(cfg, run_stage_manifest_build(cfg))


if __name__ == "__main__":
    main()
