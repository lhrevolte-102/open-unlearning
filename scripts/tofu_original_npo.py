#!/usr/bin/env python3

import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from selective.utils import build_stage_manifests_from_ordered_indices, load_json, save_json  # noqa: E402


VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
ORIGINAL_ROOT = ROOT_DIR / "saves" / "original"
UNLEARN_ROOT = ROOT_DIR / "saves" / "unlearn"

ENV_DEFAULTS = {
    "MODEL": "Llama-3.2-3B-Instruct",
    "FORGET_SPLIT": "forget10",
    "RETAIN_SPLIT": "retain90",
    "TOTAL_EPOCHS": "5",
    "STAGE_PERCENTILES": "",
    "STAGE_EPOCH_RATIOS": "",
    "STAGE_SUBSET_MODE": "cumulative",
    "STAGE_SPLIT_SEED": "0",
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
    forget_dataset_name: str


METHOD = MethodSpec(
    label="NPO",
    experiment="unlearn/tofu/default.yaml",
    trainer_name="NPO",
    task_suffix="npo_original",
    forget_dataset_name="TOFU_QA_forget",
)


@dataclass(frozen=True)
class RunConfig:
    model: str
    forget_split: str
    retain_split: str
    total_epochs_token: str
    total_epochs: float
    stage_percentiles_raw: str
    stage_epoch_ratios_raw: str
    stage_subset_mode: str
    stage_split_seed: int
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

    @property
    def base_task_name(self) -> str:
        return f"tofu_{self.model}_{self.forget_split}_{METHOD.task_suffix}"


@dataclass(frozen=True)
class StagePlan:
    epoch_ratios: list[float]
    stage_percentiles: list[float]
    staged: bool
    subset_staged: bool


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
        stage_percentiles_raw=env_value("STAGE_PERCENTILES"),
        stage_epoch_ratios_raw=env_value("STAGE_EPOCH_RATIOS"),
        stage_subset_mode=env_value("STAGE_SUBSET_MODE"),
        stage_split_seed=int(env_value("STAGE_SPLIT_SEED")),
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


def parse_stage_percentiles(cfg: RunConfig) -> list[float]:
    if not cfg.stage_percentiles_raw.strip():
        return []
    percentiles = [float(value) for value in json.loads(cfg.stage_percentiles_raw)]
    if not percentiles:
        raise SystemExit("STAGE_PERCENTILES must not be empty when provided.")
    return percentiles


def resolve_stage_plan(cfg: RunConfig) -> StagePlan:
    epoch_ratios = parse_stage_epoch_ratios(cfg)
    stage_percentiles = parse_stage_percentiles(cfg)
    staged = cfg.stage_epoch_ratios_raw.strip() != "" and len(epoch_ratios) > 1
    subset_staged = bool(stage_percentiles)
    if subset_staged and len(stage_percentiles) != len(epoch_ratios):
        raise SystemExit("STAGE_PERCENTILES and STAGE_EPOCH_RATIOS must have same length.")
    if subset_staged and not staged:
        raise SystemExit("STAGE_PERCENTILES requires multi-stage STAGE_EPOCH_RATIOS.")
    if subset_staged and cfg.stage_subset_mode not in {"cumulative", "disjoint"}:
        raise SystemExit("STAGE_SUBSET_MODE must be 'cumulative' or 'disjoint'.")
    return StagePlan(
        epoch_ratios=epoch_ratios,
        stage_percentiles=stage_percentiles,
        staged=staged,
        subset_staged=subset_staged,
    )


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
    print(f"[Original-{METHOD.label}] {message}", flush=True)


def run_repo_python(args: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run([str(VENV_PYTHON), *args], cwd=ROOT_DIR, env=env, check=True)


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


def plan_task_prefix(cfg: RunConfig, plan: StagePlan) -> str:
    if not plan.staged and not plan.subset_staged:
        return cfg.base_task_name
    staged_base = f"{cfg.base_task_name}_beta{cfg.beta}_epoch{cfg.total_epochs_token}"
    if plan.subset_staged:
        return (
            f"{staged_base}_{cfg.stage_subset_mode}_{format_float_list_suffix(plan.stage_percentiles, 'pct')}_"
            f"{format_float_list_suffix(plan.epoch_ratios, 'ratio')}"
        )
    return f"{staged_base}_{format_float_list_suffix(plan.epoch_ratios, 'ratio')}"


def stage_manifest_dir(cfg: RunConfig, plan: StagePlan) -> Path:
    return ORIGINAL_ROOT / "stage" / f"{plan_task_prefix(cfg, plan)}_stages" / "stages"


def build_original_stage_manifests(cfg: RunConfig, plan: StagePlan) -> list[Path]:
    from datasets import load_dataset

    stage_dir = stage_manifest_dir(cfg, plan)
    stage_dir.mkdir(parents=True, exist_ok=True)
    forget_dataset = load_dataset("locuslab/TOFU", name=cfg.forget_split, split="train")
    ordered_indices = list(range(len(forget_dataset)))
    random.Random(cfg.stage_split_seed).shuffle(ordered_indices)
    manifests = build_stage_manifests_from_ordered_indices(
        ordered_indices=ordered_indices,
        stage_percentiles=plan.stage_percentiles,
        stage_epoch_ratios=plan.epoch_ratios,
        stage_subset_mode=cfg.stage_subset_mode,
    )
    summary = {
        "method": METHOD.label,
        "model": cfg.model,
        "forget_split": cfg.forget_split,
        "retain_split": cfg.retain_split,
        "stage_subset_mode": cfg.stage_subset_mode,
        "stage_percentiles": plan.stage_percentiles,
        "stage_epoch_ratios": plan.epoch_ratios,
        "stage_split_seed": cfg.stage_split_seed,
        "stages": [],
    }
    for manifest in manifests:
        manifest.update(
            {"method": METHOD.label, "ordering": "random_seeded_original", "stage_split_seed": cfg.stage_split_seed}
        )
        output_path = stage_dir / f"{manifest['stage_name']}.json"
        save_json(output_path, manifest)
        summary["stages"].append(
            {
                "stage_name": manifest["stage_name"],
                "output_path": str(output_path.resolve()),
                "num_examples": manifest["num_examples"],
                "percentile": manifest["percentile"],
                "epoch_ratio": manifest["epoch_ratio"],
            }
        )
    save_json(stage_dir / "stages.json", summary)
    return sorted(stage_dir.glob("stage[0-9]*.json"))


def run_single_stage(cfg: RunConfig, task_name: str) -> str:
    output_dir = UNLEARN_ROOT / task_name
    if cfg.resume and training_output_complete(output_dir):
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


def run_multi_stage_full_data(cfg: RunConfig, task_prefix: str, stage_epoch_ratios: list[float]) -> str:
    stage_limit = len(stage_epoch_ratios) if cfg.max_stage_id <= 0 else min(cfg.max_stage_id, len(stage_epoch_ratios))
    prev_output_dir: Path | None = None
    final_task_name = ""
    for stage_id, epoch_ratio in enumerate(stage_epoch_ratios[:stage_limit], start=1):
        stage_name = f"stage{stage_id}"
        task_name = f"{task_prefix}_{stage_name}"
        output_dir = UNLEARN_ROOT / task_name
        final_task_name = task_name
        if cfg.resume and training_output_complete(output_dir):
            log(f"Skipping {METHOD.label} {stage_name}; found completed model output at {output_dir}.")
            prev_output_dir = output_dir
            continue
        extra_args: list[str] = []
        pretrained_model_path = cfg.base_model_path
        if cfg.resume:
            latest_checkpoint = latest_checkpoint_in_dir(output_dir)
            if latest_checkpoint is not None:
                log(f"Resuming {METHOD.label} {stage_name} from in-progress checkpoint {latest_checkpoint}.")
                extra_args.append(f"resume_from_checkpoint={latest_checkpoint}")
        if not extra_args and prev_output_dir is not None:
            latest_checkpoint = latest_checkpoint_in_dir(prev_output_dir)
            if latest_checkpoint is None:
                raise SystemExit(f"No checkpoint found under {prev_output_dir} for {METHOD.label} {stage_name} resume.")
            log(f"Initializing {METHOD.label} {stage_name} from previous stage model {latest_checkpoint}.")
            pretrained_model_path = str(latest_checkpoint)
        run_train_command(
            cfg=cfg,
            task_name=task_name,
            output_dir=output_dir,
            num_train_epochs=str(max(epoch_ratio * cfg.total_epochs, 1.0)),
            pretrained_model_path=pretrained_model_path,
            extra_args=extra_args,
        )
        prev_output_dir = output_dir
    if not final_task_name:
        raise SystemExit(f"No stages were executed for {METHOD.label}.")
    return final_task_name


def run_multi_stage_subset(cfg: RunConfig, task_prefix: str, stage_manifests: list[Path]) -> str:
    stage_limit = len(stage_manifests) if cfg.max_stage_id <= 0 else min(cfg.max_stage_id, len(stage_manifests))
    prev_output_dir: Path | None = None
    final_task_name = ""
    for stage_manifest in stage_manifests[:stage_limit]:
        manifest = load_json(stage_manifest)
        stage_name = manifest["stage_name"]
        task_name = f"{task_prefix}_{stage_name}"
        output_dir = UNLEARN_ROOT / task_name
        final_task_name = task_name
        if cfg.resume and training_output_complete(output_dir):
            log(f"Skipping {METHOD.label} {stage_name}; found completed model output at {output_dir}.")
            prev_output_dir = output_dir
            continue
        extra_args: list[str] = [
            f"data.forget.{METHOD.forget_dataset_name}.args.allowed_indices_path={stage_manifest}"
        ]
        pretrained_model_path = cfg.base_model_path
        if cfg.resume:
            latest_checkpoint = latest_checkpoint_in_dir(output_dir)
            if latest_checkpoint is not None:
                log(f"Resuming {METHOD.label} {stage_name} from in-progress checkpoint {latest_checkpoint}.")
                extra_args.append(f"resume_from_checkpoint={latest_checkpoint}")
        if len(extra_args) == 1 and prev_output_dir is not None:
            latest_checkpoint = latest_checkpoint_in_dir(prev_output_dir)
            if latest_checkpoint is None:
                raise SystemExit(f"No checkpoint found under {prev_output_dir} for {METHOD.label} {stage_name} resume.")
            log(f"Initializing {METHOD.label} {stage_name} from previous stage model {latest_checkpoint}.")
            pretrained_model_path = str(latest_checkpoint)
        run_train_command(
            cfg=cfg,
            task_name=task_name,
            output_dir=output_dir,
            num_train_epochs=str(max(float(manifest["epoch_ratio"]) * cfg.total_epochs, 1.0)),
            pretrained_model_path=pretrained_model_path,
            extra_args=extra_args,
        )
        prev_output_dir = output_dir
    if not final_task_name:
        raise SystemExit(f"No stages were executed for {METHOD.label}.")
    return final_task_name


def run_unlearn(cfg: RunConfig, plan: StagePlan) -> str:
    task_prefix = plan_task_prefix(cfg, plan)
    if not plan.staged:
        return run_single_stage(cfg, task_prefix)
    if not plan.subset_staged:
        return run_multi_stage_full_data(cfg, task_prefix, plan.epoch_ratios)
    return run_multi_stage_subset(cfg, task_prefix, build_original_stage_manifests(cfg, plan))


def run_eval(cfg: RunConfig, task_name: str) -> None:
    output_dir = UNLEARN_ROOT / task_name
    eval_dir = output_dir / "evals"
    eval_file = eval_dir / "TOFU_EVAL.json"
    if cfg.resume and tofu_eval_has_full_metrics(eval_file):
        log(f"Skipping {METHOD.label} eval; found existing full-metric eval logs at {eval_file}.")
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
    plan = resolve_stage_plan(cfg)
    if cfg.resume:
        log("Resume mode enabled. Completed unlearning runs and full-metric evals will be skipped.")
    else:
        log("Resume mode disabled. Existing outputs will be reused only when the underlying command does so implicitly.")
    if plan.staged:
        suffix = f" and MAX_STAGE_ID={cfg.max_stage_id}" if cfg.max_stage_id > 0 else ""
        if plan.subset_staged:
            log(
                "Subset-staged original training enabled with "
                f"STAGE_SUBSET_MODE={cfg.stage_subset_mode}, "
                f"STAGE_PERCENTILES={plan.stage_percentiles}, "
                f"STAGE_EPOCH_RATIOS={plan.epoch_ratios}, "
                f"STAGE_SPLIT_SEED={cfg.stage_split_seed}{suffix}."
            )
        else:
            log(f"Staged original training enabled with STAGE_EPOCH_RATIOS={plan.epoch_ratios}{suffix}.")
    final_task_name = run_unlearn(cfg, plan)
    run_eval(cfg, final_task_name)
    log(f"{METHOD.label} output: saves/unlearn/{final_task_name}")


if __name__ == "__main__":
    main()
