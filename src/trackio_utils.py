import os
import logging
from numbers import Number
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKIO_DIR = REPO_ROOT / "saves" / "trackio"
DEFAULT_GRADIO_TEMP_DIR = REPO_ROOT / "saves" / "gradio_tmp"


def _set_default_dir_env(env_name: str, default_dir: Path) -> str:
    target_dir = Path(os.environ.get(env_name) or default_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_name, str(target_dir))
    return str(target_dir)


def _prepare_trackio_environment(trackio_dir: str | None = None) -> None:
    default_trackio_dir = (
        Path(trackio_dir).expanduser() if trackio_dir else DEFAULT_TRACKIO_DIR
    )
    _set_default_dir_env("TRACKIO_DIR", default_trackio_dir)
    _set_default_dir_env("GRADIO_TEMP_DIR", DEFAULT_GRADIO_TEMP_DIR)


def is_trackio_enabled(cfg: DictConfig) -> bool:
    trackio_cfg = cfg.get("trackio", None)
    return bool(trackio_cfg and trackio_cfg.get("enabled", False))


def _import_trackio():
    _prepare_trackio_environment()
    try:
        import trackio
    except ImportError as exc:
        raise ImportError(
            "Trackio is enabled but the `trackio` package is not installed."
        ) from exc
    return trackio


def _build_run_config(cfg: DictConfig) -> dict[str, Any]:
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved_cfg, dict)

    model_cfg = resolved_cfg.get("model", {})
    trainer_cfg = resolved_cfg.get("trainer", {})
    paths_cfg = resolved_cfg.get("paths", {})

    return {
        "mode": resolved_cfg.get("mode"),
        "task_name": resolved_cfg.get("task_name"),
        "model_path": model_cfg.get("model_args", {}).get(
            "pretrained_model_name_or_path"
        ),
        "trainer": trainer_cfg.get("handler"),
        "trainer_args": trainer_cfg.get("args", {}),
        "trainer_method_args": trainer_cfg.get("method_args", {}),
        "forget_split": resolved_cfg.get("forget_split"),
        "retain_split": resolved_cfg.get("retain_split"),
        "data_split": resolved_cfg.get("data_split"),
        "output_dir": paths_cfg.get("output_dir"),
    }


def init_trackio_run(cfg: DictConfig) -> bool:
    if not is_trackio_enabled(cfg):
        return False

    trackio_cfg = cfg.trackio
    local_dir = trackio_cfg.get("local_dir")
    _prepare_trackio_environment(trackio_dir=local_dir)

    trackio = _import_trackio()

    init_kwargs: dict[str, Any] = {
        "project": trackio_cfg.get("project") or "open-unlearning",
        "name": trackio_cfg.get("name") or cfg.get("task_name"),
        "group": trackio_cfg.get("group"),
        "config": _build_run_config(cfg),
    }

    for key in (
        "space_id",
        "dataset_id",
        "webhook_url",
        "webhook_min_level",
    ):
        value = trackio_cfg.get(key)
        if value not in (None, ""):
            init_kwargs[key] = value

    auto_log_gpu = trackio_cfg.get("auto_log_gpu")
    if auto_log_gpu is not None:
        init_kwargs["auto_log_gpu"] = auto_log_gpu

    gpu_log_interval = trackio_cfg.get("gpu_log_interval")
    if gpu_log_interval is not None:
        init_kwargs["gpu_log_interval"] = gpu_log_interval

    trackio.init(**init_kwargs)
    return True


def log_trackio_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    if not metrics:
        return

    payload = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, Number) and not isinstance(value, bool)
    }
    if not payload:
        return

    trackio = _import_trackio()
    trackio.log(payload, step=step)


def emit_trackio_alert(title: str, text: str, level: str = "ERROR") -> None:
    trackio = _import_trackio()
    trackio.alert(title=title, text=text, level=level)


def finish_trackio_run() -> None:
    trackio = _import_trackio()
    trackio.finish()
    if hasattr(trackio, "context_vars"):
        trackio.context_vars.current_run.set(None)
