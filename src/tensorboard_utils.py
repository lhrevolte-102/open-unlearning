from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def get_tensorboard_log_dir(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser() / "logs"


def create_tensorboard_writer(output_dir: str | Path) -> SummaryWriter:
    log_dir = get_tensorboard_log_dir(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _coerce_scalar(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            scalar = value.item()
        except (TypeError, ValueError):
            return None
        return _coerce_scalar(scalar)
    return None


def log_tensorboard_metrics(
    writer: SummaryWriter,
    metrics: dict[str, Any],
    *,
    prefix: str | None = None,
    step: int | None = None,
) -> None:
    for key, value in metrics.items():
        scalar = _coerce_scalar(value)
        if scalar is None:
            continue
        tag = f"{prefix}/{key}" if prefix else key
        writer.add_scalar(tag, scalar, global_step=step)
    writer.flush()
