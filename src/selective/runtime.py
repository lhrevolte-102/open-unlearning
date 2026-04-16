import os

import torch


def _get_visible_world_size():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size


def apply_intra_stage_ordering(cfg):
    if "intra_stage_order" not in cfg:
        return cfg

    mode = cfg.get("intra_stage_order", "random")
    if mode not in {"random", "difficulty_strict"}:
        raise ValueError(
            f"Unsupported intra_stage_order '{mode}', expected 'random' or 'difficulty_strict'"
        )

    if "trainer" in cfg:
        cfg.trainer.train_sampler = "random"

    data_cfg = cfg.get("data")
    forget_cfg = data_cfg.get("forget") if data_cfg else None
    if forget_cfg:
        for dataset_cfg in forget_cfg.values():
            dataset_args = dataset_cfg.get("args")
            if dataset_args is not None:
                dataset_args.preserve_manifest_order = False

    if mode == "random":
        return cfg

    if data_cfg is None or data_cfg.get("anchor") != "forget":
        raise ValueError(
            "intra_stage_order=difficulty_strict requires data.anchor=forget"
        )

    if cfg.trainer.args.get("group_by_length", False):
        raise ValueError(
            "intra_stage_order=difficulty_strict is incompatible with trainer.args.group_by_length=true"
        )

    if _get_visible_world_size() != 1:
        raise ValueError(
            "intra_stage_order=difficulty_strict currently supports only single-process training"
        )

    if torch.cuda.device_count() > 1:
        raise ValueError(
            "intra_stage_order=difficulty_strict currently supports only a single visible GPU"
        )

    if "trainer" in cfg:
        cfg.trainer.train_sampler = "sequential"

    if forget_cfg:
        for dataset_cfg in forget_cfg.values():
            dataset_args = dataset_cfg.get("args")
            if dataset_args is not None:
                dataset_args.preserve_manifest_order = True

    return cfg
