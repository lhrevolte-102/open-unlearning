import gc
import warnings

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from data import get_collators, get_datasets
from selective.score import build_difficulty_payload, score_dataset_with_reference
from selective.utils import (
    build_reference_manifest,
    build_reference_split_manifests,
    clone_dataset_with_indices,
    get_allowed_indices_from_manifest,
    load_json,
    load_reference_manifest,
    save_json,
    build_stage_manifests,
)


def _load_model_from_path(model_cfg, model_path):
    from model import get_model

    runtime_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=False))
    with open_dict(runtime_cfg.model_args):
        runtime_cfg.model_args.pretrained_model_name_or_path = model_path
    model, _ = get_model(runtime_cfg)
    return model


def _prepare_model_for_scoring(model):
    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model = model.to(runtime_device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _require_single_dataset(dataset, context):
    if isinstance(dataset, dict):
        raise ValueError(f"{context} expects exactly one forget dataset config in cfg.data")
    return dataset


def build_reference_artifacts(cfg: DictConfig):
    from model import get_tokenizer

    tokenizer = get_tokenizer(cfg.model.tokenizer_args)
    template_args = cfg.model.template_args
    dataset = get_datasets(cfg.data, tokenizer=tokenizer, template_args=template_args)
    dataset = _require_single_dataset(dataset, "selective_reference")

    all_indices = [int(idx) for idx in dataset.data["index"]]
    reference_split_manifests = build_reference_split_manifests(
        all_indices=all_indices,
        output_dir=cfg.reference_splits_output_dir,
        num_repeats=cfg.num_repeats,
        repeat_split_seed=cfg.repeat_split_seed,
    )

    reference_splits_summary = {
        "metadata": {
            "method": cfg.method,
            "model": cfg.model.model_args.pretrained_model_name_or_path,
            "forget_split": cfg.get("forget_split", None),
            "retain_split": cfg.get("retain_split", None),
            "num_repeats": cfg.num_repeats,
            "repeat_split_seed": cfg.repeat_split_seed,
            "all_indices": all_indices,
        },
        "reference_splits": [
            {
                "split_id": split["split_id"],
                "split_name": split["split_name"],
                "split_seed": split["split_seed"],
                "repeat_id": split.get("repeat_id", None),
                "partition_id": split.get("partition_id", None),
                "train_manifest_path": split["train_manifest_path"],
                "heldout_manifest_path": split["heldout_manifest_path"],
                "num_train_examples": len(split["train_indices"]),
                "num_heldout_examples": len(split["heldout_indices"]),
            }
            for split in reference_split_manifests
        ],
    }
    save_json(cfg.reference_splits_summary_path, reference_splits_summary)

    reference_manifest = build_reference_manifest(
        metadata={
            "method": cfg.method,
            "model": cfg.model.model_args.pretrained_model_name_or_path,
            "forget_split": cfg.get("forget_split", None),
            "retain_split": cfg.get("retain_split", None),
            "num_repeats": cfg.num_repeats,
            "repeat_split_seed": cfg.repeat_split_seed,
            "all_indices": all_indices,
        },
        reference_split_manifests=reference_split_manifests,
        checkpoint_root_dir=cfg.checkpoint_root_dir,
        reference_manifest_output_path=cfg.reference_manifest_output_path,
        validate_checkpoint_paths=cfg.validate_checkpoint_paths,
    )
    return reference_splits_summary, reference_manifest


def prepare_difficulty_payload(cfg: DictConfig, dataset, collator, target_model):
    all_indices = [int(idx) for idx in dataset.data["index"]]
    reference_manifest = load_reference_manifest(cfg.reference_manifest_path)
    reference_specs = reference_manifest["references"]

    score_records = {}
    scored_reference_specs = []
    for reference_spec in reference_specs:
        heldout_manifest = load_json(reference_spec["heldout_manifest_path"])
        heldout_indices = get_allowed_indices_from_manifest(heldout_manifest)
        if not heldout_indices:
            continue

        reference_model = _prepare_model_for_scoring(
            _load_model_from_path(cfg.model, reference_spec["checkpoint_path"])
        )
        dataset_subset = clone_dataset_with_indices(dataset, heldout_indices)
        fold_scores = score_dataset_with_reference(
            model=reference_model,
            ref_model=target_model,
            dataset=dataset_subset,
            collator=collator,
            method=cfg.method,
            beta=cfg.beta,
            batch_size=cfg.batch_size,
        )
        for idx, scores in fold_scores.items():
            score_records.setdefault(idx, []).extend(scores)

        scored_reference_specs.append(
            {
                "split_id": reference_spec["split_id"],
                "split_name": reference_spec["split_name"],
                "split_seed": reference_spec["split_seed"],
                "repeat_id": reference_spec.get("repeat_id", None),
                "partition_id": reference_spec.get("partition_id", None),
                "checkpoint_path": reference_spec["checkpoint_path"],
                "heldout_manifest_path": reference_spec["heldout_manifest_path"],
                "num_examples": len(heldout_indices),
            }
        )
        del reference_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scored_indices = sorted(int(idx) for idx in score_records.keys())
    unscored_indices = sorted(set(all_indices) - set(scored_indices))
    if unscored_indices:
        warnings.warn(
            "Difficulty scoring left some forget examples unscored; "
            f"{len(unscored_indices)} examples will fall back to deterministic tail "
            "ordering in stage construction.",
            stacklevel=2,
        )

    difficulty_payload = build_difficulty_payload(
        score_records=score_records,
        metadata={
            "method": cfg.method,
            "forget_split": cfg.get("forget_split", None),
            "model": cfg.model.model_args.pretrained_model_name_or_path,
            "reference_manifest_path": cfg.reference_manifest_path,
            "num_reference_models": len(scored_reference_specs),
            "reference_metadata": reference_manifest["metadata"],
            "references": scored_reference_specs,
            "all_indices": all_indices,
            "num_scored_examples": len(scored_indices),
            "num_unscored_examples": len(unscored_indices),
            "coverage_fraction": (
                float(len(scored_indices) / len(all_indices)) if all_indices else 0.0
            ),
            "unscored_indices": unscored_indices,
        },
    )
    return difficulty_payload


def build_difficulty_artifacts(cfg: DictConfig):
    from model import get_model, get_tokenizer

    tokenizer = get_tokenizer(cfg.model.tokenizer_args)
    template_args = cfg.model.template_args
    dataset = get_datasets(cfg.data, tokenizer=tokenizer, template_args=template_args)
    dataset = _require_single_dataset(dataset, "selective_prepare")
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    target_model, _ = get_model(
        OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=False))
    )
    target_model = _prepare_model_for_scoring(target_model)

    difficulty_payload = prepare_difficulty_payload(
        cfg=cfg,
        dataset=dataset,
        collator=collator,
        target_model=target_model,
    )
    save_json(cfg.score_output_path, difficulty_payload)
    return difficulty_payload


def build_stage_artifacts(cfg: DictConfig):
    difficulty_payload = load_json(cfg.difficulty_path)
    manifests = build_stage_manifests(
        difficulty_payload=difficulty_payload,
        stage_percentiles=list(cfg.stage_percentiles),
        stage_epoch_ratios=list(cfg.stage_epoch_ratios),
    )

    summary = {
        "difficulty_path": cfg.difficulty_path,
        "intra_stage_order": cfg.intra_stage_order,
        "stage_percentiles": list(cfg.stage_percentiles),
        "stage_epoch_ratios": list(cfg.stage_epoch_ratios),
        "stages": [],
    }

    for manifest in manifests:
        manifest["intra_stage_order"] = cfg.intra_stage_order
        output_path = f"{cfg.output_dir}/{manifest['stage_name']}.json"
        save_json(output_path, manifest)
        summary["stages"].append(
            {
                "stage_name": manifest["stage_name"],
                "output_path": output_path,
                "num_examples": manifest["num_examples"],
                "percentile": manifest["percentile"],
                "epoch_ratio": manifest["epoch_ratio"],
                "intra_stage_order": cfg.intra_stage_order,
            }
        )

    save_json(f"{cfg.output_dir}/stages.json", summary)
    return summary


def build_selective_step_cfg(cfg: DictConfig, step_name: str):
    step_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    step_overrides = step_cfg.get(step_name, {})
    with open_dict(step_cfg):
        for key, value in step_overrides.items():
            step_cfg[key] = value
    return step_cfg


def _normalize_pipeline_steps(steps):
    normalized = [str(step) for step in steps]
    valid_steps = {"reference", "prepare", "stage"}
    invalid_steps = [step for step in normalized if step not in valid_steps]
    if invalid_steps:
        raise ValueError(
            f"Unsupported pipeline_steps {invalid_steps}; expected values from {sorted(valid_steps)}"
        )
    return normalized


def run_selective_pipeline(cfg: DictConfig):
    results = {}
    pipeline_steps = _normalize_pipeline_steps(cfg.pipeline_steps)

    if "reference" in pipeline_steps:
        reference_cfg = build_selective_step_cfg(cfg, "reference")
        reference_summary, reference_manifest = build_reference_artifacts(reference_cfg)
        results["reference"] = {
            "summary": reference_summary,
            "manifest": reference_manifest,
            "reference_manifest_path": reference_cfg.reference_manifest_output_path,
        }

    if "prepare" in pipeline_steps:
        prepare_cfg = build_selective_step_cfg(cfg, "prepare")
        difficulty_payload = build_difficulty_artifacts(prepare_cfg)
        results["prepare"] = {
            "difficulty_payload": difficulty_payload,
            "score_output_path": prepare_cfg.score_output_path,
        }

    if "stage" in pipeline_steps:
        stage_cfg = build_selective_step_cfg(cfg, "stage")
        stage_summary = build_stage_artifacts(stage_cfg)
        results["stage"] = {
            "summary": stage_summary,
            "output_dir": stage_cfg.output_dir,
        }

    return results
