import gc

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from data import get_collators, get_datasets
from runtime_utils import seed_everything
from selective.score import build_difficulty_payload, score_dataset_with_reference
from selective.utils import (
    clone_dataset_with_indices,
    get_allowed_indices_from_manifest,
    load_json,
    load_reference_manifest,
    save_json,
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


def prepare_difficulty_payload(
    cfg: DictConfig,
    dataset,
    collator,
    target_model,
):
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
                "fold_id": reference_spec["fold_id"],
                "checkpoint_path": reference_spec["checkpoint_path"],
                "heldout_manifest_path": reference_spec["heldout_manifest_path"],
                "num_examples": len(heldout_indices),
            }
        )
        del reference_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        },
    )
    return difficulty_payload


@hydra.main(
    version_base=None, config_path="../configs", config_name="selective_prepare.yaml"
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    from model import get_model, get_tokenizer

    tokenizer = get_tokenizer(cfg.model.tokenizer_args)
    template_args = cfg.model.template_args
    dataset = get_datasets(cfg.data, tokenizer=tokenizer, template_args=template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    if isinstance(dataset, dict):
        raise ValueError(
            "selective_prepare expects exactly one forget dataset config in cfg.data"
        )

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


if __name__ == "__main__":
    main()
