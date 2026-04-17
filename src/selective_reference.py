import hydra
from omegaconf import DictConfig

from data import get_datasets
from runtime_utils import seed_everything
from selective.utils import (
    build_reference_manifest,
    build_reference_split_manifests,
    save_json,
)


def build_reference_artifacts(cfg: DictConfig):
    from model import get_tokenizer

    tokenizer = get_tokenizer(cfg.model.tokenizer_args)
    template_args = cfg.model.template_args
    dataset = get_datasets(cfg.data, tokenizer=tokenizer, template_args=template_args)

    if isinstance(dataset, dict):
        raise ValueError(
            "selective_reference expects exactly one forget dataset config in cfg.data"
        )

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


@hydra.main(
    version_base=None, config_path="../configs", config_name="selective_reference.yaml"
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    build_reference_artifacts(cfg)


if __name__ == "__main__":
    main()
