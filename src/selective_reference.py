import hydra
from omegaconf import DictConfig

from data import get_datasets
from runtime_utils import seed_everything
from selective.utils import build_fold_manifests, build_reference_manifest, save_json


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
    fold_manifests = build_fold_manifests(
        all_indices=all_indices,
        num_folds=cfg.num_folds,
        fold_assignment_seed=cfg.fold_assignment_seed,
        output_dir=cfg.folds_output_dir,
    )

    folds_summary = {
        "metadata": {
            "method": cfg.method,
            "model": cfg.model.model_args.pretrained_model_name_or_path,
            "forget_split": cfg.get("forget_split", None),
            "retain_split": cfg.get("retain_split", None),
            "num_folds": cfg.num_folds,
            "fold_assignment_seed": cfg.fold_assignment_seed,
            "all_indices": all_indices,
        },
        "folds": [
            {
                "fold_id": fold["fold_id"],
                "train_manifest_path": fold["train_manifest_path"],
                "heldout_manifest_path": fold["heldout_manifest_path"],
                "num_train_examples": len(fold["train_indices"]),
                "num_heldout_examples": len(fold["heldout_indices"]),
            }
            for fold in fold_manifests
        ],
    }
    save_json(cfg.folds_summary_path, folds_summary)

    reference_manifest = build_reference_manifest(
        metadata={
            "method": cfg.method,
            "model": cfg.model.model_args.pretrained_model_name_or_path,
            "forget_split": cfg.get("forget_split", None),
            "retain_split": cfg.get("retain_split", None),
            "num_folds": cfg.num_folds,
            "fold_assignment_seed": cfg.fold_assignment_seed,
            "all_indices": all_indices,
        },
        fold_manifests=fold_manifests,
        checkpoint_root_dir=cfg.checkpoint_root_dir,
        reference_manifest_output_path=cfg.reference_manifest_output_path,
        validate_checkpoint_paths=cfg.validate_checkpoint_paths,
    )
    return folds_summary, reference_manifest


@hydra.main(
    version_base=None, config_path="../configs", config_name="selective_reference.yaml"
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    build_reference_artifacts(cfg)


if __name__ == "__main__":
    main()
