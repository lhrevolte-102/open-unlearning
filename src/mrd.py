import hydra
from omegaconf import DictConfig

from data import get_collators, get_datasets
from model import get_model
from mrd_utils import (
    build_mrd_payload,
    infer_checkpoint_step,
    save_json,
    score_dataset_with_mrd,
)
from utils.runtime import seed_everything


def _require_single_dataset(dataset):
    if isinstance(dataset, dict):
        raise ValueError("MRD scoring expects exactly one forget dataset config in cfg.data")
    return dataset


@hydra.main(version_base=None, config_path="../configs", config_name="mrd.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    model_cfg = cfg.model
    model, tokenizer = get_model(model_cfg)
    dataset = get_datasets(
        cfg.data, tokenizer=tokenizer, template_args=model_cfg.template_args
    )
    dataset = _require_single_dataset(dataset)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    scores_by_index = score_dataset_with_mrd(
        model=model,
        dataset=dataset,
        collator=collator,
        batch_size=cfg.mrd.batch_size,
        sigma=cfg.mrd.sigma,
        num_mc_samples=cfg.mrd.num_mc_samples,
        eps=cfg.mrd.eps,
        seed=cfg.seed,
    )
    output_path = cfg.mrd.output_path
    payload = build_mrd_payload(
        scores_by_index=scores_by_index,
        metadata={
            "model_path": model_cfg.model_args.pretrained_model_name_or_path,
            "forget_split": cfg.get("forget_split", None),
            "retain_split": cfg.get("retain_split", None),
            "checkpoint_step": infer_checkpoint_step(
                model_cfg.model_args.pretrained_model_name_or_path
            ),
            "sigma": float(cfg.mrd.sigma),
            "num_mc_samples": int(cfg.mrd.num_mc_samples),
            "batch_size": int(cfg.mrd.batch_size),
            "eps": float(cfg.mrd.eps),
            "num_examples": len(dataset),
        },
    )
    save_json(output_path, payload)


if __name__ == "__main__":
    main()
