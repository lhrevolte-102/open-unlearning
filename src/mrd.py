import hydra
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf, open_dict

from data import get_collators, get_datasets
from model import get_model
from utils.mrd import (
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


def _load_runtime_model(model_cfg: DictConfig):
    runtime_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=False))
    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open_dict(runtime_cfg.model_args):
        if runtime_device.type == "cuda":
            runtime_cfg.model_args.setdefault("low_cpu_mem_usage", True)
            runtime_cfg.model_args.setdefault("device_map", "auto")
        elif runtime_cfg.model_args.get("attn_implementation") == "flash_attention_2":
            runtime_cfg.model_args.attn_implementation = "eager"

    model, tokenizer = get_model(runtime_cfg)
    if runtime_device.type != "cuda" and hasattr(model, "to"):
        model = model.to(runtime_device)
    if hasattr(model, "eval"):
        model.eval()
    return model, tokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="mrd.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    model_cfg = cfg.model
    model, tokenizer = _load_runtime_model(model_cfg)
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
        num_workers=cfg.mrd.num_workers,
        pin_memory=cfg.mrd.pin_memory,
        show_progress=cfg.mrd.show_progress,
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
