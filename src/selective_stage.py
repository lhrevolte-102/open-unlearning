import hydra
from omegaconf import DictConfig

from selective.utils import build_stage_manifests, load_json, save_json


@hydra.main(
    version_base=None, config_path="../configs", config_name="selective_stage.yaml"
)
def main(cfg: DictConfig):
    difficulty_payload = load_json(cfg.difficulty_path)
    manifests = build_stage_manifests(
        difficulty_payload=difficulty_payload,
        stage_percentiles=list(cfg.stage_percentiles),
        stage_epoch_ratios=list(cfg.stage_epoch_ratios),
    )

    summary = {
        "difficulty_path": cfg.difficulty_path,
        "stage_percentiles": list(cfg.stage_percentiles),
        "stage_epoch_ratios": list(cfg.stage_epoch_ratios),
        "stages": [],
    }

    for manifest in manifests:
        output_path = f"{cfg.output_dir}/{manifest['stage_name']}.json"
        save_json(output_path, manifest)
        summary["stages"].append(
            {
                "stage_name": manifest["stage_name"],
                "output_path": output_path,
                "num_examples": manifest["num_examples"],
                "percentile": manifest["percentile"],
                "epoch_ratio": manifest["epoch_ratio"],
            }
        )

    save_json(f"{cfg.output_dir}/stages.json", summary)


if __name__ == "__main__":
    main()
