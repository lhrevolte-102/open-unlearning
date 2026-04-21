import hydra
from omegaconf import DictConfig

from utils.runtime import seed_everything
from selective.pipeline import run_selective_pipeline


@hydra.main(version_base=None, config_path="../configs", config_name="selective.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    run_selective_pipeline(cfg)


if __name__ == "__main__":
    main()
