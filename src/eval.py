import hydra
from omegaconf import DictConfig

from utils.runtime import seed_everything
from model import get_model
from evals import get_evaluators
from utils.tensorboard import create_tensorboard_writer, log_tensorboard_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    writer = create_tensorboard_writer(cfg.paths.output_dir)

    try:
        for evaluator_name, evaluator in evaluators.items():
            eval_args = {
                "template_args": template_args,
                "model": model,
                "tokenizer": tokenizer,
            }
            metrics = evaluator.evaluate(**eval_args)
            if isinstance(metrics, dict):
                log_tensorboard_metrics(writer, metrics, prefix=evaluator_name)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
