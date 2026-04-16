import hydra
from omegaconf import DictConfig

from trackio_utils import (
    emit_trackio_alert,
    finish_trackio_run,
    init_trackio_run,
    is_trackio_enabled,
    log_trackio_metrics,
)
from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators


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
    trackio_active = False
    if is_trackio_enabled(cfg):
        trackio_active = init_trackio_run(cfg)

    try:
        for evaluator_name, evaluator in evaluators.items():
            eval_args = {
                "template_args": template_args,
                "model": model,
                "tokenizer": tokenizer,
            }
            metrics = evaluator.evaluate(**eval_args)
            if trackio_active and isinstance(metrics, dict):
                log_trackio_metrics(metrics)
    except Exception as exc:
        if trackio_active:
            emit_trackio_alert("Evaluation failed", str(exc), level="ERROR")
        raise
    finally:
        if trackio_active:
            finish_trackio_run()


if __name__ == "__main__":
    main()
