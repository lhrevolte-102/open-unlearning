import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from runtime_utils import seed_everything
from trackio_utils import (
    emit_trackio_alert,
    finish_trackio_run,
    init_trackio_run,
    is_trackio_enabled,
)
from trainer import load_trainer
from trainer.trackio_callback import TrackioLoggingCallback
from evals import get_evaluators


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    trackio_active = False
    if is_trackio_enabled(cfg):
        trainer.add_callback(TrackioLoggingCallback())
        if trainer.is_world_process_zero():
            trackio_active = init_trackio_run(cfg)

    try:
        if trainer_args.do_train:
            trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint", None))
            trainer.save_state()
            trainer.save_model(trainer_args.output_dir)

        if trainer_args.do_eval:
            trainer.evaluate(metric_key_prefix="eval")
    except Exception as exc:
        if trackio_active:
            emit_trackio_alert("Training failed", str(exc), level="ERROR")
        raise
    finally:
        if trackio_active:
            finish_trackio_run()


if __name__ == "__main__":
    main()
