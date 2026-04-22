# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from mrd_utils import load_weights_by_index
from torch.utils.data import Dataset, SequentialSampler, WeightedRandomSampler
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length

logger = logging.getLogger(__name__)

# When using custom evaluators without an eval dataset, pass a dummy value
# to prevent Trainer from raising on eval_dataset=None when eval_strategy is set
_EVAL_PLACEHOLDER = "_EVAL_PLACEHOLDER"


class FinetuneTrainer(Trainer):
    def __init__(
        self,
        evaluators=None,
        template_args=None,
        train_sampler="random",
        train_sampler_args=None,
        *args,
        **kwargs,
    ):
        self.evaluators = evaluators
        self.template_args = template_args
        self.train_sampler = train_sampler
        self.train_sampler_args = train_sampler_args or {}
        if kwargs.get("eval_dataset") is None and evaluators:
            kwargs["eval_dataset"] = _EVAL_PLACEHOLDER
        super().__init__(*args, **kwargs)

    def _get_weighted_train_sampler(self):
        if self.args.group_by_length:
            raise ValueError(
                "train_sampler=weighted is incompatible with group_by_length=true"
            )
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError(
                "train_sampler=weighted currently supports only single-process training"
            )
        if torch.cuda.device_count() > 1:
            raise ValueError(
                "train_sampler=weighted currently supports only a single visible GPU"
            )

        weights_path = self.train_sampler_args.get("weights_path")
        if not weights_path:
            raise ValueError(
                "train_sampler=weighted requires "
                "trainer.train_sampler_args.weights_path"
            )
        if getattr(self.train_dataset, "anchor", None) != "forget":
            raise ValueError("train_sampler=weighted requires data.anchor=forget")
        if not hasattr(self.train_dataset, "get_anchor_indices"):
            raise ValueError(
                "train_sampler=weighted requires the train dataset to implement "
                "get_anchor_indices()"
            )

        anchor_indices = self.train_dataset.get_anchor_indices()
        weights_by_index = load_weights_by_index(weights_path)
        missing_indices = [idx for idx in anchor_indices if idx not in weights_by_index]
        if missing_indices:
            raise ValueError(
                "MRD weights manifest is missing weights for anchor dataset indices: "
                f"{missing_indices[:10]}"
            )

        weights = torch.tensor(
            [float(weights_by_index[idx]) for idx in anchor_indices], dtype=torch.double
        )
        if (weights < 0).any():
            raise ValueError("train_sampler=weighted received negative sample weights")
        if not (weights > 0).any():
            raise ValueError(
                "train_sampler=weighted requires at least one positive sample weight"
            )

        replacement = bool(self.train_sampler_args.get("replacement", True))
        num_samples = self.train_sampler_args.get("num_samples")
        num_samples = len(anchor_indices) if num_samples is None else int(num_samples)
        if num_samples <= 0:
            raise ValueError(
                "train_sampler=weighted requires num_samples to be positive"
            )

        return WeightedRandomSampler(
            weights=weights, num_samples=num_samples, replacement=replacement
        )

    def _get_train_sampler(self):
        if self.train_sampler == "random":
            return super()._get_train_sampler()
        if self.train_sampler == "sequential":
            if self.args.group_by_length:
                raise ValueError(
                    "train_sampler=sequential is incompatible with group_by_length=true"
                )
            if self.train_dataset is None or not has_length(self.train_dataset):
                return None
            return SequentialSampler(self.train_dataset)
        if self.train_sampler == "weighted":
            return self._get_weighted_train_sampler()

        raise ValueError(
            "Unsupported train_sampler "
            f"'{self.train_sampler}', expected 'random', 'sequential', or 'weighted'"
        )

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators and self.accelerator.is_local_main_process:
            if self.accelerator.num_processes != 1:
                logger.warning(
                    "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                )
                return {}

            run_dir = self._get_output_dir(trial=trial)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
            os.makedirs(output_dir, exist_ok=True)
            eval_metrics = {}
            for _, evaluator in self.evaluators.items():
                eval_args = {
                    "output_dir": output_dir,
                    "template_args": self.template_args,
                    "model": self.model,
                    "tokenizer": self.processing_class,
                }
                eval_metrics.update(evaluator.evaluate(**eval_args))
            self.log(eval_metrics)
            return eval_metrics

        if eval_dataset is None or eval_dataset == _EVAL_PLACEHOLDER:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
