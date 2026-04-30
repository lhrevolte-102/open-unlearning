import gc
import json
import logging
import os
from typing import List

import torch
from torch.utils.data import Sampler
from transformers import TrainerCallback

from trainer.unlearn.npo import NPO
from trainer.utils import compute_batch_nll


logger = logging.getLogger(__name__)


class ADOSampler(Sampler[int]):
    """Weighted sample-level sampler with mutable probabilities."""

    def __init__(self, dataset_size: int, seed: int = 42):
        if dataset_size <= 0:
            raise ValueError("ADOSampler requires a non-empty dataset.")
        self.dataset_size = int(dataset_size)
        self.weights = torch.full((self.dataset_size,), 1.0 / self.dataset_size)
        self.seed = int(seed)
        self.epoch = 0
        self.yielded: List[int] = []

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            num_samples=self.dataset_size,
            replacement=True,
            generator=generator,
        ).tolist()
        self.epoch += 1
        for idx in sampled:
            self.yielded.append(int(idx))
            yield int(idx)

    def __len__(self):
        return self.dataset_size

    def take_batch_indices(self, batch_size: int) -> List[int]:
        if batch_size <= 0:
            return []
        if len(self.yielded) < batch_size:
            raise RuntimeError(
                "ADOSampler could not match sampled indices to the current batch. "
                "Use dataloader_num_workers=0 and single-process training."
            )
        indices = self.yielded[:batch_size]
        del self.yielded[:batch_size]
        return indices

    def set_weights(self, weights: torch.Tensor):
        if weights.numel() != self.dataset_size:
            raise ValueError(
                f"Expected {self.dataset_size} sampler weights, got {weights.numel()}."
            )
        weights = weights.detach().float().cpu()
        total = weights.sum().item()
        if not torch.isfinite(weights).all() or total <= 0:
            raise ValueError(
                "ADO sampler weights must be finite and sum to a positive value."
            )
        self.weights = weights / total


class ADOCallback(TrainerCallback):
    def __init__(self, trainer: "ADOMixin"):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer._ado_complete_optimizer_step(state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer._ado_maybe_refresh_weights(state.epoch, state.global_step)


class ADOMixin:
    """Sample-level ADO scheduler for unlearning methods.

    The mixin keeps the unlearning objective unchanged. It only changes which
    forget examples are sampled by updating per-example probabilities from
    measured forgetting gains.
    """

    def __init__(self, sampler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sampler_cfg = dict(sampler or {})
        self.ado_eta = float(sampler_cfg.pop("eta", 1.0))
        self.ado_beta = float(sampler_cfg.pop("beta", 0.9))
        self.ado_refresh_epochs = int(sampler_cfg.pop("refresh_epochs", 1))
        self.ado_gain_floor = float(sampler_cfg.pop("gain_floor", 0.0))
        self.ado_gain_clip = sampler_cfg.pop("gain_clip", None)
        self.ado_prob_floor = float(sampler_cfg.pop("prob_floor", 0.0))
        self.ado_uniform_mix = float(sampler_cfg.pop("uniform_mix", 0.0))
        self.ado_log_every = int(sampler_cfg.pop("log_every", 10))
        self.ado_signal = sampler_cfg.pop("signal", "logp")
        if sampler_cfg:
            raise ValueError(f"Unknown ADO sampler keys: {sorted(sampler_cfg.keys())}")
        if self.ado_signal != "logp":
            raise ValueError("ADO_NPO currently supports sampler.signal=logp only.")
        if not 0.0 <= self.ado_beta < 1.0:
            raise ValueError("ADO sampler beta must satisfy 0 <= beta < 1.")
        if self.ado_refresh_epochs <= 0:
            raise ValueError("ADO sampler refresh_epochs must be positive.")
        if self.ado_uniform_mix < 0.0 or self.ado_uniform_mix >= 1.0:
            raise ValueError(
                "ADO sampler uniform_mix must satisfy 0 <= uniform_mix < 1."
            )

        self.forget_dataset = None
        self.retain_dataset = None
        self.sampler = None
        self.ado_g_ema = None
        self.ado_seen_counts = None
        self._ado_pending_batches = []
        self._ado_log_dir = None
        self._ado_last_refresh_epoch = 0

        self.add_callback(ADOCallback(self))

    def _ensure_ado_dir(self):
        if self._ado_log_dir is None:
            self._ado_log_dir = os.path.join(self.args.output_dir, "ado")
            os.makedirs(self._ado_log_dir, exist_ok=True)
        return self._ado_log_dir

    def _ado_write_jsonl(self, filename: str, payload: dict):
        log_dir = self._ensure_ado_dir()
        path = os.path.join(log_dir, filename)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _ado_collate_forget_indices(self, indices: List[int]):
        items = [self.forget_dataset[idx] for idx in indices]
        batch = self.data_collator(items)
        batch = self._prepare_inputs(batch)
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def _ado_logp_signal_from_inputs(self, model, inputs):
        nll, _ = compute_batch_nll(model, inputs)
        return -nll.detach().float()

    def _ado_capture_before_signal(self, model, forget_inputs):
        if self.sampler is None:
            return
        batch_size = int(forget_inputs["input_ids"].shape[0])
        indices = self.sampler.take_batch_indices(batch_size)
        with torch.no_grad():
            signal_before = self._ado_logp_signal_from_inputs(
                model, forget_inputs
            ).cpu()
        self._ado_pending_batches.append(
            {
                "indices": indices,
                "signal_before": signal_before,
            }
        )

    def _ado_complete_optimizer_step(self, global_step: int):
        if not self._ado_pending_batches:
            return

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.ado_g_ema.mul_(self.ado_beta)
            for pending in self._ado_pending_batches:
                indices = pending["indices"]
                signal_before = pending["signal_before"]
                forget_inputs = self._ado_collate_forget_indices(indices)
                signal_after = self._ado_logp_signal_from_inputs(
                    self.model, forget_inputs
                ).cpu()
                gains = signal_before - signal_after
                if self.ado_gain_clip is not None:
                    gains = gains.clamp(max=float(self.ado_gain_clip))
                gains = gains.clamp(min=self.ado_gain_floor)
                for idx, gain in zip(indices, gains.tolist()):
                    self.ado_g_ema[idx] += (1.0 - self.ado_beta) * float(gain)
                    self.ado_seen_counts[idx] += 1

        if was_training:
            self.model.train()
        self._ado_pending_batches.clear()

        if self.ado_log_every > 0 and global_step % self.ado_log_every == 0:
            self._ado_write_jsonl(
                "events.jsonl",
                {
                    "event": "step_gain",
                    "global_step": int(global_step),
                    "ema_mean": float(self.ado_g_ema.mean().item()),
                    "ema_max": float(self.ado_g_ema.max().item()),
                    "seen_examples": int((self.ado_seen_counts > 0).sum().item()),
                },
            )

        gc.collect()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()

    def _ado_maybe_refresh_weights(self, epoch, global_step: int):
        if self.sampler is None or self.ado_g_ema is None:
            return
        if epoch is None:
            return
        completed_epoch = int(epoch)
        if completed_epoch <= 0:
            return
        if completed_epoch == self._ado_last_refresh_epoch:
            return
        if completed_epoch % self.ado_refresh_epochs != 0:
            return

        old_weights = self.sampler.weights
        logits = (
            torch.log(old_weights.clamp_min(1e-12))
            + self.ado_eta * self.ado_g_ema
        )
        logits = logits - logits.max()
        new_weights = torch.exp(logits)
        if self.ado_prob_floor > 0.0:
            new_weights = new_weights.clamp_min(self.ado_prob_floor)
        new_weights = new_weights / new_weights.sum()
        if self.ado_uniform_mix > 0.0:
            uniform = torch.full_like(new_weights, 1.0 / new_weights.numel())
            new_weights = (
                (1.0 - self.ado_uniform_mix) * new_weights
                + self.ado_uniform_mix * uniform
            )
            new_weights = new_weights / new_weights.sum()

        self.sampler.set_weights(new_weights)
        self._ado_last_refresh_epoch = completed_epoch
        topk = torch.topk(
            self.sampler.weights, k=min(10, self.sampler.weights.numel())
        )
        self._ado_write_jsonl(
            "events.jsonl",
            {
                "event": "refresh_weights",
                "epoch": completed_epoch,
                "global_step": int(global_step),
                "eta": self.ado_eta,
                "ema_mean": float(self.ado_g_ema.mean().item()),
                "ema_max": float(self.ado_g_ema.max().item()),
                "weight_min": float(self.sampler.weights.min().item()),
                "weight_max": float(self.sampler.weights.max().item()),
                "top10": [
                    {"index": int(idx), "weight": float(weight)}
                    for weight, idx in zip(topk.values.tolist(), topk.indices.tolist())
                ],
            },
        )

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset

        self.forget_dataset = train_dataset.forget
        self.retain_dataset = train_dataset.retain

        assert self.accelerator.num_processes == 1, (
            "ADO dynamic sampling requires single-process training so per-step "
            "forget indices can be matched to model updates."
        )
        assert self.args.dataloader_num_workers == 0, (
            "ADO dynamic sampling requires dataloader_num_workers=0."
        )

        self.sampler = ADOSampler(
            dataset_size=len(self.forget_dataset),
            seed=int(self.args.seed),
        )
        self.ado_g_ema = torch.zeros(len(self.forget_dataset), dtype=torch.float32)
        self.ado_seen_counts = torch.zeros(len(self.forget_dataset), dtype=torch.long)
        logger.info(
            "ADO sampler initialized over %d forget examples.",
            len(self.forget_dataset),
        )
        return self.sampler


class ADO_NPO(ADOMixin, NPO):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        self._ado_capture_before_signal(model, forget_inputs)
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
