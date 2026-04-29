import csv
import gc
import json
import logging
import os
import random
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Sampler
from transformers import TrainerCallback

from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import NPO
from trainer.unlearn.simnpo import SimNPO


logger = logging.getLogger(__name__)


def _safe_zscore(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    tensor = torch.tensor(list(values.values()), dtype=torch.float32)
    mean = tensor.mean().item()
    std = tensor.std(unbiased=False).item()
    if std < 1e-8:
        return {idx: 0.0 for idx in values}
    return {idx: (value - mean) / std for idx, value in values.items()}


class InfoCURLSampler(Sampler[int]):
    """Cursor-based sampler over a mutable shared order list."""

    def __init__(self, initial_order: Iterable[int]):
        self.order = list(initial_order)
        self.cursor = 0

    def __iter__(self):
        while self.cursor < len(self.order):
            yield self.order[self.cursor]
            self.cursor += 1
        self.cursor = 0

    def __len__(self):
        return len(self.order)


class InfoCURLCallback(TrainerCallback):
    def __init__(self, trainer: "InfoCURLMixin"):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        mode, _, _ = trainer._effective_schedule(state.global_step, state.max_steps)
        if mode not in {"easy", "hard", "active", "soft"}:
            return
        if state.global_step == 0:
            return
        if mode == "active" or state.global_step % trainer.K == 0:
            trainer._rescore_unread_tail(state.global_step, state.max_steps)


class InfoCURLMixin:
    """NPO-oriented InfoCURL sampler mixin.

    This minimal implementation focuses on the NPO + TOFU path needed for the
    current experiment budget. It supports:
    - `score_norm_static`: one-shot score ordering from the initial checkpoint
    - `easy` / `hard`: dynamic subpool rescoring every K steps
    - `active`: dynamic full-tail rescoring every step

    The scoring functional is based on the proposal's length-normalized
    empirical Fisher surrogate:

        D_empF(x) = T(x) * || grad_theta mean_logp(x) ||^2

    plus an optional reference-logp gap term for NPO:

        score = z(D_empF / D0_empF) + gamma * z(Delta)

    An optional retain-aware conflict term can be added via `lam > 0`. The
    retain term does not change the training objective; it only changes the
    sampler's acquisition score during rescoring.
    """

    def __init__(self, sampler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sampler_cfg = dict(sampler or {})
        self.mode = sampler_cfg.pop("mode", "uniform")
        self.K = int(sampler_cfg.pop("K", 5))
        self.score_gamma = float(sampler_cfg.pop("gamma", 0.0))
        self.score_subpool = int(sampler_cfg.pop("score_subpool", 64))
        self.use_ref_gap = bool(sampler_cfg.pop("use_ref_gap", True))
        self.cached_order_path = sampler_cfg.pop("cached_order_path", None)
        self.length_normalize = bool(sampler_cfg.pop("length_normalize", True))
        self.param_scope = sampler_cfg.pop("param_scope", "last_layer_lm_head")
        self.schedule = sampler_cfg.pop("schedule", None)
        self.stage1_mode = sampler_cfg.pop("stage1_mode", self.mode)
        self.stage2_mode = sampler_cfg.pop("stage2_mode", self.mode)
        self.stage1_gamma = float(sampler_cfg.pop("stage1_gamma", self.score_gamma))
        self.stage2_gamma = float(sampler_cfg.pop("stage2_gamma", self.score_gamma))
        self.switch_frac = float(sampler_cfg.pop("switch_frac", 0.35))
        self.transition_frac = float(sampler_cfg.pop("transition_frac", 0.30))
        self.score_lam = float(sampler_cfg.pop("lam", 0.0))
        self.retain_batch_size = int(sampler_cfg.pop("retain_batch_size", 8))
        self.retain_ema_decay = float(sampler_cfg.pop("retain_ema_decay", 0.9))
        if sampler_cfg:
            raise ValueError(f"Unknown sampler keys: {sorted(sampler_cfg.keys())}")

        self.forget_dataset = None
        self.retain_dataset = None
        self.base_scores: Dict[int, float] = {}
        self.base_mean_logp: Dict[int, float] = {}
        self.token_counts: Dict[int, int] = {}
        self.sampler = None
        self._score_parameter_cache = None
        self._infocurl_log_dir = None
        self._retain_anchor = None

        self.add_callback(InfoCURLCallback(self))

    def _schedule_alpha(self, step=0, total_steps=None):
        if self.schedule != "soft_easy_to_hard":
            return None

        if total_steps is None or total_steps <= 0:
            total_steps = getattr(self.args, "max_steps", 0)
        if total_steps is None or total_steps <= 0:
            total_steps = getattr(getattr(self, "state", None), "max_steps", 0)
        if total_steps is None or total_steps <= 0:
            return 0.0

        progress = step / float(total_steps)
        start = self.switch_frac
        end = min(1.0, start + self.transition_frac)
        if progress <= start:
            return 0.0
        if progress >= end:
            return 1.0
        return (progress - start) / max(1e-8, end - start)

    def _effective_schedule(self, step=0, total_steps=None):
        alpha = self._schedule_alpha(step, total_steps)
        if alpha is not None:
            blended_gamma = (1.0 - alpha) * self.stage1_gamma + alpha * self.stage2_gamma
            return "soft", blended_gamma, alpha

        if self.schedule != "easy_to_hard":
            return self.mode, self.score_gamma, None

        if total_steps is None or total_steps <= 0:
            total_steps = getattr(self.args, "max_steps", 0)
        if total_steps is None or total_steps <= 0:
            total_steps = getattr(getattr(self, "state", None), "max_steps", 0)

        if total_steps and (step / float(total_steps)) >= self.switch_frac:
            return self.stage2_mode, self.stage2_gamma, 1.0
        return self.stage1_mode, self.stage1_gamma, 0.0

    def _merge_orders(self, easy_order, hard_order, alpha, seed):
        if alpha <= 0.0:
            return list(easy_order)
        if alpha >= 1.0:
            return list(hard_order)

        rng = random.Random(seed)
        merged = []
        used = set()
        easy_ptr = 0
        hard_ptr = 0

        def next_unused(order, ptr):
            while ptr < len(order) and order[ptr] in used:
                ptr += 1
            if ptr >= len(order):
                return None, ptr
            item = order[ptr]
            ptr += 1
            return item, ptr

        while len(merged) < len(easy_order):
            choose_hard = rng.random() < alpha
            if choose_hard:
                item, hard_ptr = next_unused(hard_order, hard_ptr)
                if item is None:
                    item, easy_ptr = next_unused(easy_order, easy_ptr)
            else:
                item, easy_ptr = next_unused(easy_order, easy_ptr)
                if item is None:
                    item, hard_ptr = next_unused(hard_order, hard_ptr)
            if item is None:
                break
            merged.append(item)
            used.add(item)

        return merged

    @property
    def _score_parameters(self):
        if self._score_parameter_cache is None:
            named_params = [
                (name, param)
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ]
            if self.param_scope == "all":
                selected = [param for _, param in named_params]
            elif self.param_scope == "lm_head_only":
                selected = [
                    param
                    for name, param in named_params
                    if name.startswith("lm_head") or ".lm_head." in name
                ]
            else:
                layer_indices = []
                for name, _ in named_params:
                    if ".layers." not in name:
                        continue
                    suffix = name.split(".layers.", 1)[1]
                    layer_idx = suffix.split(".", 1)[0]
                    if layer_idx.isdigit():
                        layer_indices.append(int(layer_idx))
                last_layer_idx = max(layer_indices) if layer_indices else None
                selected = []
                for name, param in named_params:
                    if name.startswith("lm_head") or ".lm_head." in name:
                        selected.append(param)
                        continue
                    if (
                        last_layer_idx is not None
                        and f".layers.{last_layer_idx}." in name
                    ):
                        selected.append(param)
                if not selected:
                    selected = [param for _, param in named_params]
            logger.info(
                "InfoCURL scoring uses %d parameter tensors (scope=%s).",
                len(selected),
                self.param_scope,
            )
            self._score_parameter_cache = selected
        return self._score_parameter_cache

    def _ensure_infocurl_dir(self):
        if self._infocurl_log_dir is None:
            self._infocurl_log_dir = os.path.join(self.args.output_dir, "infocurl")
            os.makedirs(self._infocurl_log_dir, exist_ok=True)
        return self._infocurl_log_dir

    def _write_jsonl(self, filename: str, payload: dict):
        log_dir = self._ensure_infocurl_dir()
        path = os.path.join(log_dir, filename)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _write_phase0_scores(self, scores: Dict[int, float], filename: str):
        log_dir = self._ensure_infocurl_dir()
        path = os.path.join(log_dir, filename)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "score"])
            for idx, score in sorted(scores.items(), key=lambda item: item[1]):
                writer.writerow([idx, f"{score:.8f}"])

    def _token_count(self, labels: torch.Tensor) -> int:
        shifted = labels[..., 1:]
        return max(1, int((shifted != -100).sum().item()))

    def _collate_forget_index(self, dataset_index: int):
        item = self.forget_dataset[dataset_index]
        batch = self.data_collator([item])
        return self._prepare_inputs(batch)

    def _mean_logp(self, model, inputs) -> float:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
        return -float(outputs.loss.detach().item())

    def _score_single(self, dataset_index: int):
        batch = self._collate_forget_index(dataset_index)
        token_count = self._token_count(batch["labels"])

        with self.compute_loss_context_manager():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        mean_logp = -outputs.loss
        grads = torch.autograd.grad(
            mean_logp,
            self._score_parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        grad_norm_sq = 0.0
        for grad in grads:
            if grad is None:
                continue
            grad_norm_sq += float(grad.detach().float().pow(2).sum().item())
        mean_logp_value = float(mean_logp.detach().item())
        del grads
        del outputs
        del batch
        del mean_logp
        gc.collect()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
        score = token_count * grad_norm_sq if self.length_normalize else grad_norm_sq
        return score, mean_logp_value, token_count

    def _loss_grads(self, loss):
        grads = torch.autograd.grad(
            loss,
            self._score_parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        return [grad.detach().float() if grad is not None else None for grad in grads]

    def _normalize_grads(self, grads):
        norm_sq = 0.0
        for grad in grads:
            if grad is None:
                continue
            norm_sq += float(grad.pow(2).sum().item())
        norm = norm_sq ** 0.5
        if norm < 1e-12:
            return None
        return [grad / norm if grad is not None else None for grad in grads]

    def _dot_grads(self, grads_a, grads_b):
        total = 0.0
        for grad_a, grad_b in zip(grads_a, grads_b):
            if grad_a is None or grad_b is None:
                continue
            total += float((grad_a * grad_b).sum().item())
        return total

    def _sample_retain_batch(self):
        if self.retain_dataset is None or len(self.retain_dataset) == 0:
            return None
        batch_size = min(len(self.retain_dataset), self.retain_batch_size)
        indices = random.sample(range(len(self.retain_dataset)), batch_size)
        items = [self.retain_dataset[idx] for idx in indices]
        batch = self.data_collator(items)
        batch = self._prepare_inputs(batch)
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def _update_retain_anchor(self):
        retain_inputs = self._sample_retain_batch()
        if retain_inputs is None:
            return None

        with self.compute_loss_context_manager():
            retain_loss = self.compute_retain_loss(self.model, retain_inputs)
        grads = self._loss_grads(retain_loss)
        grads = self._normalize_grads(grads)
        del retain_loss
        del retain_inputs
        gc.collect()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
        if grads is None:
            return self._retain_anchor

        if self._retain_anchor is None:
            self._retain_anchor = grads
            return self._retain_anchor

        blended = []
        for old, new in zip(self._retain_anchor, grads):
            if old is None and new is None:
                blended.append(None)
            elif old is None:
                blended.append(new)
            elif new is None:
                blended.append(old)
            else:
                blended.append(self.retain_ema_decay * old + (1.0 - self.retain_ema_decay) * new)
        self._retain_anchor = self._normalize_grads(blended)
        return self._retain_anchor

    def _cache_phase0(self):
        if self.base_scores:
            return

        if self.use_ref_gap and getattr(self, "ref_model", None) is None:
            self.ref_model = self._prepare_ref_model(self.model)

        was_training = self.model.training
        self.model.eval()
        if getattr(self, "ref_model", None) is not None:
            self.ref_model.eval()

        logger.info(
            "InfoCURL phase-0 scoring started on %d forget examples.",
            len(self.forget_dataset),
        )
        for dataset_index in range(len(self.forget_dataset)):
            score, mean_logp, token_count = self._score_single(dataset_index)
            self.base_scores[dataset_index] = max(score, 1e-8)
            self.base_mean_logp[dataset_index] = mean_logp
            self.token_counts[dataset_index] = token_count

            if dataset_index % 25 == 0 or dataset_index == len(self.forget_dataset) - 1:
                logger.info(
                    "InfoCURL phase-0 progress: %d/%d",
                    dataset_index + 1,
                    len(self.forget_dataset),
                )

        if was_training:
            self.model.train()

        gc.collect()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()

        self._write_phase0_scores(self.base_scores, "phase0_scores.csv")

    def _load_cached_order(self) -> List[int]:
        if not self.cached_order_path:
            raise ValueError("cached_order_path is required for sampler.mode=cached")
        with open(self.cached_order_path, "r", encoding="utf-8") as handle:
            rows = [line.strip() for line in handle if line.strip()]
        if not rows:
            raise ValueError(f"Cached order file is empty: {self.cached_order_path}")
        if "," not in rows[0]:
            return [int(row) for row in rows]
        reader = csv.DictReader(rows)
        return [int(row["index"]) for row in reader]

    def _initial_order(self) -> List[int]:
        if self.mode == "cached":
            return self._load_cached_order()

        self._cache_phase0()

        mode, _, alpha = self._effective_schedule(0, getattr(self.args, "max_steps", 0))

        if mode == "uniform":
            order = list(range(len(self.forget_dataset)))
            random.shuffle(order)
            return order

        easy_order = sorted(
            range(len(self.forget_dataset)),
            key=lambda idx: self.base_scores[idx],
            reverse=False,
        )
        hard_order = list(reversed(easy_order))
        if mode == "soft":
            ordered = self._merge_orders(
                easy_order,
                hard_order,
                alpha or 0.0,
                seed=int(self.args.seed),
            )
            ascending = None
        else:
            ascending = mode in {"score_norm_static", "easy"}
            ordered = easy_order if ascending else hard_order
        self._write_jsonl(
            "events.jsonl",
            {
                "event": "initial_order",
                "mode": mode,
                "ascending": ascending,
                "alpha": alpha,
                "top10": ordered[:10],
            },
        )
        return ordered

    def _rescore_unread_tail(self, step=None, total_steps=None):
        tail = self.sampler.order[self.sampler.cursor :]
        if not tail:
            return

        was_training = self.model.training
        self.model.eval()

        mode, score_gamma, alpha = self._effective_schedule(step or 0, total_steps)
        retain_anchor = self._update_retain_anchor() if self.score_lam > 0 else None

        if mode == "active":
            candidate_indices = list(tail)
        else:
            sample_size = min(len(tail), self.score_subpool)
            candidate_indices = random.sample(tail, sample_size)

        relative_scores = {}
        ref_gaps = {}
        conflicts = {}
        raw_scores = {}
        for dataset_index in candidate_indices:
            batch = self._collate_forget_index(dataset_index)
            token_count = self._token_count(batch["labels"])
            with self.compute_loss_context_manager():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            mean_logp = -outputs.loss
            grads = self._loss_grads(mean_logp)
            grad_norm_sq = 0.0
            for grad in grads:
                if grad is None:
                    continue
                grad_norm_sq += float(grad.pow(2).sum().item())
            score = token_count * grad_norm_sq if self.length_normalize else grad_norm_sq
            raw_scores[dataset_index] = score
            relative_scores[dataset_index] = score / self.base_scores[dataset_index]
            if self.use_ref_gap:
                ref_gaps[dataset_index] = float(mean_logp.detach().item()) - self.base_mean_logp[dataset_index]
            if retain_anchor is not None:
                normalized_forget = self._normalize_grads(grads)
                if normalized_forget is None:
                    conflicts[dataset_index] = 0.0
                else:
                    cosine = self._dot_grads(normalized_forget, retain_anchor)
                    conflicts[dataset_index] = max(0.0, -cosine)
            del outputs
            del mean_logp
            del grads
            del batch

        z_rel = _safe_zscore(relative_scores)
        z_gap = _safe_zscore(ref_gaps) if ref_gaps else {}
        z_conflict = _safe_zscore(conflicts) if conflicts else {}
        acquisition = {}
        for dataset_index in candidate_indices:
            acquisition[dataset_index] = (
                z_rel[dataset_index]
                + score_gamma * z_gap.get(dataset_index, 0.0)
                - self.score_lam * z_conflict.get(dataset_index, 0.0)
            )

        easy_order = sorted(candidate_indices, key=lambda idx: acquisition[idx], reverse=False)
        hard_order = list(reversed(easy_order))
        if mode == "soft":
            reranked = self._merge_orders(
                easy_order,
                hard_order,
                alpha or 0.0,
                seed=int((step if step is not None else self.state.global_step) + self.args.seed),
            )
        else:
            reverse = mode in {"hard", "active"}
            reranked = hard_order if reverse else easy_order
        if mode == "active":
            reranked = reranked[:1]

        remaining = [idx for idx in tail if idx not in candidate_indices]
        self.sampler.order[self.sampler.cursor :] = reranked + remaining

        self._write_jsonl(
            "events.jsonl",
            {
                "event": "rescore",
                "global_step": int(step if step is not None else self.state.global_step),
                "mode": mode,
                "score_gamma": score_gamma,
                "score_lam": self.score_lam,
                "alpha": alpha,
                "cursor": int(self.sampler.cursor),
                "tail_size": len(tail),
                "candidate_size": len(candidate_indices),
                "top5": [
                    {
                        "index": idx,
                        "raw_score": raw_scores[idx],
                        "relative_score": relative_scores[idx],
                        "ref_gap": ref_gaps.get(idx, 0.0),
                        "conflict": conflicts.get(idx, 0.0),
                        "acquisition": acquisition[idx],
                    }
                    for idx in reranked[:5]
                ],
            },
        )

        if was_training:
            self.model.train()
        gc.collect()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset

        self.forget_dataset = train_dataset.forget
        self.retain_dataset = train_dataset.retain

        dynamic_modes = {self.mode}
        if self.schedule == "easy_to_hard":
            dynamic_modes.update({self.stage1_mode, self.stage2_mode})

        if self.schedule == "soft_easy_to_hard":
            dynamic_modes.update({"soft", self.stage1_mode, self.stage2_mode})

        if dynamic_modes & {"easy", "hard", "active", "soft"}:
            assert self.accelerator.num_processes == 1, (
                "Dynamic InfoCURL modes require single-process training. "
                "Use one device per run and launch variants in parallel instead."
            )
            assert self.args.dataloader_num_workers == 0, (
                "Dynamic InfoCURL modes require dataloader_num_workers=0."
            )
            effective_batch = (
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
            )
            prefetch_lead = self.args.per_device_train_batch_size
            min_forget_size = 4 * (effective_batch + prefetch_lead)
            assert len(train_dataset) >= min_forget_size, (
                "Forget pool too small for dynamic InfoCURL under the current "
                f"batch geometry: need >= {min_forget_size}, got {len(train_dataset)}."
            )

        self.sampler = InfoCURLSampler(self._initial_order())
        return self.sampler


class InfoCURL_NPO(InfoCURLMixin, NPO):
    pass


class InfoCURL_SimNPO(InfoCURLMixin, SimNPO):
    pass


class InfoCURL_GradDiff(InfoCURLMixin, GradDiff):
    pass
