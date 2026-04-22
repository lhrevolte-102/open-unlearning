import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=False)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_checkpoint_step(model_path: str | Path) -> int | None:
    path = Path(model_path)
    checkpoint_match = re.search(r"checkpoint-(\d+)$", path.name)
    if checkpoint_match:
        return int(checkpoint_match.group(1))

    trainer_state_path = path / "trainer_state.json"
    if not trainer_state_path.is_file():
        return None

    payload = load_json(trainer_state_path)
    step = payload.get("global_step")
    return int(step) if step is not None else None


def clamp_abs(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(tensor.abs(), min=float(eps))


def _strip_index(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in batch.items() if key != "index"}


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def get_answer_log_probs(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_mask = shift_labels != -100

    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = token_log_probs.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~valid_mask, 0.0)
    return token_log_probs, valid_mask


def compute_mrd_loss(
    base_log_probs: torch.Tensor,
    perturbed_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    relative_delta = (base_log_probs - perturbed_log_probs) / clamp_abs(
        base_log_probs, eps
    )
    relative_delta = relative_delta.masked_fill(~valid_mask, 0.0)
    return relative_delta.sum(dim=-1).abs()


def perturb_model_parameters(
    model: torch.nn.Module, sigma: float, generator: torch.Generator
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    perturbations = []
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            noise = torch.randn(
                param.shape, generator=generator, device="cpu", dtype=torch.float32
            ).to(device=param.device, dtype=param.dtype)
            noise.mul_(float(sigma))
            param.add_(noise)
            perturbations.append((param, noise))
    return perturbations


def revert_model_perturbations(
    perturbations: list[tuple[torch.nn.Parameter, torch.Tensor]]
) -> None:
    with torch.no_grad():
        for param, noise in perturbations:
            param.sub_(noise)


def score_mrd_for_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sigma: float,
    num_mc_samples: int,
    eps: float,
    generator: torch.Generator,
) -> tuple[list[int], list[float]]:
    if "index" not in batch:
        raise KeyError(
            "MRD scoring expects the collated batch to contain an 'index' tensor"
        )

    device = next(model.parameters()).device
    batch = move_batch_to_device(batch, device)
    indices = [int(idx) for idx in batch["index"].detach().cpu().tolist()]
    model_inputs = _strip_index(batch)

    with torch.no_grad():
        base_outputs = model(**model_inputs)
    base_log_probs, valid_mask = get_answer_log_probs(
        base_outputs.logits, model_inputs["labels"]
    )

    accumulated_scores = torch.zeros(
        base_log_probs.shape[0],
        device=base_log_probs.device,
        dtype=base_log_probs.dtype,
    )
    for _ in range(int(num_mc_samples)):
        perturbations = perturb_model_parameters(
            model, sigma=sigma, generator=generator
        )
        try:
            with torch.no_grad():
                perturbed_outputs = model(**model_inputs)
            perturbed_log_probs, _ = get_answer_log_probs(
                perturbed_outputs.logits, model_inputs["labels"]
            )
        finally:
            revert_model_perturbations(perturbations)

        accumulated_scores += compute_mrd_loss(
            base_log_probs=base_log_probs,
            perturbed_log_probs=perturbed_log_probs,
            valid_mask=valid_mask,
            eps=eps,
        )

    batch_scores = accumulated_scores / float(num_mc_samples)
    return indices, [float(score) for score in batch_scores.detach().cpu().tolist()]


def score_dataset_with_mrd(
    model: torch.nn.Module,
    dataset,
    collator,
    batch_size: int,
    sigma: float,
    num_mc_samples: int,
    eps: float,
    seed: int = 0,
) -> dict[int, float]:
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    scores_by_index = {}

    model.eval()
    for batch in dataloader:
        indices, scores = score_mrd_for_batch(
            model=model,
            batch=batch,
            sigma=sigma,
            num_mc_samples=num_mc_samples,
            eps=eps,
            generator=generator,
        )
        for idx, score in zip(indices, scores):
            scores_by_index[int(idx)] = float(score)

    return scores_by_index


def build_mrd_payload(
    scores_by_index: dict[int, float],
    metadata: dict[str, Any],
    weight_floor: float = 1e-12,
) -> dict[str, Any]:
    normalized_scores = {
        int(idx): float(score) for idx, score in sorted(scores_by_index.items())
    }
    weights_by_index = {
        idx: max(float(score), float(weight_floor))
        for idx, score in normalized_scores.items()
    }

    ranked_indices = [
        idx
        for idx, _ in sorted(
            normalized_scores.items(), key=lambda item: (-float(item[1]), int(item[0]))
        )
    ]

    payload_scores = {}
    payload_weights = {}
    total = len(normalized_scores)
    for rank, idx in enumerate(ranked_indices, start=1):
        payload_scores[str(idx)] = {
            "score": normalized_scores[idx],
            "rank": rank,
            "percentile": (rank / total) if total else 0.0,
        }
    for idx, weight in weights_by_index.items():
        payload_weights[str(idx)] = weight

    return {
        "metadata": metadata,
        "scores_by_index": payload_scores,
        "weights_by_index": payload_weights,
    }


def load_weights_by_index(weights_path: str | Path) -> dict[int, float]:
    payload = load_json(weights_path)
    weights = payload.get("weights_by_index")
    if weights is None:
        raise ValueError(
            f"No 'weights_by_index' mapping found in MRD weights manifest: {weights_path}"
        )

    normalized = {int(idx): float(value) for idx, value in weights.items()}
    if not normalized:
        raise ValueError(f"MRD weights manifest is empty: {weights_path}")

    return normalized
