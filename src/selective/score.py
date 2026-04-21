from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from utils.loss import compute_dpo_loss


def _strip_index(batch):
    return {key: value for key, value in batch.items() if key != "index"}


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = _move_batch_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _normalize_method(method):
    method = method.lower()
    if method in {"npo"}:
        return "npo"
    if method in {"dpo", "idkdpo", "idk_dpo"}:
        return "idkdpo"
    raise ValueError(f"Unsupported selective method '{method}'")


def _ensure_batch_matches_method(batch, method):
    has_pairwise_keys = isinstance(batch, dict) and {
        "original",
        "alternate",
    }.issubset(batch.keys())
    has_flat_index = isinstance(batch, dict) and "index" in batch

    if method == "npo" and not has_flat_index:
        if has_pairwise_keys:
            raise ValueError(
                "Received an IdkDPO-style paired batch while selective scoring is "
                "configured with method='npo'. Check that the experiment override "
                "is applied after the base selective config."
            )
        raise KeyError("index")

    if method == "idkdpo" and not has_pairwise_keys:
        raise ValueError(
            "Received a flat batch while selective scoring is configured with "
            "method='idkdpo'. Check that the experiment override matches the "
            "dataset used for selective scoring."
        )


def compute_unlearning_forget_losses(model, ref_model, batch, method, beta):
    method = _normalize_method(method)
    _ensure_batch_matches_method(batch, method)
    device = next(model.parameters()).device

    if method == "npo":
        indices = batch["index"].detach().cpu().tolist()
        lose_inputs = _move_batch_to_device(_strip_index(batch), device)
        with torch.no_grad():
            losses, _ = compute_dpo_loss(
                model=model,
                ref_model=ref_model,
                win_inputs=None,
                lose_inputs=lose_inputs,
                beta=beta,
                reduction="none",
            )
    else:
        original_batch = batch["original"]
        alternate_batch = batch["alternate"]
        indices = original_batch["index"].detach().cpu().tolist()
        win_inputs = _move_batch_to_device(_strip_index(alternate_batch), device)
        lose_inputs = _move_batch_to_device(_strip_index(original_batch), device)
        with torch.no_grad():
            losses, _ = compute_dpo_loss(
                model=model,
                ref_model=ref_model,
                win_inputs=win_inputs,
                lose_inputs=lose_inputs,
                beta=beta,
                reduction="none",
            )

    return indices, losses.detach().cpu().tolist()


def score_dataset_with_reference(
    model,
    ref_model,
    dataset,
    collator,
    method,
    beta,
    batch_size,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    scores_by_index = defaultdict(list)

    model.eval()
    ref_model.eval()
    for batch in dataloader:
        indices, losses = compute_unlearning_forget_losses(
            model=model,
            ref_model=ref_model,
            batch=batch,
            method=method,
            beta=beta,
        )
        for idx, loss in zip(indices, losses):
            scores_by_index[int(idx)].append(float(loss))

    return scores_by_index


def build_difficulty_payload(score_records, metadata):
    all_indices = sorted(int(idx) for idx in metadata["all_indices"])
    averaged_scores = {
        int(idx): {
            "score": float(sum(scores) / len(scores)),
            "num_refs": len(scores),
        }
        for idx, scores in score_records.items()
        if scores
    }

    ranked_indices = [
        idx
        for idx, _ in sorted(
            averaged_scores.items(),
            key=lambda item: (item[1]["score"], item[0]),
        )
    ]
    total_examples = len(all_indices)

    for rank, idx in enumerate(ranked_indices, start=1):
        averaged_scores[idx]["rank"] = rank
        averaged_scores[idx]["percentile"] = rank / total_examples

    return {
        "metadata": {
            **metadata,
            "all_indices": all_indices,
        },
        "scores_by_index": {
            str(idx): value for idx, value in sorted(averaged_scores.items())
        },
    }
