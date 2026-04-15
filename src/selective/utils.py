import copy
import json
import math
from pathlib import Path

from data.utils import filter_dataset_by_index


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=False)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_indices(indices):
    return sorted({int(idx) for idx in indices})


def get_allowed_indices_from_manifest(payload):
    if isinstance(payload, dict):
        indices = payload.get("allowed_indices")
    else:
        indices = payload

    if indices is None:
        raise ValueError("No 'allowed_indices' list found in manifest payload")

    return _normalize_indices(indices)


def assign_holdout_fold(index, num_folds, seed=0):
    if num_folds <= 0:
        raise ValueError(f"num_folds must be positive, got {num_folds}")
    return (int(index) + int(seed)) % int(num_folds)


def load_reference_manifest(reference_manifest_path):
    manifest = load_json(reference_manifest_path)
    references = manifest.get("references", None)
    if not references:
        raise ValueError(
            f"Reference manifest {reference_manifest_path} does not contain any references"
        )
    return manifest


def build_fold_manifests(
    all_indices,
    num_folds,
    fold_assignment_seed,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_indices = _normalize_indices(all_indices)
    fold_manifests = []
    for fold_id in range(num_folds):
        heldout_indices = [
            idx
            for idx in normalized_indices
            if assign_holdout_fold(idx, num_folds, fold_assignment_seed) == fold_id
        ]
        heldout_set = set(heldout_indices)
        train_indices = [idx for idx in normalized_indices if idx not in heldout_set]

        train_manifest = {
            "fold_id": fold_id,
            "split": "train",
            "num_folds": int(num_folds),
            "allowed_indices": train_indices,
            "num_examples": len(train_indices),
            "total_examples": len(normalized_indices),
        }
        heldout_manifest = {
            "fold_id": fold_id,
            "split": "heldout",
            "num_folds": int(num_folds),
            "allowed_indices": heldout_indices,
            "num_examples": len(heldout_indices),
            "total_examples": len(normalized_indices),
        }

        train_manifest_path = output_dir / f"fold{fold_id}_train.json"
        heldout_manifest_path = output_dir / f"fold{fold_id}_heldout.json"
        save_json(train_manifest_path, train_manifest)
        save_json(heldout_manifest_path, heldout_manifest)

        fold_manifests.append(
            {
                "fold_id": fold_id,
                "train_manifest_path": str(train_manifest_path.resolve()),
                "heldout_manifest_path": str(heldout_manifest_path.resolve()),
                "train_indices": train_indices,
                "heldout_indices": heldout_indices,
            }
        )

    return fold_manifests


def build_reference_manifest(
    metadata,
    fold_manifests,
    checkpoint_root_dir,
    reference_manifest_output_path,
    validate_checkpoint_paths=False,
):
    checkpoint_root_dir = Path(checkpoint_root_dir)
    references = []
    for fold_manifest in fold_manifests:
        checkpoint_path = checkpoint_root_dir / f"fold{fold_manifest['fold_id']}"
        if validate_checkpoint_paths and not checkpoint_path.exists():
            raise ValueError(
                f"Expected checkpoint path for fold {fold_manifest['fold_id']} not found: {checkpoint_path}"
            )

        references.append(
            {
                "fold_id": int(fold_manifest["fold_id"]),
                "train_manifest_path": fold_manifest["train_manifest_path"],
                "heldout_manifest_path": fold_manifest["heldout_manifest_path"],
                "checkpoint_path": str(checkpoint_path.resolve()),
                "method": metadata["method"],
                "model": metadata["model"],
                "forget_split": metadata["forget_split"],
                "retain_split": metadata["retain_split"],
            }
        )

    reference_manifest = {
        "metadata": {
            **metadata,
            "num_reference_models": len(references),
        },
        "references": references,
    }
    save_json(reference_manifest_output_path, reference_manifest)
    return reference_manifest


def clone_dataset_with_indices(dataset, allowed_indices):
    subset = copy.copy(dataset)
    subset.data = filter_dataset_by_index(dataset.data, _normalize_indices(allowed_indices))
    return subset


def order_indices_by_difficulty(all_indices, scores_by_index):
    scored_indices = [
        int(idx)
        for idx, _ in sorted(
            scores_by_index.items(),
            key=lambda item: (float(item[1]["score"]), int(item[0])),
        )
    ]
    missing_indices = sorted(set(int(idx) for idx in all_indices) - set(scored_indices))
    return scored_indices + missing_indices


def build_stage_manifests(
    difficulty_payload,
    stage_percentiles,
    stage_epoch_ratios=None,
):
    if not stage_percentiles:
        raise ValueError("stage_percentiles must contain at least one value")

    if stage_epoch_ratios is None:
        stage_epoch_ratios = [1.0 / len(stage_percentiles)] * len(stage_percentiles)

    if len(stage_percentiles) != len(stage_epoch_ratios):
        raise ValueError("stage_percentiles and stage_epoch_ratios must have same length")

    all_indices = difficulty_payload["metadata"]["all_indices"]
    scores_by_index = difficulty_payload["scores_by_index"]
    ordered_indices = order_indices_by_difficulty(all_indices, scores_by_index)
    total_examples = len(ordered_indices)

    manifests = []
    for stage_id, (percentile, epoch_ratio) in enumerate(
        zip(stage_percentiles, stage_epoch_ratios), start=1
    ):
        if percentile <= 0 or percentile > 1:
            raise ValueError(
                f"stage percentile must be in (0, 1], got {percentile} at stage {stage_id}"
            )

        subset_size = total_examples
        if percentile < 1:
            subset_size = max(1, math.ceil(total_examples * percentile))

        manifests.append(
            {
                "stage_id": stage_id,
                "stage_name": f"stage{stage_id}",
                "percentile": float(percentile),
                "epoch_ratio": float(epoch_ratio),
                "allowed_indices": ordered_indices[:subset_size],
                "num_examples": subset_size,
                "total_examples": total_examples,
            }
        )
    return manifests
