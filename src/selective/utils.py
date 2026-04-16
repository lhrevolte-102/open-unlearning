import copy
import json
import math
import random
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


def _build_random_repeated_halving_reference_split_manifests(
    normalized_indices,
    num_repeats,
    repeat_split_seed,
    output_dir,
):
    if num_repeats is None:
        raise ValueError(
            "num_repeats must be explicitly set for "
            "reference_split_strategy='random_repeated_halving'"
        )
    if int(num_repeats) <= 0:
        raise ValueError(f"num_repeats must be positive, got {num_repeats}")
    if len(normalized_indices) < 2:
        raise ValueError(
            "random_repeated_halving requires at least 2 examples in the forget set"
        )

    total_examples = len(normalized_indices)
    first_half_size = total_examples // 2
    reference_split_manifests = []

    for repeat_id in range(int(num_repeats)):
        split_seed = int(repeat_split_seed) + repeat_id
        shuffled_indices = list(normalized_indices)
        random.Random(split_seed).shuffle(shuffled_indices)

        partition0_indices = sorted(shuffled_indices[:first_half_size])
        partition1_indices = sorted(shuffled_indices[first_half_size:])

        for partition_id, (train_indices, heldout_indices) in enumerate(
            (
                (partition0_indices, partition1_indices),
                (partition1_indices, partition0_indices),
            )
        ):
            split_id = repeat_id * 2 + partition_id
            split_name = f"split{split_id}"

            train_manifest = {
                "split_id": split_id,
                "split_name": split_name,
                "reference_split_strategy": "random_repeated_halving",
                "split_seed": split_seed,
                "repeat_id": repeat_id,
                "partition_id": partition_id,
                "split": "train",
                "num_repeats": int(num_repeats),
                "repeat_split_seed": int(repeat_split_seed),
                "allowed_indices": train_indices,
                "num_examples": len(train_indices),
                "total_examples": total_examples,
            }
            heldout_manifest = {
                "split_id": split_id,
                "split_name": split_name,
                "reference_split_strategy": "random_repeated_halving",
                "split_seed": split_seed,
                "repeat_id": repeat_id,
                "partition_id": partition_id,
                "split": "heldout",
                "num_repeats": int(num_repeats),
                "repeat_split_seed": int(repeat_split_seed),
                "allowed_indices": heldout_indices,
                "num_examples": len(heldout_indices),
                "total_examples": total_examples,
            }

            train_manifest_path = output_dir / f"{split_name}_train.json"
            heldout_manifest_path = output_dir / f"{split_name}_heldout.json"
            save_json(train_manifest_path, train_manifest)
            save_json(heldout_manifest_path, heldout_manifest)

            reference_split_manifests.append(
                {
                    "split_id": split_id,
                    "split_name": split_name,
                    "reference_split_strategy": "random_repeated_halving",
                    "split_seed": split_seed,
                    "repeat_id": repeat_id,
                    "partition_id": partition_id,
                    "train_manifest_path": str(train_manifest_path.resolve()),
                    "heldout_manifest_path": str(heldout_manifest_path.resolve()),
                    "train_indices": train_indices,
                    "heldout_indices": heldout_indices,
                }
            )

    return reference_split_manifests


def load_reference_manifest(reference_manifest_path):
    manifest = load_json(reference_manifest_path)
    references = manifest.get("references", None)
    if not references:
        raise ValueError(
            f"Reference manifest {reference_manifest_path} does not contain any references"
        )
    return manifest


def build_reference_split_manifests(
    all_indices,
    output_dir,
    num_repeats,
    repeat_split_seed=0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_indices = _normalize_indices(all_indices)
    return _build_random_repeated_halving_reference_split_manifests(
        normalized_indices=normalized_indices,
        num_repeats=num_repeats,
        repeat_split_seed=repeat_split_seed,
        output_dir=output_dir,
    )


def build_reference_manifest(
    metadata,
    reference_split_manifests,
    checkpoint_root_dir,
    reference_manifest_output_path,
    validate_checkpoint_paths=False,
):
    checkpoint_root_dir = Path(checkpoint_root_dir)
    references = []
    for split_manifest in reference_split_manifests:
        checkpoint_path = checkpoint_root_dir / split_manifest["split_name"]
        if validate_checkpoint_paths and not checkpoint_path.exists():
            raise ValueError(
                "Expected checkpoint path for "
                f"{split_manifest['split_name']} not found: {checkpoint_path}"
            )

        references.append(
            {
                "split_id": int(split_manifest["split_id"]),
                "split_name": split_manifest["split_name"],
                "reference_split_strategy": split_manifest["reference_split_strategy"],
                "split_seed": int(split_manifest["split_seed"]),
                "repeat_id": split_manifest.get("repeat_id", None),
                "partition_id": split_manifest.get("partition_id", None),
                "train_manifest_path": split_manifest["train_manifest_path"],
                "heldout_manifest_path": split_manifest["heldout_manifest_path"],
                "checkpoint_path": str(checkpoint_path.resolve()),
                "num_train_examples": len(split_manifest["train_indices"]),
                "num_heldout_examples": len(split_manifest["heldout_indices"]),
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
