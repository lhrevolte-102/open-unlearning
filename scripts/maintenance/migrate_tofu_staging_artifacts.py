#!/usr/bin/env python3

import csv
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAVES = ROOT / "saves"


def rename_hparam_order(name: str) -> str:
    name = re.sub(r"_lr([^_]+)_beta([^_]+)_alpha([^_]+)_", r"_lr\1_alpha\3_beta\2_", name)
    return name.replace("_mrd_npo_", "_MRD-NPO_")


def rename_dir(src: Path, dst: Path) -> None:
    if not src.exists() or src == dst:
        return
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def rewrite_stage_jsons(stage_root: Path) -> None:
    if not stage_root.exists():
        return

    for stage_file in sorted(stage_root.glob("stage*.json")):
        with stage_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.pop("subset_mode", None)
        with stage_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)

    summary_file = stage_root / "stages.json"
    if not summary_file.exists():
        return

    with summary_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.pop("stage_subset_mode", None)
    for stage in payload.get("stages", []):
        stage_name = stage.get("stage_name")
        if stage_name:
            stage["output_path"] = str((stage_root / f"{stage_name}.json").resolve())
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def migrate_selective_name(name: str) -> str:
    return name.replace("_cumulative_pct", "_pct")


def migrate_original_name(name: str) -> str:
    return re.sub(r"_cumulative_pct[^_]+_", "_", name)


def migrate_cumulative_dirs() -> None:
    selective_stage_root = SAVES / "selective" / "stage"
    for src in sorted(selective_stage_root.glob("*_cumulative_pct*_stages")):
        dst = src.with_name(migrate_selective_name(src.name))
        rename_dir(src, dst)
        rewrite_stage_jsons(dst / "stages")

    original_stage_root = SAVES / "original" / "stage"
    for src in sorted(original_stage_root.glob("*_cumulative_pct*_stages")):
        dst = src.with_name(migrate_original_name(src.name))
        rename_dir(src, dst)
        rewrite_stage_jsons(dst / "stages")

    unlearn_root = SAVES / "unlearn"
    for src in sorted(unlearn_root.glob("*_Selective-*_cumulative_pct*")):
        dst = src.with_name(migrate_selective_name(src.name))
        rename_dir(src, dst)

    for src in sorted(unlearn_root.glob("*_NPO_beta*_epoch*_cumulative_pct*_stage*")):
        dst = src.with_name(migrate_original_name(src.name))
        rename_dir(src, dst)

    for src in sorted(unlearn_root.glob("*_DPO_beta*_epoch*_cumulative_pct*_stage*")):
        dst = src.with_name(migrate_original_name(src.name))
        rename_dir(src, dst)


def rewrite_text_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    updated = text
    updated = updated.replace("_cumulative_pct", "_pct")
    updated = re.sub(r"_(NPO|DPO)_beta([^_]+)_epoch([^_]+)_cumulative_pct[^_]+_", r"_\1_beta\2_epoch\3_", updated)
    updated = re.sub(r"_lr([^_]+)_beta([^_]+)_alpha([^_]+)_", r"_lr\1_alpha\3_beta\2_", updated)
    updated = updated.replace("_mrd_npo_", "_MRD-NPO_")
    updated = updated.replace("mrd_npo,", "MRD-NPO,")
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def rename_hparam_dirs() -> None:
    roots = [
        SAVES / "unlearn",
        SAVES / "mrd",
        SAVES / "selective" / "stage",
        SAVES / "selective" / "prepare",
        SAVES / "selective" / "reference",
    ]
    for root in roots:
        if not root.exists():
            continue
        for src in sorted(root.iterdir(), key=lambda p: len(p.name), reverse=True):
            dst = src.with_name(rename_hparam_order(src.name))
            rename_dir(src, dst)


def rewrite_nested_metadata() -> None:
    for path in SAVES.rglob("*.json"):
        rewrite_text_file(path)
    for path in SAVES.rglob("*.yaml"):
        rewrite_text_file(path)
    for path in SAVES.rglob("*.csv"):
        rewrite_text_file(path)


def delete_disjoint_dirs() -> None:
    for path in sorted(SAVES.rglob("*disjoint*"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)


def rewrite_eval_summary() -> None:
    path = SAVES / "unlearn" / "eval_summary.csv"
    if not path.exists():
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys() if rows else []

    rewritten = []
    for row in rows:
        strategy = row["strategy"]
        if "disjoint" in strategy:
            continue
        if strategy == "cumulative":
            row["strategy"] = "original"
        elif strategy == "selective-cumulative":
            row["strategy"] = "selective"
        rewritten.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rewritten)


def rewrite_model_utility_components() -> None:
    path = SAVES / "unlearn" / "model_utility_components.csv"
    if not path.exists():
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys() if rows else []

    for row in rows:
        if row["strategy"] == "original" and row["stage"] == "original":
            row["stage"] = "stage3"

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    migrate_cumulative_dirs()
    rename_hparam_dirs()
    rewrite_nested_metadata()
    delete_disjoint_dirs()
    rewrite_eval_summary()
    rewrite_model_utility_components()


if __name__ == "__main__":
    main()
