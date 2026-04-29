#!/usr/bin/env python

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
UNLEARN_DIR = ROOT / "saves" / "unlearn"
BASELINE = (
    UNLEARN_DIR
    / "tofu_Llama-3.2-3B-Instruct_forget10_NPO_effbs32_8x4_full_s0"
    / "evals"
    / "TOFU_SUMMARY.json"
)
OUT_CSV = UNLEARN_DIR / "infocurl_effbs32_search_analysis.csv"

METRIC_DIRECTIONS = {
    "memorization": "higher",
    "model_utility": "higher",
    "forget_Q_A_Prob": "lower",
    "forget_Q_A_ROUGE": "lower",
    "forget_quality": "lower",
    "forget_truth_ratio": "lower",
    "privleak": "lower",
    "exact_memorization": "lower",
    "extraction_strength": "lower",
}


def parse_run_name(name: str):
    parts = name.split("_")
    mode = None
    gamma = None
    k = None
    for part in parts:
        if part in {"easy", "hard", "active"}:
            mode = part
        elif part.startswith("g"):
            gamma = part
        elif part.startswith("k"):
            k = part
    return {"mode": mode, "gamma": gamma, "K": k}


def metric_win(delta: float, direction: str):
    if direction == "higher":
        return delta > 0
    return delta < 0


def main():
    baseline = json.loads(BASELINE.read_text())
    rows = []
    grouped = {"mode": {}, "gamma": {}, "K": {}}

    for run_dir in sorted(UNLEARN_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        if "InfoCURL" not in run_dir.name or "effbs32" not in run_dir.name:
            continue
        summary_path = run_dir / "evals" / "TOFU_SUMMARY.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        meta = parse_run_name(run_dir.name)

        row = {"run_name": run_dir.name, **meta}
        win_count = 0
        for metric, direction in METRIC_DIRECTIONS.items():
            if metric not in summary or metric not in baseline:
                continue
            delta = summary[metric] - baseline[metric]
            row[f"delta_{metric}"] = delta
            if metric_win(delta, direction):
                win_count += 1
        row["win_count"] = win_count
        rows.append(row)

        for axis in ["mode", "gamma", "K"]:
            key = meta[axis]
            if key is None:
                continue
            grouped[axis].setdefault(key, []).append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(OUT_CSV)
    print("\nRanked by win_count:")
    for row in sorted(rows, key=lambda item: (-item["win_count"], item["run_name"])):
        print(f"{row['run_name']}: win_count={row['win_count']}")

    print("\nAverage win_count by factor:")
    for axis, groups in grouped.items():
        print(f"[{axis}]")
        for key, group_rows in sorted(groups.items()):
            print(f"  {key}: {mean(row['win_count'] for row in group_rows):.2f}")


if __name__ == "__main__":
    main()
