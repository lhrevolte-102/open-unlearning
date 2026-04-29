#!/usr/bin/env python

import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNLEARN_DIR = ROOT / "saves" / "unlearn"
OUT_CSV = ROOT / "saves" / "unlearn" / "tofu10_experiment_summary.csv"


def infer_trainer(run_name):
    if "InfoCURL_GradDiff" in run_name:
        return "InfoCURL_GradDiff"
    if "GradDiff" in run_name:
        return "GradDiff"
    if "InfoCURL_SimNPO" in run_name:
        return "InfoCURL_SimNPO"
    if "SimNPO" in run_name:
        return "SimNPO"
    if "InfoCURL_NPO" in run_name or "InfoCURL" in run_name:
        return "InfoCURL_NPO"
    if "NPO" in run_name:
        return "NPO"
    return "unknown"


def infer_setting(run_name):
    prefix = "tofu_Llama-3.2-3B-Instruct_forget10_"
    if run_name.startswith(prefix):
        return run_name[len(prefix) :]
    return run_name


def main():
    rows = []
    for run_dir in sorted(UNLEARN_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        if "forget10" not in run_dir.name:
            continue
        summary_path = run_dir / "evals" / "TOFU_SUMMARY.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

        trainer = infer_trainer(run_dir.name)
        rows.append(
            {
                "run_name": run_dir.name,
                "trainer": trainer,
                "setting": infer_setting(run_dir.name),
                "forget_Q_A_Prob": summary.get("forget_Q_A_Prob"),
                "forget_Q_A_ROUGE": summary.get("forget_Q_A_ROUGE"),
                "forget_quality": summary.get("forget_quality"),
                "forget_truth_ratio": summary.get("forget_truth_ratio"),
                "model_utility": summary.get("model_utility"),
                "privleak": summary.get("privleak"),
                "exact_memorization": summary.get("exact_memorization"),
                "memorization": summary.get("memorization"),
                "extraction_strength": summary.get("extraction_strength"),
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "trainer",
                "setting",
                "forget_Q_A_Prob",
                "forget_Q_A_ROUGE",
                "forget_quality",
                "forget_truth_ratio",
                "model_utility",
                "privleak",
                "exact_memorization",
                "memorization",
                "extraction_strength",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(OUT_CSV)


if __name__ == "__main__":
    main()
