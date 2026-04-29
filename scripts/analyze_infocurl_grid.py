#!/usr/bin/env python

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNLEARN_DIR = ROOT / "saves" / "unlearn"
GRID_CSV = UNLEARN_DIR / "infocurl_grid_manifest.csv"
OUT_CSV = UNLEARN_DIR / "infocurl_grid_results.csv"
BASELINE = (
    UNLEARN_DIR
    / "tofu_Llama-3.2-3B-Instruct_forget10_NPO_bs8ga2_full_s0"
    / "evals"
    / "TOFU_SUMMARY.json"
)


def main():
    baseline = json.loads(BASELINE.read_text())
    with open(GRID_CSV, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    enriched = []
    for row in rows:
        summary_path = UNLEARN_DIR / row["run_tag"] / "evals" / "TOFU_SUMMARY.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        record = dict(row)
        for metric in [
            "memorization",
            "model_utility",
            "forget_Q_A_Prob",
            "forget_Q_A_ROUGE",
            "forget_quality",
            "forget_truth_ratio",
            "privleak",
        ]:
            record[metric] = summary.get(metric)
            if metric in baseline and metric in summary:
                record[f"delta_{metric}"] = summary[metric] - baseline[metric]
        enriched.append(record)

    fieldnames = sorted({key for row in enriched for key in row.keys()})
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    print(OUT_CSV)
    print("\nTop by memorization then model_utility:")
    ranked = sorted(
        enriched,
        key=lambda row: (float(row.get("memorization", -1e9)), float(row.get("model_utility", -1e9))),
        reverse=True,
    )
    for row in ranked[:20]:
        print(
            row["run_tag"],
            "mem=", row.get("memorization"),
            "mu=", row.get("model_utility"),
            "d_mem=", row.get("delta_memorization"),
            "d_mu=", row.get("delta_model_utility"),
        )


if __name__ == "__main__":
    main()
