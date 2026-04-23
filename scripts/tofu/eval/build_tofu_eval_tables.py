#!/usr/bin/env python3

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT_DIR / "saves" / "eval"

SUMMARY_COLUMNS = [
    "method",
    "strategy",
    "order",
    "stage",
    "exact_memorization",
    "extraction_strength",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_truth_ratio",
    "mia_gradnorm",
    "mia_loss",
    "mia_min_k",
    "mia_min_k_plus_plus",
    "mia_reference",
    "mia_zlib",
    "model_utility",
    "privleak",
]

ALL_STAGES_COLUMNS = [
    "method",
    "strategy",
    "stage",
    "exact_memorization",
    "extraction_strength",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_truth_ratio",
    "mia_gradnorm",
    "mia_loss",
    "mia_min_k",
    "mia_min_k_plus_plus",
    "mia_reference",
    "mia_zlib",
    "model_utility",
    "privleak",
]

UTILITY_COLUMNS = [
    "method",
    "strategy",
    "order",
    "stage",
    "model_utility",
    "retain_Q_A_Prob",
    "retain_Q_A_ROUGE",
    "retain_Truth_Ratio",
    "ra_Q_A_Prob_normalised",
    "ra_Q_A_ROUGE",
    "ra_Truth_Ratio",
    "wf_Q_A_Prob_normalised",
    "wf_Q_A_ROUGE",
    "wf_Truth_Ratio",
]

ROUND_TO_STAGE = {
    "01": "stage1",
    "03": "stage2",
    "05": "stage3",
}


@dataclass(frozen=True)
class RunMeta:
    task_name: str
    method: str
    strategy: str
    order: str
    stage: str
    sort_key: tuple[int, str, str, str]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scalar_metric(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, dict):
        return value.get("agg_value")
    return value


def parse_task_name(task_name: str) -> RunMeta | None:
    if "forget" not in task_name:
        return None

    if "MRD-NPO" in task_name:
        round_match = re.search(r"_round(\d{2})$", task_name)
        if round_match is None:
            return None
        round_id = round_match.group(1)
        if round_id not in ROUND_TO_STAGE:
            return None
        return RunMeta(
            task_name=task_name,
            method="NPO",
            strategy="MRD",
            order="original",
            stage=ROUND_TO_STAGE[round_id],
            sort_key=(2, "NPO", "MRD", ROUND_TO_STAGE[round_id]),
        )

    selective_match = re.search(r"_Selective-(DPO|NPO)_.+_(random|strict)_stage([123])$", task_name)
    if selective_match is not None:
        method = selective_match.group(1)
        order = selective_match.group(2)
        stage = f"stage{selective_match.group(3)}"
        return RunMeta(
            task_name=task_name,
            method=method,
            strategy="selective",
            order=order,
            stage=stage,
            sort_key=(1, method, order, stage),
        )

    original_match = re.search(r"_(DPO|NPO)_beta[^/]*_stage([123])$", task_name)
    if original_match is not None:
        method = original_match.group(1)
        stage = f"stage{original_match.group(2)}"
        return RunMeta(
            task_name=task_name,
            method=method,
            strategy="original",
            order="original",
            stage=stage,
            sort_key=(0, method, "original", stage),
        )

    return None


def collect_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    utility_rows: list[dict[str, object]] = []

    for summary_path in sorted(EVAL_ROOT.glob("*/TOFU_SUMMARY.json")):
        task_name = summary_path.parent.name
        meta = parse_task_name(task_name)
        if meta is None:
            continue

        eval_path = summary_path.parent / "TOFU_EVAL.json"
        if not eval_path.exists():
            raise SystemExit(f"Missing TOFU_EVAL.json for {task_name}")

        summary_payload = load_json(summary_path)
        eval_payload = load_json(eval_path)

        summary_row = {
            "method": meta.method,
            "strategy": meta.strategy,
            "order": meta.order,
            "stage": meta.stage,
        }
        for column in SUMMARY_COLUMNS[4:]:
            summary_row[column] = summary_payload.get(column)
        summary_row["_sort_key"] = meta.sort_key
        summary_rows.append(summary_row)

        utility_row = {
            "method": meta.method,
            "strategy": meta.strategy,
            "order": meta.order,
            "stage": meta.stage,
        }
        for column in UTILITY_COLUMNS[4:]:
            utility_row[column] = scalar_metric(eval_payload, column)
        utility_row["_sort_key"] = meta.sort_key
        utility_rows.append(utility_row)

    summary_rows.sort(key=lambda row: row["_sort_key"])
    utility_rows.sort(key=lambda row: row["_sort_key"])
    return summary_rows, utility_rows


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_csv_value(row.get(column)) for column in columns})


def final_stage_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["stage"] == "stage3"]


def final_display_name(row: dict[str, object]) -> str:
    strategy = str(row["strategy"])
    method = str(row["method"])
    if strategy == "MRD":
        return f"MRD-{method}"
    return f"{strategy}-{method}"


def title_case_label(label: str) -> str:
    replacements = {
        "exact_memorization": "Exact Memorization",
        "extraction_strength": "Extraction Strength",
        "forget_Q_A_Prob": "Prob",
        "forget_Q_A_ROUGE": "ROUGE",
        "forget_truth_ratio": "Truth Ratio",
        "mia_gradnorm": "MIA Gradnorm",
        "mia_loss": "MIA Loss",
        "mia_min_k": "MIA Min K",
        "mia_min_k_plus_plus": "MIA Min K Plus Plus",
        "mia_reference": "MIA Reference",
        "mia_zlib": "MIA Zlib",
        "model_utility": "Model Utility",
        "privleak": "Privleak",
        "retain_Q_A_Prob": "Retain Q A Prob",
        "retain_Q_A_ROUGE": "Retain Q A ROUGE",
        "retain_Truth_Ratio": "Retain Truth Ratio",
        "ra_Q_A_Prob_normalised": "Ra Q A Prob Normalised",
        "ra_Q_A_ROUGE": "Ra Q A ROUGE",
        "ra_Truth_Ratio": "Ra Truth Ratio",
        "wf_Q_A_Prob_normalised": "Wf Q A Prob Normalised",
        "wf_Q_A_ROUGE": "Wf Q A ROUGE",
        "wf_Truth_Ratio": "Wf Truth Ratio",
        "method": "Method",
        "strategy": "Strategy",
        "stage": "Stage",
    }
    if label in replacements:
        return replacements[label]
    return " ".join(part[:1].upper() + part[1:] for part in label.replace("-", " ").split("_"))


def pretty_strategy(value: str) -> str:
    if value.upper() == "MRD":
        return "MRD"
    return value[:1].upper() + value[1:].lower()


def pretty_stage(value: str) -> str:
    if value.startswith("stage"):
        return f"Stage{value[5:]}"
    return value


def format_csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def main() -> None:
    summary_rows, utility_rows = collect_rows()
    if not summary_rows:
        raise SystemExit(f"No TOFU eval runs found under {EVAL_ROOT}")

    final_summary_rows = final_stage_rows(summary_rows)
    final_utility_rows = final_stage_rows(utility_rows)
    for row_group in (summary_rows, utility_rows, final_summary_rows, final_utility_rows):
        for row in row_group:
            row["method"] = str(row["method"]).upper()
            row["strategy"] = pretty_strategy(str(row["strategy"]))
            row["stage"] = pretty_stage(str(row["stage"]))

    write_csv(
        EVAL_ROOT / "tofu_eval_summary_final.csv",
        final_summary_rows,
        ["method", "strategy", *SUMMARY_COLUMNS[4:]],
    )
    write_csv(
        EVAL_ROOT / "tofu_model_utility_components_final.csv",
        final_utility_rows,
        ["method", "strategy", *UTILITY_COLUMNS[4:]],
    )
    write_csv(EVAL_ROOT / "tofu_eval_summary_all_stages.csv", summary_rows, ALL_STAGES_COLUMNS)

    for path in (
        EVAL_ROOT / "tofu_eval_summary_final.csv",
        EVAL_ROOT / "tofu_model_utility_components_final.csv",
        EVAL_ROOT / "tofu_eval_summary_all_stages.csv",
    ):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        rows[0] = [title_case_label(cell) for cell in rows[0]]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
