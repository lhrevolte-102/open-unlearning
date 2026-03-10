import json
import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "saves", "eval")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saves", "summary", "data")

TARGET_MODELS = {
    "tofu_Llama-3.2-3B-Instruct_full": "Full",
    "tofu_Llama-3.2-3B-Instruct_retain90": "Retain",
}

UNLEARN_MODELS = {
    "tofu_Llama-3.2-3B-Instruct_forget10_NPO_alpha1_beta0.05": "NPO-α1-β0.05",
    "tofu_Llama-3.2-3B-Instruct_forget10_DPO_alpha1_beta0.1": "DPO-α1-β0.1",
    # "tofu_Llama-3.2-3B-Instruct_forget10_NPO_alpha1_beta0.1": "NPO-α1-β0.1",
    # "tofu_Llama-3.2-3B-Instruct_forget10_NPO_alpha1_beta0.5": "NPO-α1-β0.5",
    # "tofu_Llama-3.2-3B-Instruct_forget10_RMU_alpha1_steering_coeff1": "RMU-α1-sc1",
    # "tofu_Llama-3.2-3B-Instruct_forget10_RMU_alpha1_steering_coeff10": "RMU-α1-sc10",
}

# All four conditions in unified form
CONDITIONS = ["base_base", "base_pr", "pr_base", "pr_pr"]

CONDITION_LABELS = {
    "base_base": "Baseline - Baseline",
    "base_pr": "Baseline - Prompt Repetition",
    "pr_base": "Prompt Repetition - Baseline",
    "pr_pr": "Prompt Repetition - Prompt Repetition",
}

# Target models only have base / pr (maps to unified conditions)
TARGET_CONDITION_MAP = {
    "base_base": "base",
    "base_pr": "pr",
    "pr_base": None,  # fill with 0
    "pr_pr": None,  # fill with 0
}

METRIC_ORDER = [
    "Exact Memorization",
    "Extraction Strength",
    "Prob.",
    "Para. Prob.",
    "ROUGE",
    "Para. ROUGE",
    "Truth Ratio",
    "Norm. PrivLeak",
    "Model Utility",
]


def standardize_metric(name: str) -> str:
    """Convert metric key to Title Case standard form."""
    # Remove prefixes
    name = name.replace("forget_Q_A_", "")
    name = name.replace("forget_truth_ratio", "truth_ratio")
    result = name.replace("_", " ").title()
    # Preserve ROUGE uppercase
    result = result.replace("Rouge", "ROUGE")
    # Add dots after Para and Prob.
    result = result.replace("Para", "Para.")
    result = result.replace("Prob", "Prob.")
    # Rename Privleak to Norm. PrivLeak
    result = result.replace("Privleak", "Norm. PrivLeak")
    return result


def normalize_value(metric: str, value):
    """Apply per-metric normalization."""
    if metric == "privleak":
        return abs(value / 100)
    return value


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_model_data(model_dir: str, condition_suffix: str):
    """Load TOFU_SUMMARY.json for a given model directory and condition suffix."""
    folder = f"{model_dir}_{condition_suffix}"
    path = os.path.join(BASE_DIR, folder, "TOFU_SUMMARY.json")
    return load_json(path)


def collect_rows():
    rows = []

    # --- Target models ---
    for model_key, model_short in TARGET_MODELS.items():
        for cond in CONDITIONS:
            target_suffix = TARGET_CONDITION_MAP[cond]
            if target_suffix is not None:
                data = load_model_data(model_key, target_suffix)
            else:
                data = None  # will fill zeros

            if data is not None:
                for metric, value in data.items():
                    rows.append(
                        {
                            "Model": model_short,
                            "Condition": CONDITION_LABELS[cond],
                            "Metric": standardize_metric(metric),
                            "Value": normalize_value(metric, value),
                        }
                    )
            else:
                # Use keys from any available file to keep metric list consistent
                ref_data = load_model_data(model_key, "base")
                if ref_data is None:
                    continue
                for metric in ref_data:
                    rows.append(
                        {
                            "Model": model_short,
                            "Condition": CONDITION_LABELS[cond],
                            "Metric": standardize_metric(metric),
                            "Value": 0,
                        }
                    )

    # --- Unlearn models ---
    for model_key, model_short in UNLEARN_MODELS.items():
        for cond in CONDITIONS:
            data = load_model_data(model_key, cond)
            if data is None:
                print(f"[WARN] Missing: {model_key}_{cond}")
                continue
            for metric, value in data.items():
                rows.append(
                    {
                        "Model": model_short,
                        "Condition": CONDITION_LABELS[cond],
                        "Metric": standardize_metric(metric),
                        "Value": normalize_value(metric, value),
                    }
                )

    return rows


def main():
    rows = collect_rows()
    if not rows:
        print("No data found.")
        return

    df = pd.DataFrame(rows)
    # Pivot so each metric is a column
    df_pivot = df.pivot_table(
        index=["Model", "Condition"],
        columns="Metric",
        values="Value",
        aggfunc="first",
    ).reset_index()
    df_pivot.columns.name = None

    # Reorder columns
    ordered_cols = ["Model", "Condition"] + [m for m in METRIC_ORDER if m in df_pivot.columns]
    # Append any remaining columns not in METRIC_ORDER
    remaining = [c for c in df_pivot.columns if c not in ordered_cols]
    df_pivot = df_pivot[ordered_cols + remaining]

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "tofu_summary_pr_1.csv")
    df_pivot.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
