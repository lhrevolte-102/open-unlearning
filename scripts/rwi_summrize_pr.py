import json
import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "saves", "eval")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saves", "summary", "data")

INITIAL_MODELS = {
    "rwi_Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "rwi_Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
}

CONDITIONS = ["base", "pr"]

CONDITION_LABELS = {
    "base": "Baseline",
    "pr": "Prompt Repetition",
}

METRIC_ORDER = [
    "Exact Memorization",
    "Extraction Strength",
    "Prob.",
    "Para. Prob.",
    "ROUGE",
    "Para. ROUGE",
    "Model Utility",
]


def standardize_metric(name: str) -> str:
    """Convert metric key to display form."""
    name = name.replace("forget_Q_A_", "")
    result = name.replace("_", " ").title()
    result = result.replace("Rouge", "ROUGE")
    result = result.replace("Para", "Para.")
    result = result.replace("Prob", "Prob.")
    result = result.replace("Privleak", "Norm. PrivLeak")
    return result


def normalize_value(metric: str, value):
    """Apply per-metric normalization if needed."""
    if metric == "privleak":
        return abs(value / 100)
    return value


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_model_data(model_dir: str, condition_suffix: str):
    """Load RWI_SUMMARY.json for a given model and condition."""
    folder = f"{model_dir}_{condition_suffix}"
    path = os.path.join(BASE_DIR, folder, "RWI_SUMMARY.json")
    return load_json(path)


def collect_rows():
    rows = []
    for model_key, model_short in INITIAL_MODELS.items():
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
    df_pivot = df.pivot_table(
        index=["Model", "Condition"],
        columns="Metric",
        values="Value",
        aggfunc="first",
    ).reset_index()
    df_pivot.columns.name = None

    ordered_cols = ["Model", "Condition"] + [
        m for m in METRIC_ORDER if m in df_pivot.columns
    ]
    remaining = [c for c in df_pivot.columns if c not in ordered_cols]
    df_pivot = df_pivot[ordered_cols + remaining]

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "rwi_summary_pr_0.csv")
    df_pivot.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
