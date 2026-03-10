import pandas as pd
import altair as alt
from pathlib import Path

# ── configuration ──────────────────────────────────────────────────────────────
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "saves"
    / "summary"
    / "data"
    / "tofu_summary_pr_1.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "saves" / "summary" / "figures"

MODEL_ORDER = [
    "Full",
    "Retain",
    "DPO-α1-β0.1",
    "NPO-α1-β0.05",
    # "NPO-α1-β0.1",
    # "NPO-α1-β0.5",
    # "RMU-α1-sc1",
    # "RMU-α1-sc10",
]

CONDITION_ORDER = [
    "Baseline - Baseline",
    "Baseline - Prompt Repetition",
    "Prompt Repetition - Baseline",
    "Prompt Repetition - Prompt Repetition",
]

CONDITION_COLORS = {
    "Baseline - Baseline": "#4C78A8",
    "Baseline - Prompt Repetition": "#F58518",
    "Prompt Repetition - Baseline": "#54A24B",
    "Prompt Repetition - Prompt Repetition": "#E45756",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in ("Model", "Condition")]
    long = df.melt(
        id_vars=["Model", "Condition"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Value",
    )
    return long


def make_chart(
    df_long: pd.DataFrame, metric_order: list[str], ncols: int = 4
) -> alt.FacetChart:
    color_scale = alt.Scale(
        domain=list(CONDITION_COLORS.keys()),
        range=list(CONDITION_COLORS.values()),
    )

    bar = (
        alt.Chart()
        .mark_bar(size=10)
        .encode(
            y=alt.Y(
                "Model:N",
                sort=MODEL_ORDER,
                title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=10),
            ),
            yOffset=alt.YOffset(
                "Condition:N",
                sort=CONDITION_ORDER,
                scale=alt.Scale(paddingInner=0.1),
            ),
            x=alt.X(
                "Value:Q",
                title=None,
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(grid=True, tickCount=5),
            ),
            color=alt.Color(
                "Condition:N",
                scale=color_scale,
                legend=alt.Legend(title=None, orient="right"),
            ),
        )
        .properties(height=alt.Step(10))
    )

    chart = (
        alt.layer(bar, data=df_long)
        .facet(
            facet=alt.Facet(
                "Metric:N",
                sort=metric_order,
                title=None,
                header=alt.Header(labelFontSize=12),
            ),
            columns=ncols,
        )
        .resolve_scale(x="independent", y="shared")
    )
    return chart


def save_pdf(chart, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import vl_convert as vlc

        pdf_bytes = vlc.vegalite_to_pdf(chart.to_json())
        path.write_bytes(pdf_bytes)
        print(f"Saved: {path}")
    except ImportError:
        try:
            chart.save(str(path))
            print(f"Saved (altair_saver): {path}")
        except Exception as e:
            html_path = path.with_suffix(".html")
            chart.save(str(html_path))
            print(f"[WARN] PDF export failed ({e}). Saved as HTML: {html_path}")


def main():
    alt.data_transformers.enable("default", max_rows=None)

    df = load_data()
    df_long = melt_to_long(df)

    # Get ordered metric list from columns (preserving CSV column order)
    metric_cols = [c for c in df.columns if c not in ("Model", "Condition")]

    chart = make_chart(df_long, metric_order=metric_cols, ncols=3)
    save_pdf(chart, OUTPUT_DIR / "tofu_summary_pr_1.pdf")


if __name__ == "__main__":
    main()
