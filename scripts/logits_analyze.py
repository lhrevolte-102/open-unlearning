#!/usr/bin/env python3
"""
Analyze logits from RWI/TOFU dataset for baseline and prompt repetition conditions.

This script extracts token log probabilities from *_EVAL.json files (using the
forget_Q_A_token_Prob metric) and performs comparative analysis between baseline
and prompt repetition conditions including:
- Log probability differences
- KL divergence computation
- Per-sample and aggregated statistics
- High-diff token extraction for sensitive entity analysis
- Visualization generation (PDF format)

Output structure:
- saves/summary/data/: JSON data files (aggregate stats, full results, high-diff tokens)
- saves/summary/figures/: PDF plot files

File naming convention:
- {dataset}_{model}_*.json/pdf (e.g., tofu_Llama-3.2-3B-Instruct_full_aggregate_stats.json)
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from html import escape

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for a model pair."""

    name: str
    model_path: str


@dataclass
class SampleData:
    """Data for a single sample."""

    index: int
    intra_item_idx: str
    num_target_tokens: int
    label_tokens: List[str]
    token_logprobs: List[float]
    token_probs: List[float]
    sequence_logprob: float
    avg_token_logprob: float


@dataclass
class AnalysisResult:
    """Results from analyzing a sample pair."""

    index: int
    intra_item_idx: str
    num_target_tokens: int
    label_tokens: List[str]

    # Baseline stats
    baseline_seq_logprob: float
    baseline_avg_logprob: float

    # PR stats
    pr_seq_logprob: float
    pr_avg_logprob: float

    # Differences
    seq_logprob_diff: float  # baseline - PR
    avg_logprob_diff: float  # baseline - PR

    # Per-token logprob differences
    per_token_diff: List[float]

    # KL divergence (symmetric)
    kl_baseline_to_pr: float
    kl_pr_to_baseline: float
    kl_symmetric: float


def load_token_probability_file(filepath: str) -> List[SampleData]:
    """Load samples from a token_probability.jsonl file."""
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            sample = SampleData(
                index=data["index"],
                intra_item_idx=data["intra_item_idx"],
                num_target_tokens=data["num_target_tokens"],
                label_tokens=data["label_tokens"],
                token_logprobs=data["token_logprobs"],
                token_probs=data["token_probs"],
                sequence_logprob=data["sequence_logprob"],
                avg_token_logprob=data["avg_token_logprob"],
            )
            samples.append(sample)
    return samples


def load_eval_file(
    filepath: str, metric_key: str = "forget_Q_A_token_Prob", dataset: str = "rwi"
) -> List[SampleData]:
    """Load samples from an RWI_EVAL.json or TOFU_EVAL.json file.

    Args:
        filepath: Path to the *_EVAL.json file
        metric_key: The metric key to extract (e.g., 'forget_Q_A_token_Prob')
        dataset: Dataset type ('rwi' or 'tofu')

    Returns:
        List of SampleData objects
    """
    samples = []
    with open(filepath, "r") as f:
        data = json.load(f)

    if metric_key not in data:
        raise ValueError(f"Metric key '{metric_key}' not found in {filepath}")

    metric_data = data[metric_key]
    value_by_index = metric_data.get("value_by_index", {})

    for index_str, item_data in value_by_index.items():
        index = int(index_str)
        sample = SampleData(
            index=index,
            intra_item_idx=str(index),  # Use index as intra_item_idx
            num_target_tokens=item_data.get(
                "num_tokens", len(item_data.get("tokens", []))
            ),
            label_tokens=item_data.get("tokens", []),
            token_logprobs=item_data.get("token_logprobs", []),
            token_probs=item_data.get("token_probs", []),
            sequence_logprob=item_data.get("sequence_logprob", 0.0),
            avg_token_logprob=item_data.get("mean_token_logprob", 0.0),
        )
        samples.append(sample)

    return samples


def compute_kl_divergence(
    p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10
) -> float:
    """
    Compute KL divergence D_KL(P || Q).

    Args:
        p: First probability distribution
        q: Second probability distribution (should be non-zero)
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence value
    """
    # Normalize to ensure valid probability distributions
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def analyze_sample_pair(baseline: SampleData, pr: SampleData) -> AnalysisResult:
    """Analyze a pair of baseline and PR samples."""
    # Convert to numpy arrays
    baseline_probs = np.array(baseline.token_probs, dtype=np.float64)
    pr_probs = np.array(pr.token_probs, dtype=np.float64)
    baseline_logprobs = np.array(baseline.token_logprobs, dtype=np.float64)
    pr_logprobs = np.array(pr.token_logprobs, dtype=np.float64)

    # Compute per-token log probability differences (baseline - PR)
    # More positive = baseline more confident
    # More negative = PR more confident
    per_token_diff: List[float] = [float(x) for x in (baseline_logprobs - pr_logprobs)]

    # Compute KL divergences
    kl_baseline_to_pr = compute_kl_divergence(baseline_probs, pr_probs)
    kl_pr_to_baseline = compute_kl_divergence(pr_probs, baseline_probs)
    kl_symmetric = kl_baseline_to_pr + kl_pr_to_baseline

    return AnalysisResult(
        index=baseline.index,
        intra_item_idx=baseline.intra_item_idx,
        num_target_tokens=baseline.num_target_tokens,
        label_tokens=baseline.label_tokens,
        baseline_seq_logprob=baseline.sequence_logprob,
        baseline_avg_logprob=baseline.avg_token_logprob,
        pr_seq_logprob=pr.sequence_logprob,
        pr_avg_logprob=pr.avg_token_logprob,
        seq_logprob_diff=baseline.sequence_logprob - pr.sequence_logprob,
        avg_logprob_diff=baseline.avg_token_logprob - pr.avg_token_logprob,
        per_token_diff=per_token_diff,
        kl_baseline_to_pr=kl_baseline_to_pr,
        kl_pr_to_baseline=kl_pr_to_baseline,
        kl_symmetric=kl_symmetric,
    )


def match_samples(
    baseline_samples: List[SampleData], pr_samples: List[SampleData]
) -> List[Tuple[SampleData, SampleData]]:
    """Match baseline and PR samples by index."""
    baseline_by_index = {s.index: s for s in baseline_samples}
    pr_by_index = {s.index: s for s in pr_samples}

    matched_pairs = []
    for idx in baseline_by_index:
        if idx in pr_by_index:
            matched_pairs.append((baseline_by_index[idx], pr_by_index[idx]))

    return matched_pairs


def compute_aggregate_stats(results: List[AnalysisResult]) -> Dict:
    """Compute aggregate statistics from analysis results."""
    if not results:
        return {}

    # Extract metrics
    seq_logprob_diffs: List[float] = [r.seq_logprob_diff for r in results]
    avg_logprob_diffs: List[float] = [r.avg_logprob_diff for r in results]
    kl_symmetric_values: List[float] = [r.kl_symmetric for r in results]
    kl_baseline_to_pr_values: List[float] = [r.kl_baseline_to_pr for r in results]
    kl_pr_to_baseline_values: List[float] = [r.kl_pr_to_baseline for r in results]

    # Compute per-token mean differences for each sample
    per_token_mean_diffs: List[float] = [
        float(np.mean(r.per_token_diff)) for r in results
    ]

    def compute_stats(values: List[float]) -> Dict:
        values_arr = np.array(values)
        return {
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr)),
            "median": float(np.median(values_arr)),
            "min": float(np.min(values_arr)),
            "max": float(np.max(values_arr)),
            "q25": float(np.percentile(values_arr, 25)),
            "q75": float(np.percentile(values_arr, 75)),
        }

    return {
        "num_samples": len(results),
        "sequence_logprob_diff": compute_stats(seq_logprob_diffs),
        "avg_token_logprob_diff": compute_stats(avg_logprob_diffs),
        "per_token_mean_logprob_diff": compute_stats(per_token_mean_diffs),
        "kl_symmetric": compute_stats(kl_symmetric_values),
        "kl_baseline_to_pr": compute_stats(kl_baseline_to_pr_values),
        "kl_pr_to_baseline": compute_stats(kl_pr_to_baseline_values),
    }


def generate_plots(results: List[AnalysisResult], output_dir: str, dataset: str = "rwi", model_name: str = ""):
    """Generate visualization plots as PDF files."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    seq_diffs = [r.seq_logprob_diff for r in results]
    kl_symmetric = [r.kl_symmetric for r in results]

    # Plot 1: Histogram of sequence log probability differences and KL divergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sequence logprob difference histogram
    axes[0].hist(seq_diffs, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0, color="r", linestyle="--", label="Zero")
    axes[0].set_xlabel("Sequence Log Probability Difference (Baseline - PR)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"[{dataset.upper()}] Distribution of Sequence Log Probability Differences")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # KL divergence histogram
    axes[1].hist(kl_symmetric, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Symmetric KL Divergence")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"[{dataset.upper()}] Distribution of KL Divergence (Symmetric)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset.lower()}_{model_name}_logit_analysis_overview.pdf"), format="pdf")
    plt.close()

    # Plot 2: Cumulative distribution functions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # CDF of sequence logprob differences
    sorted_seq_diffs = np.sort(seq_diffs)
    cdf_seq = np.arange(1, len(sorted_seq_diffs) + 1) / len(sorted_seq_diffs)
    axes[0].plot(sorted_seq_diffs, cdf_seq)
    axes[0].axvline(x=0, color="r", linestyle="--")
    axes[0].set_xlabel("Sequence Log Probability Difference")
    axes[0].set_ylabel("Cumulative Proportion")
    axes[0].set_title(f"[{dataset.upper()}] CDF of Log Probability Differences")
    axes[0].grid(True, alpha=0.3)

    # CDF of KL divergence
    sorted_kl = np.sort(kl_symmetric)
    cdf_kl = np.arange(1, len(sorted_kl) + 1) / len(sorted_kl)
    axes[1].plot(sorted_kl, cdf_kl)
    axes[1].set_xlabel("Symmetric KL Divergence")
    axes[1].set_ylabel("Cumulative Proportion")
    axes[1].set_title(f"[{dataset.upper()}] CDF of KL Divergence")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset.lower()}_{model_name}_cdf_analysis.pdf"), format="pdf")
    plt.close()

    print(f"Plots saved to {output_dir}")


def generate_token_highlight_pdf(
    results: List[AnalysisResult],
    output_dir: str,
    dataset: str = "rwi",
    model_name: str = "",
    positive_threshold: float = 1.0,
    negative_threshold: float = -1.0,
    max_samples: int = 6,
):
    """Generate PDF visualization with highlighted tokens using weasyprint.
    
    Args:
        results: List of AnalysisResult objects
        output_dir: Directory to save PDF file
        dataset: Dataset type ('rwi' or 'tofu')
        model_name: Model name for file naming
        positive_threshold: Threshold for positive diff (green, baseline more confident)
        negative_threshold: Threshold for negative diff (red, PR more confident)
        max_samples: Maximum number of samples to include
    """
    try:
        from weasyprint import HTML, CSS
    except ImportError:
        print("weasyprint not installed, run: pip install weasyprint")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate HTML content and calculate actual content height
    html_content, content_height_px = _generate_highlight_html_content(
        results, dataset, model_name, positive_threshold, negative_threshold, max_samples
    )
    
    # Convert to PDF with auto-sized page
    # Use content height to determine page size (1px ≈ 0.26mm at 96 DPI)
    # Add 20mm for body margin and padding
    content_height_mm = content_height_px * 0.26 + 20
    output_path = os.path.join(output_dir, f"{dataset.lower()}_{model_name}_token_highlight.pdf")
    
    html_obj = HTML(string=html_content)
    html_obj.write_pdf(
        output_path,
        stylesheets=[CSS(string=f"""
            @page {{
                size: 210mm {content_height_mm:.1f}mm;
                margin: 10mm;
            }}
            body {{
                font-family: "DejaVu Sans", "Arial", sans-serif;
                font-size: 7pt;
                width: 190mm;
            }}
            .sample {{
                margin: 8px 0;
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 2px;
            }}
            .token {{
                padding: 1px 2px;
                border-radius: 1px;
                display: inline;
                white-space: pre-wrap;
                word-break: break-all;
            }}
            @media print {{
                .sample {{ page-break-inside: avoid; }}
            }}
        """)]
    )
    
    print(f"PDF visualization saved to {output_path}")


def _generate_highlight_html_content(
    results: List[AnalysisResult],
    dataset: str,
    model_name: str,
    positive_threshold: float,
    negative_threshold: float,
    max_samples: int,
) -> Tuple[str, int]:
    """Generate HTML content for token highlight visualization.
    
    Returns:
        Tuple of (html_content, estimated_content_height_px)
    """
    # Filter samples with less than 100 tokens
    filtered_results = [r for r in results if len(r.label_tokens) < 100]
    
    # Select samples that have tokens with negative diff exceeding threshold (PR more confident)
    # Only keep samples where at least one token has diff < negative_threshold
    candidate_samples = [
        r for r in filtered_results
        if any(d < negative_threshold for d in r.per_token_diff)
    ]
    
    # Sort by number of negative diff tokens (descending), then by min diff value (ascending)
    samples_to_show = sorted(
        candidate_samples,
        key=lambda r: (-sum(1 for d in r.per_token_diff if d < negative_threshold), min(r.per_token_diff)),
    )[:max_samples]
    
    # Estimate content height based on actual token content
    # Each sample: h3 (~12px) + p with tokens (variable based on line count)
    # Line height is 1.4 * 7pt = ~14px per line
    content_height_px = 0
    line_height_px = 14  # 7pt * 1.4 line height
    chars_per_line = 120  # Approximate chars per line at 190mm width
    
    for result in samples_to_show:
        # h3 height
        content_height_px += 15
        
        # Calculate token content height
        total_token_chars = sum(len(t) for t in result.label_tokens)
        lines_needed = max(1, (total_token_chars + chars_per_line - 1) // chars_per_line)
        content_height_px += lines_needed * line_height_px
        
        # Sample padding and margin (~10px total)
        content_height_px += 15
    
    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Token Highlight Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 5px; font-size: 7pt; }
        .sample {
            margin: 5px 0;
            padding: 5px;
            border: 1px solid #eee;
            border-radius: 2px;
            page-break-inside: avoid;
        }
        .sample h3 { margin: 0 0 2px 0; color: #555; font-size: 8pt; }
        .token {
            padding: 1px 2px;
            border-radius: 1px;
            display: inline;
            white-space: pre-wrap;
            font-size: 7pt;
        }
    </style>
</head>
<body>
""",
    ]
    
    for result in samples_to_show:
        # Decode tokens and handle special characters
        tokens = [t.replace("Ġ", " ").replace("Ċ", "").replace("ĉ", "c") for t in result.label_tokens]
        diffs = result.per_token_diff
        
        token_spans = []
        for token, diff in zip(tokens, diffs):
            escaped_token = escape(token)
            
            # Highlight all tokens, use alpha to indicate diff magnitude
            # Positive diff (baseline more confident): green
            # Negative diff (PR more confident): red
            # Near zero (similar confidence): light gray
            if diff > 0.5:
                # Green for positive diff, alpha scales with diff value
                alpha = min(0.9, 0.15 + (diff / 5.0) * 0.75)
                color = f"rgba(0, 128, 0, {alpha})"
            elif diff < -0.5:
                # Red for negative diff, alpha scales with absolute diff value
                alpha = min(0.9, 0.15 + (abs(diff) / 5.0) * 0.75)
                color = f"rgba(255, 0, 0, {alpha})"
            else:
                # Light gray for zero diff
                color = "rgba(200, 200, 200, 0.3)"
            
            token_spans.append(
                f'<span class="token" style="background-color: {color}" '
                f'title="Token: {escaped_token} Diff: {diff:.3f}">{escaped_token}</span>'
            )
        
        html_parts.append(f"""
    <div class="sample">
        <h3>Sample {result.index}</h3>
        <p style="line-height: 1.4; margin: 3px 0;">{''.join(token_spans)}</p>
    </div>
""")
    
    html_parts.append("""
</body>
</html>
""")
    
    return "".join(html_parts), content_height_px


def extract_high_diff_tokens(
    results: List[AnalysisResult], threshold: float = 2.0, top_k: Optional[int] = None
) -> List[Dict]:
    """Extract tokens with high log probability differences for analysis.

    This function extracts tokens where the log probability difference between
    baseline and PR exceeds a threshold, which can help identify sensitive
    entities like names, locations, etc.

    Args:
        results: List of AnalysisResult objects
        threshold: Minimum absolute log prob difference to consider
        top_k: If specified, return only top_k highest diff tokens per sample

    Returns:
        List of dictionaries containing token info with high diff values
    """
    high_diff_tokens = []

    for result in results:
        for token_idx, (token, diff) in enumerate(
            zip(result.label_tokens, result.per_token_diff)
        ):
            if abs(diff) >= threshold:
                high_diff_tokens.append(
                    {
                        "index": result.index,
                        "intra_item_idx": result.intra_item_idx,
                        "token_position": token_idx,
                        "token": token,
                        "logprob_diff": diff,
                        "abs_diff": abs(diff),
                        "num_target_tokens": result.num_target_tokens,
                        "label_tokens": result.label_tokens,
                    }
                )

    # Sort by absolute difference (descending)
    high_diff_tokens.sort(key=lambda x: x["abs_diff"], reverse=True)

    # If top_k is specified, limit per sample
    if top_k:
        filtered_tokens = []
        sample_counts = defaultdict(int)
        for token_info in high_diff_tokens:
            sample_key = (token_info["index"], token_info["intra_item_idx"])
            if sample_counts[sample_key] < top_k:
                filtered_tokens.append(token_info)
                sample_counts[sample_key] += 1
        high_diff_tokens = filtered_tokens

    return high_diff_tokens


def save_high_diff_tokens_json(high_diff_tokens: List[Dict], output_path: str):
    """Save high-diff tokens to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(high_diff_tokens, f, indent=2, ensure_ascii=False)
    print(f"High-diff tokens saved to {output_path}")


def generate_per_token_plots(
    results: List[AnalysisResult],
    output_dir: str,
    dataset: str = "rwi",
    model_name: str = "",
    annotation_threshold: float = 1.0,
    negative_diff_threshold: float = -1.0,
    max_samples: int = 8,
):
    """Generate per-token token highlight visualization as PDF.

    This function generates a PDF file with token-level highlighting to show
    log probability differences between baseline and PR conditions.
    
    Tokens with high positive diff (baseline more confident) are highlighted in green.
    Tokens with high negative diff (PR more confident) are highlighted in red.

    Args:
        results: List of AnalysisResult objects
        output_dir: Directory to save plots
        dataset: Dataset type ('rwi' or 'tofu')
        model_name: Model name for file naming
        annotation_threshold: Minimum positive diff value for green highlighting
        negative_diff_threshold: Minimum negative diff value for red highlighting (should be negative)
        max_samples: Maximum number of samples to include in the visualization
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate token highlight PDF using weasyprint
    generate_token_highlight_pdf(
        results, output_dir, dataset, model_name,
        positive_threshold=annotation_threshold,
        negative_threshold=negative_diff_threshold,
        max_samples=max_samples,
    )

    print(f"Token highlight visualization saved to {output_dir}")


def generate_token_diff_summary_plot(results: List[AnalysisResult], output_dir: str, dataset: str = "rwi", model_name: str = ""):
    """Generate a summary plot showing representative tokens and their diff values."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping summary plot")
        return

    # Collect all tokens with their diff values
    all_token_diffs = []
    for result in results:
        for token, diff in zip(result.label_tokens, result.per_token_diff):
            all_token_diffs.append(
                {
                    "token": token,
                    "diff": diff,
                    "abs_diff": abs(diff),
                    "sample_index": result.index,
                }
            )

    # Sort by absolute diff and get top 50 unique tokens
    all_token_diffs.sort(key=lambda x: x["abs_diff"], reverse=True)

    # Get unique tokens (keep highest diff occurrence)
    seen_tokens = set()
    unique_top_tokens = []
    for item in all_token_diffs:
        if item["token"] not in seen_tokens:
            seen_tokens.add(item["token"])
            unique_top_tokens.append(item)
            if len(unique_top_tokens) >= 50:
                break

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Replace Ġ and Ċ with space for display
    tokens = [item["token"].replace("Ġ", " ").replace("Ċ", "") for item in unique_top_tokens]
    diffs = [item["diff"] for item in unique_top_tokens]
    colors = ["red" if d > 0 else "blue" for d in diffs]

    # Create barh plot without edge stroke
    ax.barh(range(len(tokens)), diffs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("Log Prob Difference (Baseline - PR)")
    ax.set_title(f"[{dataset.upper()}] Top 50 Tokens by Log Probability Difference")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (token, diff) in enumerate(zip(tokens, diffs)):
        ax.text(
            diff,
            i,
            f"{diff:.2f}",
            va="center",
            ha="left" if diff > 0 else "right",
            fontsize=7,
            color="black",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset.lower()}_{model_name}_top_token_diffs_summary.pdf"), format="pdf")
    plt.close()


def save_results(
    results: List[AnalysisResult],
    aggregate_stats: Dict,
    output_dir: str,
    model_name: str,
    dataset: str = "rwi",
    save_high_diff: bool = True,
    high_diff_threshold: float = 2.0,
):
    """Save analysis results to JSON files.

    Args:
        results: List of AnalysisResult objects
        aggregate_stats: Aggregate statistics dictionary
        output_dir: Base output directory
        model_name: Model name for file naming
        dataset: Dataset type ('rwi' or 'tofu')
        save_high_diff: Whether to save high-diff tokens
        high_diff_threshold: Threshold for high-diff token extraction
    """
    # Create subdirectories
    data_dir = os.path.join(output_dir, "data")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Save aggregate statistics as JSON
    json_path = os.path.join(data_dir, f"{dataset}_{model_name}_aggregate_stats.json")
    with open(json_path, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"Aggregate statistics saved to {json_path}")

    # Save full results as JSON
    json_full_path = os.path.join(data_dir, f"{dataset}_{model_name}_full_results.json")
    full_results = []
    for r in results:
        full_results.append(
            {
                "index": r.index,
                "intra_item_idx": r.intra_item_idx,
                "num_target_tokens": r.num_target_tokens,
                "label_tokens": r.label_tokens,
                "baseline_seq_logprob": r.baseline_seq_logprob,
                "baseline_avg_logprob": r.baseline_avg_logprob,
                "pr_seq_logprob": r.pr_seq_logprob,
                "pr_avg_logprob": r.pr_avg_logprob,
                "seq_logprob_diff": r.seq_logprob_diff,
                "avg_logprob_diff": r.avg_logprob_diff,
                "per_token_diff": r.per_token_diff,
                "kl_baseline_to_pr": r.kl_baseline_to_pr,
                "kl_pr_to_baseline": r.kl_pr_to_baseline,
                "kl_symmetric": r.kl_symmetric,
            }
        )

    with open(json_full_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Full results saved to {json_full_path}")

    # Save high-diff tokens for analysis
    if save_high_diff:
        high_diff_tokens = extract_high_diff_tokens(
            results, threshold=high_diff_threshold
        )
        high_diff_path = os.path.join(data_dir, f"{dataset}_{model_name}_high_diff_tokens.json")
        save_high_diff_tokens_json(high_diff_tokens, high_diff_path)


def analyze_model(
    model_name: str,
    base_path: str,
    pr_path: str,
    output_dir: str,
    dataset: str = "rwi",
    metric_key: str = "forget_Q_A_token_Prob",
) -> Optional[Dict]:
    """Analyze a single model's baseline vs PR outputs."""
    print(f"\n{'='*60}")
    print(f"Analyzing model: {model_name}")
    print(f"Dataset: {dataset.upper()}")
    print(f"{'='*60}")

    # Check if files exist
    if not os.path.exists(base_path):
        print(f"ERROR: Baseline file not found: {base_path}")
        return None
    if not os.path.exists(pr_path):
        print(f"ERROR: PR file not found: {pr_path}")
        return None

    # Load data
    print(f"Loading baseline data from {base_path}...")
    baseline_samples = load_eval_file(base_path, metric_key=metric_key, dataset=dataset)
    print(f"  Loaded {len(baseline_samples)} baseline samples")

    print(f"Loading PR data from {pr_path}...")
    pr_samples = load_eval_file(pr_path, metric_key=metric_key, dataset=dataset)
    print(f"  Loaded {len(pr_samples)} PR samples")

    # Match samples
    matched_pairs = match_samples(baseline_samples, pr_samples)
    print(f"  Matched {len(matched_pairs)} sample pairs")

    if not matched_pairs:
        print("WARNING: No matched samples found!")
        return None

    # Analyze each pair
    print("Analyzing sample pairs...")
    results = []
    for baseline, pr in matched_pairs:
        result = analyze_sample_pair(baseline, pr)
        results.append(result)

    # Compute aggregate statistics
    print("Computing aggregate statistics...")
    aggregate_stats = compute_aggregate_stats(results)

    # Print summary
    print(f"\n--- Summary for {model_name} ---")
    print(f"Number of samples: {aggregate_stats['num_samples']}")
    print(f"Sequence Log Probability Difference (Baseline - PR):")
    print(f"  Mean: {aggregate_stats['sequence_logprob_diff']['mean']:.6f}")
    print(f"  Std:  {aggregate_stats['sequence_logprob_diff']['std']:.6f}")
    print(f"  Median: {aggregate_stats['sequence_logprob_diff']['median']:.6f}")
    print(f"Symmetric KL Divergence:")
    print(f"  Mean: {aggregate_stats['kl_symmetric']['mean']:.6f}")
    print(f"  Std:  {aggregate_stats['kl_symmetric']['std']:.6f}")

    # Save results
    save_results(results, aggregate_stats, output_dir, model_name, dataset)

    # Generate plots
    figures_dir = os.path.join(output_dir, "figures")
    generate_plots(results, figures_dir, dataset, model_name)
    
    # Generate per-token token highlight visualization
    # Focus on samples with negative token diffs (PR more confident than baseline)
    generate_per_token_plots(results, figures_dir, dataset, model_name)

    return aggregate_stats


def main():
    """Main entry point.
    
    This script analyzes all available datasets (RWI and TOFU) and models
    without requiring command-line arguments.
    
    Default configurations:
    - RWI models: Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct
    - TOFU models: Llama-3.2-3B-Instruct_full, Llama-3.1-8B-Instruct_full
    - Metric key: forget_Q_A_token_Prob
    """
    eval_dir = "saves/eval"
    output_dir = "saves/summary"
    metric_key = "forget_Q_A_token_Prob"

    # Define all model configurations
    all_configs = [
        # RWI dataset
        {
            "dataset": "rwi",
            "models": {
                "Llama-3.2-3B-Instruct": {
                    "base": os.path.join(eval_dir, "rwi_Llama-3.2-3B-Instruct_base", "RWI_EVAL.json"),
                    "pr": os.path.join(eval_dir, "rwi_Llama-3.2-3B-Instruct_pr", "RWI_EVAL.json"),
                },
                "Llama-3.1-8B-Instruct": {
                    "base": os.path.join(eval_dir, "rwi_Llama-3.1-8B-Instruct_base", "RWI_EVAL.json"),
                    "pr": os.path.join(eval_dir, "rwi_Llama-3.1-8B-Instruct_pr", "RWI_EVAL.json"),
                },
            },
        },
        # TOFU dataset
        {
            "dataset": "tofu",
            "models": {
                "Llama-3.2-3B-Instruct_full": {
                    "base": os.path.join(eval_dir, "tofu_Llama-3.2-3B-Instruct_full_base", "TOFU_EVAL.json"),
                    "pr": os.path.join(eval_dir, "tofu_Llama-3.2-3B-Instruct_full_pr", "TOFU_EVAL.json"),
                },
                "Llama-3.1-8B-Instruct_full": {
                    "base": os.path.join(eval_dir, "tofu_Llama-3.1-8B-Instruct_full_base", "TOFU_EVAL.json"),
                    "pr": os.path.join(eval_dir, "tofu_Llama-3.1-8B-Instruct_full_pr", "TOFU_EVAL.json"),
                },
            },
        },
    ]

    # Analyze all datasets and models
    for config in all_configs:
        dataset = config["dataset"]
        models = config["models"]
        
        all_stats = {}
        for model_name, model_config in models.items():
            stats = analyze_model(
                model_name=model_name,
                base_path=model_config["base"],
                pr_path=model_config["pr"],
                output_dir=output_dir,
                dataset=dataset,
                metric_key=metric_key,
            )
            if stats:
                all_stats[model_name] = stats

        # Save combined summary for this dataset
        if all_stats:
            combined_summary_path = os.path.join(
                output_dir, "data", f"{dataset}_combined_summary.json"
            )
            os.makedirs(os.path.dirname(combined_summary_path), exist_ok=True)
            with open(combined_summary_path, "w") as f:
                json.dump(all_stats, f, indent=2)
            print(f"\nCombined summary saved to {combined_summary_path}")

            # Print combined comparison
            print(f"\n{'='*60}")
            print(f"[{dataset.upper()}] COMBINED MODEL COMPARISON")
            print(f"{'='*60}")

            for model_name, stats in all_stats.items():
                print(f"\n{model_name}:")
                print(
                    f"  Seq LogProb Diff (mean): {stats['sequence_logprob_diff']['mean']:.6f}"
                )
                print(f"  KL Symmetric (mean):     {stats['kl_symmetric']['mean']:.6f}")

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
