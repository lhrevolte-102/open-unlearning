#!/usr/bin/env python

import csv
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "saves" / "unlearn" / "infocurl_grid_manifest.csv"


def add_row(rows, family, run_tag, **kwargs):
    row = {
        "family": family,
        "run_tag": run_tag,
        "trainer_name": kwargs.get("trainer_name", "InfoCURL_NPO"),
        "mode": kwargs.get("mode", ""),
        "score_gamma": kwargs.get("score_gamma", ""),
        "lam": kwargs.get("lam", ""),
        "score_subpool": kwargs.get("score_subpool", ""),
        "K_steps": kwargs.get("K_steps", ""),
        "param_scope": kwargs.get("param_scope", ""),
        "retain_batch_size": kwargs.get("retain_batch_size", ""),
        "retain_ema_decay": kwargs.get("retain_ema_decay", ""),
        "stage_schedule": kwargs.get("stage_schedule", ""),
        "stage1_mode": kwargs.get("stage1_mode", ""),
        "stage2_mode": kwargs.get("stage2_mode", ""),
        "stage1_gamma": kwargs.get("stage1_gamma", ""),
        "stage2_gamma": kwargs.get("stage2_gamma", ""),
        "switch_frac": kwargs.get("switch_frac", ""),
        "transition_frac": kwargs.get("transition_frac", ""),
        "train_batch_size": 8,
        "grad_accum": 2,
        "seed": kwargs.get("seed", 0),
        "run_eval": kwargs.get("run_eval", 1),
        "max_steps": kwargs.get("max_steps", ""),
        "notes": kwargs.get("notes", ""),
    }
    rows.append(row)


def main():
    rows = []

    # Search policy:
    # - keep train micro-batch and grad accumulation fixed at the current
    #   stable setting (8 x 2)
    # - only search InfoCURL-specific mechanism parameters and schedule forms

    # Baseline reference row for convenience. Existing runs will be skipped by the worker.
    add_row(
        rows,
        family="baseline",
        run_tag="tofu_Llama-3.2-3B-Instruct_forget10_NPO_bs8ga2_full_s0",
        trainer_name="NPO",
        notes="reference original NPO",
    )

    # Fixed easy / hard schedules
    for mode, gamma, k_steps, score_subpool in product(
        ["easy", "hard"],
        [0.08, 0.10, 0.12, 0.14, 0.16],
        [10, 20, 30],
        [32, 64],
    ):
        add_row(
            rows,
            family="fixed",
            run_tag=(
                f"tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_"
                f"{mode}_g{gamma:.2f}_k{k_steps}_sp{score_subpool}_bs8ga2_s0"
            ).replace(".", "p"),
            mode=mode,
            score_gamma=gamma,
            K_steps=k_steps,
            score_subpool=score_subpool,
            param_scope="last_layer_lm_head",
            notes="fixed schedule sweep",
        )

    # Hard schedules with retain-aware penalty.
    for gamma, lam, k_steps in product(
        [0.08, 0.10, 0.12, 0.14],
        [0.02, 0.05, 0.10],
        [20],
    ):
        add_row(
            rows,
            family="hard_retain",
            run_tag=(
                f"tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_"
                f"hard_g{gamma:.2f}_l{lam:.2f}_k{k_steps}_bs8ga2_s0"
            ).replace(".", "p"),
            mode="hard",
            score_gamma=gamma,
            lam=lam,
            K_steps=k_steps,
            score_subpool=64,
            param_scope="last_layer_lm_head",
            retain_batch_size=8,
            retain_ema_decay=0.9,
            notes="retain-aware hard schedule",
        )

    # Hard switch easy -> hard.
    for stage2_gamma, switch_frac in product(
        [0.08, 0.10, 0.12, 0.14],
        [0.20, 0.35, 0.50],
    ):
        add_row(
            rows,
            family="switch_e2h",
            run_tag=(
                f"tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_"
                f"e2h_sw{switch_frac:.2f}_g1to{stage2_gamma:.2f}_k20_bs8ga2_s0"
            ).replace(".", "p"),
            mode="easy",
            score_gamma=1.0,
            K_steps=20,
            score_subpool=64,
            param_scope="last_layer_lm_head",
            stage_schedule="easy_to_hard",
            stage1_mode="easy",
            stage2_mode="hard",
            stage1_gamma=1.0,
            stage2_gamma=stage2_gamma,
            switch_frac=switch_frac,
            notes="hard switch easy->hard",
        )

    # Soft easy -> hard.
    for stage2_gamma, switch_frac, transition_frac in product(
        [0.08, 0.10, 0.12, 0.14],
        [0.20, 0.30],
        [0.20, 0.30, 0.40],
    ):
        add_row(
            rows,
            family="soft_e2h",
            run_tag=(
                f"tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_"
                f"soft_sw{switch_frac:.2f}_tr{transition_frac:.2f}_"
                f"g1to{stage2_gamma:.2f}_k20_bs8ga2_s0"
            ).replace(".", "p"),
            mode="easy",
            score_gamma=1.0,
            K_steps=20,
            score_subpool=64,
            param_scope="last_layer_lm_head",
            stage_schedule="soft_easy_to_hard",
            stage1_mode="easy",
            stage2_mode="hard",
            stage1_gamma=1.0,
            stage2_gamma=stage2_gamma,
            switch_frac=switch_frac,
            transition_frac=transition_frac,
            notes="soft easy->hard schedule",
        )

    # Soft schedule with retain-aware penalty.
    for stage2_gamma, lam, switch_frac, transition_frac in product(
        [0.10, 0.14],
        [0.02, 0.05],
        [0.20, 0.30],
        [0.30, 0.40],
    ):
        add_row(
            rows,
            family="soft_retain",
            run_tag=(
                f"tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_"
                f"soft_l{lam:.2f}_sw{switch_frac:.2f}_tr{transition_frac:.2f}_"
                f"g1to{stage2_gamma:.2f}_k20_bs8ga2_s0"
            ).replace(".", "p"),
            mode="easy",
            score_gamma=1.0,
            lam=lam,
            K_steps=20,
            score_subpool=64,
            param_scope="last_layer_lm_head",
            retain_batch_size=8,
            retain_ema_decay=0.9,
            stage_schedule="soft_easy_to_hard",
            stage1_mode="easy",
            stage2_mode="hard",
            stage1_gamma=1.0,
            stage2_gamma=stage2_gamma,
            switch_frac=switch_frac,
            transition_frac=transition_frac,
            notes="soft schedule with retain-aware penalty",
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(OUT)
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
