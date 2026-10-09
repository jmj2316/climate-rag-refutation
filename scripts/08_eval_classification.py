"""
Classification evaluation: Macro F1 (3-way: supports/refutes/nei) per (model, arm),
plus per-CARDS-category accuracy breakdown (error analysis table).

Run: conda activate climate_rag && python 08_eval_classification.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
ARMS = ["model_a", "model_b", "model_c"]
CF_LABEL_TO_VERDICT = {0: "supports", 1: "refutes", 2: "nei", 3: "nei"}


def load_all(split: str) -> pd.DataFrame:
    rows = []
    for arm_dir in ARMS:
        d = RESULTS_DIR / arm_dir
        if not d.exists():
            continue
        for path in d.glob(f"*_{split}.jsonl"):
            for line in open(path):
                r = json.loads(line)
                rows.append(r)
    df = pd.DataFrame(rows)
    df["true_verdict"] = df["true_label"].map(CF_LABEL_TO_VERDICT)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="full", choices=["pilot", "full"])
    args = p.parse_args()
    df = load_all(args.split)
    print(f"Loaded {len(df)} predictions across {df['model'].nunique()} models x {df['arm'].nunique()} arms")

    # ── overall metrics per (model, arm) ─────────────────────────────────────
    summary = []
    for (model, arm), g in df.groupby(["model", "arm"]):
        parsed = g[g["verdict"].notna()]
        y_true = parsed["true_verdict"]
        y_pred = parsed["verdict"]
        labels = ["supports", "refutes", "nei"]
        summary.append({
            "model": model, "arm": arm,
            "n": len(g), "n_parsed": len(parsed),
            "parse_rate": len(parsed) / len(g),
            "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "accuracy": (y_true == y_pred).mean(),
        })
    summary_df = pd.DataFrame(summary).sort_values(["arm", "model"])
    print("\n=== Overall (model x arm) ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(RESULTS_DIR / f"classification_summary_{args.split}.csv", index=False)

    # ── per-CARDS-category accuracy (error analysis heatmap table) ──────────
    cards_rows = []
    for (arm, cards_cat), g in df.groupby(["arm", "cards_category"]):
        parsed = g[g["verdict"].notna()]
        if len(parsed) == 0:
            continue
        acc = (parsed["true_verdict"] == parsed["verdict"]).mean()
        cards_rows.append({"arm": arm, "cards_category": cards_cat, "n": len(parsed), "accuracy": acc})
    cards_df = pd.DataFrame(cards_rows)
    pivot = cards_df.pivot(index="cards_category", columns="arm", values="accuracy")
    print("\n=== Accuracy by CARDS category x arm (averaged over models) ===")
    print(pivot.round(3).to_string())
    pivot.to_csv(RESULTS_DIR / f"cards_error_heatmap_{args.split}.csv")

    print(f"\nSaved -> {RESULTS_DIR / f'classification_summary_{args.split}.csv'}")
    print(f"Saved -> {RESULTS_DIR / f'cards_error_heatmap_{args.split}.csv'}")


if __name__ == "__main__":
    main()
