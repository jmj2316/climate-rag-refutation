"""
Aggregate classification, RAGAS-lite, and LLM-judge scores into one master table
per (model, arm): the main results table for the benchmark.

Run: conda activate climate_rag && python 11_aggregate_results.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in open(path)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="full", choices=["pilot", "full"])
    args = p.parse_args()

    cls_path = RESULTS_DIR / f"classification_summary_{args.split}.csv"
    cls = pd.read_csv(cls_path) if cls_path.exists() else pd.DataFrame()

    ragas = load_jsonl(RESULTS_DIR / f"ragas_lite_scores_{args.split}.jsonl")
    ragas_agg = (ragas.groupby(["model", "arm"])[["faithfulness", "answer_relevance", "context_precision"]]
                 .mean().reset_index()) if not ragas.empty else pd.DataFrame()

    judge = load_jsonl(RESULTS_DIR / f"llm_judge_scores_{args.split}.jsonl")
    judge_agg = (judge.groupby(["model", "arm"])[["logical_validity", "scientific_rigor"]]
                 .mean().reset_index()) if not judge.empty else pd.DataFrame()

    master = cls
    for extra in (ragas_agg, judge_agg):
        if not extra.empty:
            master = master.merge(extra, on=["model", "arm"], how="left")

    arm_names = {"A_zeroshot": "A: Zero-shot", "B_dense_rag": "B: Dense RAG",
                 "C_decompose_filtered_rag": "C: Decompose+Filtered RAG"}
    master["arm_name"] = master["arm"].map(arm_names).fillna(master["arm"])
    cols = ["arm_name", "model", "n", "parse_rate", "accuracy", "macro_f1",
            "faithfulness", "answer_relevance", "context_precision",
            "logical_validity", "scientific_rigor"]
    cols = [c for c in cols if c in master.columns]
    master = master[cols].sort_values(["arm_name", "model"])

    print(master.round(3).to_string(index=False))
    out_path = RESULTS_DIR / f"master_results_table_{args.split}.csv"
    master.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
