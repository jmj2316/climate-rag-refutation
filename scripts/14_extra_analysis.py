"""
Supplementary analyses for the paper:
  (a) per-CARDS-category paired accuracy deltas between pipelines (model-averaged, bootstrap over claims)
  (b) refutation length / hedging-language rates and their relation to judged logical validity
  (c) how many refutations were recovered by the label-less parsing fallback, per model x pipeline

Run: conda run -n climate_rag python 14_extra_analysis.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
CF2V = {0: "supports", 1: "refutes", 2: "nei", 3: "nei"}
ARMS = ["A_zeroshot", "B_dense_rag", "C_decompose_filtered_rag"]
HEDGE = re.compile(r"excerpt|provided evidence|evidence provided|sub-claim|not directly|does not provide|"
                   r"do not provide|no relevant|not enough information|insufficient|does not address|"
                   r"do not address|does not mention|not mentioned", re.IGNORECASE)
B = 10000
rng = np.random.default_rng(1)


def load(dirpath):
    rows = []
    for d in ["model_a", "model_b", "model_c"]:
        for p in sorted((dirpath / d).glob("*_full.jsonl")):
            for line in open(p):
                r = json.loads(line)
                rows.append({"claim_id": r["claim_id"], "model": r["model"], "arm": r["arm"],
                             "cards": r["cards_category"], "true": CF2V[r["true_label"]],
                             "verdict": r["verdict"], "refutation": r.get("refutation") or ""})
    df = pd.DataFrame(rows)
    df["correct"] = (df["true"] == df["verdict"]).astype(int)
    return df


def main():
    cur = load(RES)
    old = load(RES / "_backup_preparse")

    # (a) per-category pooled deltas
    acc = cur.pivot_table(index=["claim_id", "cards"], columns=["model", "arm"], values="correct").reset_index()
    models = sorted(cur["model"].unique())
    out = []
    for cat, g in acc.groupby("cards"):
        n = len(g)
        arm_acc = {a: np.mean([g[(m, a)].to_numpy() for m in models], axis=0) for a in ARMS}  # per-claim, model-averaged
        idx = rng.integers(0, n, size=(B, n))
        row = {"cards": cat, "n_claims": n}
        for a in ARMS:
            row[f"acc_{a[0]}"] = arm_acc[a].mean()
        for x, z in [("C_decompose_filtered_rag", "A_zeroshot"), ("C_decompose_filtered_rag", "B_dense_rag"),
                     ("B_dense_rag", "A_zeroshot")]:
            d = arm_acc[x] - arm_acc[z]
            bd = d[idx].mean(1)
            row[f"d_{x[0]}{z[0]}"] = d.mean()
            row[f"lo_{x[0]}{z[0]}"] = np.percentile(bd, 2.5)
            row[f"hi_{x[0]}{z[0]}"] = np.percentile(bd, 97.5)
        out.append(row)
    cat_df = pd.DataFrame(out)
    cat_df.to_csv(RES / "stats_cards_deltas_full.csv", index=False)
    print("=== per-CARDS deltas (model-averaged accuracy) ===")
    print(cat_df.round(3).to_string(index=False))

    # (b) length & hedging
    judge = pd.read_json(RES / "llm_judge_scores_full.jsonl", lines=True)[["claim_id", "model", "arm", "logical_validity", "scientific_rigor"]]
    cur["words"] = cur["refutation"].str.split().str.len()
    cur["hedge"] = cur["refutation"].apply(lambda t: bool(HEDGE.search(t)))
    m = cur.merge(judge, on=["claim_id", "model", "arm"], how="left")
    summ = m.groupby(["arm", "model"]).agg(words=("words", "mean"), hedge_rate=("hedge", "mean"),
                                         lv=("logical_validity", "mean")).reset_index()
    print("\n=== refutation length / hedging ===")
    print(summ.round(3).to_string(index=False))
    summ.to_csv(RES / "stats_length_hedge_full.csv", index=False)
    print("\n=== LV by hedging (within arm, pooled over models) ===")
    print(m.groupby(["arm", "hedge"])["logical_validity"].agg(["mean", "count"]).round(3))
    for a in ARMS:
        s = m[(m.arm == a) & m.logical_validity.notna() & m.words.notna()]
        rho, p = spearmanr(s["words"], s["logical_validity"])
        print(f"Spearman(words, LV) {a}: rho={rho:.3f} p={p:.2g}")

    # (c) fallback usage
    key = ["claim_id", "model", "arm"]
    mm = cur.merge(old[key + ["refutation"]], on=key, suffixes=("", "_old"))
    mm["recovered"] = (mm["refutation"] != "") & (mm["refutation_old"] == "")
    rec = mm.groupby(["arm", "model"])["recovered"].sum().reset_index()
    print("\n=== refutations recovered by label-less fallback (of 1535) ===")
    print(rec.to_string(index=False))
    rec.to_csv(RES / "stats_fallback_recovered_full.csv", index=False)


if __name__ == "__main__":
    main()


def verdict_distribution():
    cur = load(RES)
    cur["verdict"] = cur["verdict"].fillna("unparsed")
    t = cur.groupby(["arm", "model"])["verdict"].value_counts(normalize=True).unstack(fill_value=0).round(3)
    print("\n=== predicted-verdict distribution (truth: supports .426 / refutes .165 / nei .409) ===")
    print(t.to_string())
    t.to_csv(RES / "stats_verdict_distribution_full.csv")


if __name__ == "__main__":
    verdict_distribution()
