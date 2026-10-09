"""
Statistical rigor for the full benchmark: bootstrap CIs, majority/chance baselines,
paired arm comparisons (paired bootstrap on Macro-F1, exact McNemar on accuracy,
Wilcoxon on judge/RAGAS scores), Holm correction.

Unparsed verdicts are scored as errors (class "none"), so numbers differ slightly
from 08_eval_classification.py, which drops unparsed rows.

Run: conda run -n climate_rag python 12_stats_tests.py
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
CF2V = {0: "supports", 1: "refutes", 2: "nei", 3: "nei"}
LABELS = ["supports", "refutes", "nei"]
ARM_ORDER = ["A_zeroshot", "B_dense_rag", "C_decompose_filtered_rag"]
B = 10000
rng = np.random.default_rng(0)


def load_preds():
    rows = []
    for d in ["model_a", "model_b", "model_c"]:
        for p in (RES / d).glob("*_full.jsonl"):
            for line in open(p):
                r = json.loads(line)
                rows.append({k: r[k] for k in ["claim_id", "model", "arm", "true_label", "verdict", "cards_category"]})
    df = pd.DataFrame(rows)
    df["y"] = df["true_label"].map(CF2V).map({v: i for i, v in enumerate(LABELS)})
    df["p"] = df["verdict"].map({v: i for i, v in enumerate(LABELS)}).fillna(3).astype(int)
    return df


def f1_parts(y, p):
    """per-claim tp/fp/fn indicator arrays, shape (3, n)"""
    tp = np.stack([(y == k) & (p == k) for k in range(3)]).astype(np.int32)
    fp = np.stack([(y != k) & (p == k) for k in range(3)]).astype(np.int32)
    fn = np.stack([(y == k) & (p != k) for k in range(3)]).astype(np.int32)
    return tp, fp, fn


def macro_f1_from_sums(tp, fp, fn):
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean(axis=0)


def boot_stats(y, p, idx):
    tp, fp, fn = f1_parts(y, p)
    f1 = macro_f1_from_sums(tp[:, idx].sum(-1) if False else np.stack([t[idx].sum(1) for t in tp]),
                            np.stack([t[idx].sum(1) for t in fp]),
                            np.stack([t[idx].sum(1) for t in fn]))
    acc = (y == p)[idx].mean(1)
    return f1, acc


def point(y, p):
    tp, fp, fn = f1_parts(y, p)
    return macro_f1_from_sums(tp.sum(1, keepdims=True), fp.sum(1, keepdims=True), fn.sum(1, keepdims=True))[0], (y == p).mean()


def mcnemar_exact(a_correct, b_correct):
    n10 = int((a_correct & ~b_correct).sum())
    n01 = int((~a_correct & b_correct).sum())
    if n10 + n01 == 0:
        return n10, n01, 1.0
    return n10, n01, binomtest(n10, n10 + n01, 0.5).pvalue


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    df = load_preds()
    models = sorted(df["model"].unique())
    claim_ids = sorted(df["claim_id"].unique(), key=lambda s: int(s.split("_")[1]))
    n = len(claim_ids)
    idx = rng.integers(0, n, size=(B, n))

    wide = {}  # (model, arm) -> (y, p) aligned on claim_ids
    for (m, a), g in df.groupby(["model", "arm"]):
        g = g.set_index("claim_id").loc[claim_ids]
        wide[(m, a)] = (g["y"].to_numpy(), g["p"].to_numpy())
    y_any = next(iter(wide.values()))[0]

    # ── baselines ──
    cls_freq = np.bincount(y_any, minlength=3) / n
    maj = int(cls_freq.argmax())
    maj_p = np.full(n, maj)
    maj_f1, maj_acc = point(y_any, maj_p)
    base = {"majority_class": LABELS[maj], "class_freq": dict(zip(LABELS, cls_freq.round(4))),
            "majority_acc": maj_acc, "majority_macro_f1": maj_f1, "uniform_chance_acc": 1 / 3}
    print("Baselines:", base)

    # ── per (model, arm) with CI ──
    rows = []
    for m in models:
        for a in ARM_ORDER:
            y, p = wide[(m, a)]
            f1, acc = point(y, p)
            bf1, bacc = boot_stats(y, p, idx)
            n_corr = int((y == p).sum())
            rows.append({
                "model": m, "arm": a, "macro_f1": f1,
                "f1_lo": np.percentile(bf1, 2.5), "f1_hi": np.percentile(bf1, 97.5),
                "accuracy": acc, "acc_lo": np.percentile(bacc, 2.5), "acc_hi": np.percentile(bacc, 97.5),
                "acc_vs_majority_p": binomtest(n_corr, n, maj_acc).pvalue,
                "acc_minus_majority": acc - maj_acc,
                "f1_minus_majority": f1 - maj_f1,
            })
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(RES / "stats_ci_full.csv", index=False)
    print("\n=== Point estimates with 95% bootstrap CI ===")
    print(ci_df.round(3).to_string(index=False))

    # ── paired comparisons ──
    comp_rows = []
    pairs = [("B_dense_rag", "A_zeroshot"), ("C_decompose_filtered_rag", "B_dense_rag"),
             ("C_decompose_filtered_rag", "A_zeroshot")]
    for m in models:
        for x, z in pairs:
            yx, px = wide[(m, x)]
            yz, pz = wide[(m, z)]
            fx, ax = point(yx, px)
            fz, az = point(yz, pz)
            bfx, _ = boot_stats(yx, px, idx)
            bfz, _ = boot_stats(yz, pz, idx)
            d = bfx - bfz
            p_boot = 2 * min((d <= 0).mean(), (d >= 0).mean())
            n10, n01, p_mc = mcnemar_exact(yx == px, yz == pz)
            comp_rows.append({
                "model": m, "contrast": f"{x} - {z}", "delta_f1": fx - fz,
                "d_lo": np.percentile(d, 2.5), "d_hi": np.percentile(d, 97.5),
                "p_boot_f1": max(p_boot, 1.0 / B), "delta_acc": ax - az,
                "x_only_correct": n10, "z_only_correct": n01, "p_mcnemar": p_mc,
            })
    comp_df = pd.DataFrame(comp_rows)
    comp_df["p_boot_f1_holm"] = holm(comp_df["p_boot_f1"].to_numpy())
    comp_df["p_mcnemar_holm"] = holm(comp_df["p_mcnemar"].to_numpy())
    comp_df.to_csv(RES / "stats_paired_full.csv", index=False)
    print("\n=== Paired comparisons (Holm over all 9) ===")
    print(comp_df.round(4).to_string(index=False))

    # ── judge / RAGAS paired Wilcoxon ──
    judge = pd.read_json(RES / "llm_judge_scores_full.jsonl", lines=True)
    ragas = pd.read_json(RES / "ragas_lite_scores_full.jsonl", lines=True)
    score_rows = []
    specs = [(judge, ["logical_validity", "scientific_rigor"], pairs),
             (ragas, ["faithfulness", "answer_relevance", "context_precision"],
              [("C_decompose_filtered_rag", "B_dense_rag")])]
    for src, metrics, prs in specs:
        for m in models:
            for x, z in prs:
                for met in metrics:
                    px = src[(src.model == m) & (src.arm == x)].set_index("claim_id")[met]
                    pz = src[(src.model == m) & (src.arm == z)].set_index("claim_id")[met]
                    j = px.to_frame("x").join(pz.to_frame("z"), how="inner").dropna()
                    diff = j["x"] - j["z"]
                    if (diff == 0).all():
                        pv = 1.0
                    else:
                        pv = wilcoxon(j["x"], j["z"], zero_method="wilcox").pvalue
                    score_rows.append({"model": m, "metric": met, "contrast": f"{x} - {z}", "n_pairs": len(j),
                                       "mean_x": j["x"].mean(), "mean_z": j["z"].mean(),
                                       "mean_diff": diff.mean(), "p_wilcoxon": pv})
    sc_df = pd.DataFrame(score_rows)
    sc_df["p_holm"] = holm(sc_df["p_wilcoxon"].to_numpy())
    sc_df.to_csv(RES / "stats_scores_paired_full.csv", index=False)
    print("\n=== Judge / RAGAS paired tests (Holm over all) ===")
    print(sc_df.round(4).to_string(index=False))

    json.dump({k: (v if not isinstance(v, np.floating) else float(v)) for k, v in base.items()},
              open(RES / "stats_baselines_full.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
