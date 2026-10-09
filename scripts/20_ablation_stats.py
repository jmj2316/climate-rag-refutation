"""
Statistics for the ablation ladder (17_ablation.py): accuracy / macro-F1 with bootstrap CIs per
(model, arm) and paired contrasts that isolate context size, decomposition and filtering.

Arms: A, B (original runs) + E3 (raw query, n chunks of E1), E2 (raw query, n chunks of C),
E1 (decomposition, top-2 per sub-claim, no filter), C (original run).

Run: python 20_ablation_stats.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import importlib

st = importlib.import_module("12_stats_tests")
RES, LABELS, CF2V = st.RES, st.LABELS, st.CF2V

SHORT = {"HuggingFaceH4/zephyr-7b-beta": "Zephyr", "Qwen/Qwen2.5-7B-Instruct": "Qwen",
         "mistralai/Mistral-7B-Instruct-v0.3": "Mistral"}
ARMS = ["A_zeroshot", "B_dense_rag", "E3_raw_matchE1", "E2_raw_matchC", "E1_decomp_nofilter", "C_decompose_filtered_rag"]
CONTRASTS = [("E3_raw_matchE1", "B_dense_rag", "context size (B to E3)"),
             ("E2_raw_matchC", "B_dense_rag", "context size (B to E2)"),
             ("E1_decomp_nofilter", "E3_raw_matchE1", "decomposition"),
             ("C_decompose_filtered_rag", "E1_decomp_nofilter", "filter"),
             ("C_decompose_filtered_rag", "E2_raw_matchC", "decomp.+filter vs raw"),
             ("C_decompose_filtered_rag", "B_dense_rag", "C vs B")]
B = 10000


def load_all():
    df = st.load_preds()
    rows = []
    for p in sorted((RES / "ablation").glob("*/*_full.jsonl")):
        for line in open(p):
            r = json.loads(line)
            rows.append({k: r[k] for k in ["claim_id", "model", "arm", "true_label", "verdict", "cards_category"]})
    ab = pd.DataFrame(rows)
    ab["y"] = ab["true_label"].map(CF2V).map({v: i for i, v in enumerate(LABELS)})
    ab["p"] = ab["verdict"].map({v: i for i, v in enumerate(LABELS)}).fillna(3).astype(int)
    return pd.concat([df, ab], ignore_index=True)


def main():
    df = load_all()
    models = [m for m in SHORT if m in set(df.model)]
    ids = sorted(df.claim_id.unique(), key=lambda s: int(s.split("_")[1]))
    n = len(ids)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(B, n))
    wide = {}
    for (m, a), g in df.groupby(["model", "arm"]):
        if a in ARMS and len(g) == n:
            g = g.set_index("claim_id").loc[ids]
            wide[(m, a)] = (g.y.to_numpy(), g.p.to_numpy())

    rows = []
    for m in models:
        for a in ARMS:
            if (m, a) not in wide:
                continue
            y, p = wide[(m, a)]
            f1, acc = st.point(y, p)
            bf1, bacc = st.boot_stats(y, p, idx)
            rows.append({"model": m, "arm": a, "accuracy": acc, "acc_lo": np.percentile(bacc, 2.5),
                         "acc_hi": np.percentile(bacc, 97.5), "macro_f1": f1, "f1_lo": np.percentile(bf1, 2.5),
                         "f1_hi": np.percentile(bf1, 97.5), "unparsed": float((p == 3).mean()),
                         "frac_nei": float((p == 2).mean()), "frac_supports": float((p == 0).mean()),
                         "frac_refutes": float((p == 1).mean())})
    ci = pd.DataFrame(rows)
    ci.to_csv(RES / "ablation_ci.csv", index=False)
    print(ci.assign(model=ci.model.map(SHORT)).round(3).to_string(index=False))

    prow = []
    for m in models:
        for x, z, name in CONTRASTS:
            if (m, x) not in wide or (m, z) not in wide:
                continue
            yx, px = wide[(m, x)]
            yz, pz = wide[(m, z)]
            fx, ax = st.point(yx, px)
            fz, az = st.point(yz, pz)
            bfx, _ = st.boot_stats(yx, px, idx)
            bfz, _ = st.boot_stats(yz, pz, idx)
            d = bfx - bfz
            p_boot = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / B)
            n10, n01, p_mc = st.mcnemar_exact(yx == px, yz == pz)
            prow.append({"model": m, "contrast": name, "x": x, "z": z, "delta_f1": fx - fz,
                         "d_lo": np.percentile(d, 2.5), "d_hi": np.percentile(d, 97.5), "p_boot_f1": p_boot,
                         "delta_acc": ax - az, "x_only": n10, "z_only": n01, "p_mcnemar": p_mc})
    pr = pd.DataFrame(prow)
    pr["p_boot_f1_holm"] = st.holm(pr.p_boot_f1.to_numpy())
    pr["p_mcnemar_holm"] = st.holm(pr.p_mcnemar.to_numpy())
    pr.to_csv(RES / "ablation_paired.csv", index=False)
    print(pr.assign(model=pr.model.map(SHORT)).drop(columns=["x", "z"]).round(4).to_string(index=False))

    # mean across the three models (model-averaged paired delta, bootstrap over claims)
    avg = []
    for x, z, name in CONTRASTS:
        ds = []
        for m in models:
            if (m, x) in wide and (m, z) in wide:
                yx, px = wide[(m, x)]
                yz, pz = wide[(m, z)]
                bfx, _ = st.boot_stats(yx, px, idx)
                bfz, _ = st.boot_stats(yz, pz, idx)
                ds.append((st.point(yx, px)[0] - st.point(yz, pz)[0], bfx - bfz))
        if len(ds) == len(models):
            pt = np.mean([d[0] for d in ds])
            bs = np.mean([d[1] for d in ds], axis=0)
            avg.append({"contrast": name, "delta_f1_mean": pt, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
    av = pd.DataFrame(avg)
    av.to_csv(RES / "ablation_paired_modelavg.csv", index=False)
    print(av.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
