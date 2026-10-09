"""
Does the label-less-refutation fallback drive the judge / hedging / length findings?

1. Classify every output: labeled ("Refutation:" present) / fallback prose (label missing, plain
   explanation text) / fallback structured (label missing, text echoes "Sub-claim:" / "Excerpt" scaffolding).
2. Multi-verdict lines ("Supports (for ...), Refutes (for ...)").
3. Hedge rate with and without the "sub-claim" token.
4. Paired judge-LV contrasts restricted to claims whose outputs are clean (labeled or fallback prose)
   in both pipelines being compared.

Run: conda run -n climate_rag python 19_format_robustness.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
SHORT = {"HuggingFaceH4/zephyr-7b-beta": "Zephyr", "Qwen/Qwen2.5-7B-Instruct": "Qwen",
         "mistralai/Mistral-7B-Instruct-v0.3": "Mistral"}
HEDGE = re.compile(r"excerpt|provided evidence|evidence provided|sub-claim|not directly|does not provide|"
                   r"do not provide|no relevant|not enough information|insufficient|does not address|"
                   r"do not address|does not mention|not mentioned", re.IGNORECASE)
HEDGE_NO_SUB = re.compile(HEDGE.pattern.replace("sub-claim|", ""), re.IGNORECASE)
SCAFFOLD = re.compile(r"sub-?claim\s*:|^\s*excerpt\s*[\[:]", re.IGNORECASE | re.MULTILINE)
VERDICT_WORDS = re.compile(r"supports?|refutes?|not enough info", re.IGNORECASE)
B = 10000
rng = np.random.default_rng(7)


def fmt(raw):
    labeled = bool(re.search(r"Refutation\s*:", raw, re.IGNORECASE))
    if labeled:
        return "labeled"
    return "fallback_structured" if SCAFFOLD.search(raw) else "fallback_prose"


def multi(raw):
    m = re.search(r"Verdict\s*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if not m:
        return False
    return len({w.lower().rstrip("s") for w in VERDICT_WORDS.findall(m.group(1))}) > 1


rows = []
for d in ("model_a", "model_b", "model_c"):
    for p in sorted((RES / d).glob("*_full.jsonl")):
        for line in open(p):
            r = json.loads(line)
            rows.append({"claim_id": r["claim_id"], "model": SHORT[r["model"]], "arm": r["arm"][0],
                         "fmt": fmt(r["raw_response"]), "multi": multi(r["raw_response"]),
                         "ref": r.get("refutation") or "", "has_verdict": r["verdict"] is not None})
df = pd.DataFrame(rows)
judge = pd.read_json(RES / "llm_judge_scores_full.jsonl", lines=True)
judge["model"] = judge["model"].map(SHORT)
judge["arm"] = judge["arm"].str[0]
df = df.merge(judge[["claim_id", "model", "arm", "logical_validity"]], on=["claim_id", "model", "arm"], how="left")
df["words"] = df["ref"].str.split().str.len()
df["hedge"] = df["ref"].apply(lambda t: bool(HEDGE.search(t)))
df["hedge_nosub"] = df["ref"].apply(lambda t: bool(HEDGE_NO_SUB.search(t)))
df["clean"] = df["fmt"] != "fallback_structured"

print("=== output format counts (of 1535) ===")
fc = df.groupby(["model", "arm"])["fmt"].value_counts().unstack(fill_value=0)
fc["multi_verdict"] = df.groupby(["model", "arm"])["multi"].sum()
fc["clean"] = df.groupby(["model", "arm"])["clean"].sum()
print(fc.to_string())
fc.to_csv(RES / "format_counts_full.csv")

print("\n=== hedge rate with / without the 'sub-claim' token ===")
h = df.groupby(["model", "arm"]).agg(hedge=("hedge", "mean"), hedge_nosub=("hedge_nosub", "mean")).round(3)
print(h.to_string())
h.to_csv(RES / "hedge_nosub_full.csv")

print("\n=== mean LV / hedge / words by format ===")
bf = df.groupby(["arm", "fmt"]).agg(n=("claim_id", "size"), lv=("logical_validity", "mean"),
                                    hedge=("hedge", "mean"), words=("words", "mean")).round(3)
print(bf.to_string())

print("\n=== paired LV contrasts on claims clean in both arms ===")
out = []
for model in SHORT.values():
    s = df[df.model == model].pivot(index="claim_id", columns="arm", values=["logical_validity", "clean", "hedge_nosub"])
    for x, y in [("B", "A"), ("C", "A"), ("C", "B")]:
        ok = s["clean"][x].astype(bool) & s["clean"][y].astype(bool) & s["logical_validity"][x].notna() & s["logical_validity"][y].notna()
        d = (s["logical_validity"][x] - s["logical_validity"][y])[ok].to_numpy()
        bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
        out.append({"model": model, "contrast": f"{x}-{y}", "n_clean_pairs": int(ok.sum()), "delta_lv": d.mean(),
                    "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                    "hedge_x": s["hedge_nosub"][x][ok].mean(), "hedge_y": s["hedge_nosub"][y][ok].mean()})
o = pd.DataFrame(out)
print(o.round(3).to_string(index=False))
o.to_csv(RES / "lv_clean_subset_full.csv", index=False)

print("\n=== within-arm LV by hedging (no 'sub-claim' token), all outputs vs clean outputs ===")
from scipy.stats import spearmanr
for tag, sub in [("all", df), ("clean", df[df.clean])]:
    for a in "ABC":
        s = sub[(sub.arm == a) & sub.logical_validity.notna()]
        h1 = s[s.hedge_nosub].logical_validity.mean()
        h0 = s[~s.hedge_nosub].logical_validity.mean()
        rho, _ = spearmanr(s.words, s.logical_validity)
        print(f"{tag:5s} {a}: hedged LV {h1:.2f} (n={s.hedge_nosub.sum()}) vs unhedged {h0:.2f} (n={(~s.hedge_nosub).sum()}); "
              f"Spearman(words, LV)={rho:.2f}")

print("\n=== mean words by arm and model (all / clean) ===")
print(pd.concat([df.groupby(["model", "arm"]).words.mean().round(0).rename("all"),
                 df[df.clean].groupby(["model", "arm"]).words.mean().round(0).rename("clean")], axis=1).to_string())
