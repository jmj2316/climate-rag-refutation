"""
Quality proxy for the decomposition step: how often do a claim's sub-claims introduce numbers
(years, quantities, citation page numbers) that are absent from the claim itself, and how many are
more than twice the claim length. Reads results/model_c/*_full.jsonl.

Run: python 21_decomp_quality.py
"""
import json
import re
from pathlib import Path

import pandas as pd

RES = Path(__file__).parent.parent / "results"
NUM = re.compile(r"\d[\d.,]*")
rows = []
for f, name in [("zephyr-7b-beta", "Zephyr"), ("Qwen2_5-7B-Instruct", "Qwen"), ("Mistral-7B-Instruct-v0_3", "Mistral")]:
    rs = [json.loads(l) for l in open(RES / "model_c" / f"{f}_full.jsonl")]
    inv = sum(any(set(NUM.findall(s)) - set(NUM.findall(r["claim"])) for s in r["subclaims"]) for r in rs)
    longer = sum(any(len(s.split()) > 2 * len(r["claim"].split()) + 10 for s in r["subclaims"]) for r in rs)
    words = sum(len(s.split()) for r in rs for s in r["subclaims"]) / sum(len(r["subclaims"]) for r in rs)
    rows.append({"model": name, "n_claims": len(rs), "share_claims_with_new_numbers": inv / len(rs),
                 "share_claims_with_overlong_subclaim": longer / len(rs), "mean_subclaim_words": words,
                 "mean_claim_words": sum(len(r["claim"].split()) for r in rs) / len(rs),
                 "mean_n_subclaims": sum(len(r["subclaims"]) for r in rs) / len(rs)})
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
df.to_csv(RES / "decomp_quality_full.csv", index=False)
