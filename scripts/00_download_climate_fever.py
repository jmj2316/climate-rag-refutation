"""
Download CLIMATE-FEVER claims and flatten to one row per claim.
Run (has internet): conda activate climate_rag && python 00_download_climate_fever.py
"""
import json
from pathlib import Path
from datasets import load_dataset

OUT_PATH = Path(__file__).parent.parent / "data/climate_fever/claims.json"


def main():
    print("Loading tdiggelm/climate_fever ...")
    ds = load_dataset("tdiggelm/climate_fever", split="test")
    print(ds)
    print("columns:", ds.column_names)
    print("sample:", {k: str(v)[:200] for k, v in ds[0].items()})

    claims = []
    for row in ds:
        evidence_texts = [ev["evidence"] for ev in row["evidences"]]
        evidence_labels = [ev["evidence_label"] for ev in row["evidences"]]
        claims.append({
            "claim_id":       f"cf_{row['claim_id']}",
            "claim":          row["claim"],
            "true_label":     row["claim_label"],  # 0 SUPPORTS / 1 REFUTES / 2 NOT_ENOUGH_INFO / 3 DISPUTED
            "evidences":      evidence_texts,
            "evidence_labels": evidence_labels,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(claims, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(claims)} claims -> {OUT_PATH}")

    label_names = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}
    from collections import Counter
    dist = Counter(label_names[c["true_label"]] for c in claims)
    print("label distribution:", dict(dist))


if __name__ == "__main__":
    main()
