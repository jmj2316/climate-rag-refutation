"""
Select a stratified pilot sample from CLIMATE-FEVER (by true_label x CARDS category)
to validate the full 3-arm pipeline before committing to the full 1,535-claim run.

Run: conda activate climate_rag && python 04_select_pilot.py
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
IN_PATH = ROOT / "data/climate_fever/claims_with_cards.json"
OUT_PATH = ROOT / "data/climate_fever/pilot_sample.json"

SEED = 42
TARGET_N = 180  # ~180 claims: broad enough for stable per-arm metrics, cheap enough to iterate on


def main():
    claims = json.load(open(IN_PATH))
    random.seed(SEED)

    # stratify jointly on (true_label, cards_category) where possible; fall back to true_label only
    groups = defaultdict(list)
    for c in claims:
        groups[(c["true_label"], c.get("cards_category"))].append(c)

    label_names = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}
    n_groups = len(groups)
    per_group = max(1, TARGET_N // n_groups)

    sample = []
    for key, rows in groups.items():
        n = min(per_group, len(rows))
        sample.extend(random.sample(rows, n))

    # top up to TARGET_N from the remaining pool if under target
    if len(sample) < TARGET_N:
        chosen_ids = {c["claim_id"] for c in sample}
        remaining = [c for c in claims if c["claim_id"] not in chosen_ids]
        random.shuffle(remaining)
        sample.extend(remaining[: TARGET_N - len(sample)])

    random.shuffle(sample)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"Pilot sample: {len(sample)} claims -> {OUT_PATH}")
    print("true_label dist:", Counter(label_names[c["true_label"]] for c in sample))
    print("cards dist:", Counter(c.get("cards_category") for c in sample))


if __name__ == "__main__":
    main()
