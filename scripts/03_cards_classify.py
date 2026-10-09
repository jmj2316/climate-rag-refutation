"""
CARDS-taxonomy zero-shot classification.

--validate : run on a stratified sample of ClimateCheck (which has ground-truth
             CARDS super-category labels) and report accuracy / per-class F1.
--apply    : run on all CLIMATE-FEVER claims and save predicted CARDS category.

Usage:
  CUDA_VISIBLE_DEVICES=2 conda run -n mj2 python 03_cards_classify.py --validate
  CUDA_VISIBLE_DEVICES=2 conda run -n mj2 python 03_cards_classify.py --apply
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.rag_templates import CARDS_SYSTEM, build_cards_prompt
from parse_rag_response import parse_cards
from vllm_utils import load_llm, generate_with_retry

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42

ROOT = Path(__file__).parent.parent
CLIMATECHECK_PATH = ROOT.parent / "data/climatecheck_test.json"
CLIMATE_FEVER_PATH = ROOT / "data/climate_fever/claims.json"
VALIDATE_OUT = ROOT / "results/cards_validation.json"
APPLY_OUT = ROOT / "data/climate_fever/claims_with_cards.json"


def cf_super_category(cards_field: str) -> str:
    """ClimateCheck 'cards' field looks like 'its_not_us/2_1' -> '2_not_us'."""
    if cards_field is None:
        return "0_not_denial"
    tag = cards_field.split("/")[-1]  # "2_1"
    super_num = tag.split("_")[0]
    name_map = {
        "0": "0_not_denial", "1": "1_not_happening", "2": "2_not_us",
        "3": "3_not_bad", "4": "4_solutions_wont_work", "5": "5_science_unreliable",
    }
    return name_map.get(super_num, "0_not_denial")


def run_classification(llm, claims_texts: list[str]) -> list[dict]:
    tokenizer = llm.get_tokenizer()
    messages = [[{"role": "system", "content": CARDS_SYSTEM},
                 {"role": "user", "content": build_cards_prompt(c)}] for c in claims_texts]
    return generate_with_retry(llm, tokenizer, messages, lambda t: (parse_cards(t),), seed=SEED, max_tokens=100)


def validate(llm):
    data = json.load(open(CLIMATECHECK_PATH))
    for d in data:
        d["cards_super"] = cf_super_category(d.get("cards"))

    by_cat = {}
    for d in data:
        by_cat.setdefault(d["cards_super"], []).append(d)
    random.seed(SEED)
    sample = []
    for cat, rows in by_cat.items():
        n = min(40, len(rows))
        sample.extend(random.sample(rows, n))
    random.shuffle(sample)
    print(f"Validation sample: {len(sample)} claims, dist={Counter(d['cards_super'] for d in sample)}")

    results = run_classification(llm, [d["claim"] for d in sample])

    correct, total_parsed = 0, 0
    per_class = {}
    records = []
    for d, r in zip(sample, results):
        pred = r["parsed"][0]
        true = d["cards_super"]
        records.append({"claim": d["claim"], "true": true, "pred": pred, "raw": r["raw"]})
        if pred is not None:
            total_parsed += 1
            per_class.setdefault(true, {"tp": 0, "n": 0})
            per_class[true]["n"] += 1
            if pred == true:
                correct += 1
                per_class[true]["tp"] += 1

    acc = correct / max(1, total_parsed)
    print(f"\nParse success: {total_parsed}/{len(sample)}")
    print(f"Accuracy: {acc:.3f}")
    for cat, s in sorted(per_class.items()):
        print(f"  {cat}: {s['tp']}/{s['n']} = {s['tp']/max(1,s['n']):.3f}")

    VALIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATE_OUT, "w") as f:
        json.dump({"accuracy": acc, "n": len(sample), "parsed": total_parsed,
                    "per_class": per_class, "records": records}, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {VALIDATE_OUT}")


def apply(llm):
    claims = json.load(open(CLIMATE_FEVER_PATH))
    print(f"Classifying {len(claims)} CLIMATE-FEVER claims ...")
    results = run_classification(llm, [c["claim"] for c in claims])
    for c, r in zip(claims, results):
        c["cards_category"] = r["parsed"][0]
        c["cards_parse_success"] = r["parse_success"]

    dist = Counter(c["cards_category"] for c in claims)
    print("CARDS category distribution:", dict(dist))

    APPLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(APPLY_OUT, "w") as f:
        json.dump(claims, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {APPLY_OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    if not (args.validate or args.apply):
        p.error("pass --validate and/or --apply")

    llm = load_llm(MODEL, seed=SEED)
    if args.validate:
        validate(llm)
    if args.apply:
        apply(llm)
