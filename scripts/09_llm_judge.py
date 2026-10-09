"""
LLM-as-a-judge (G-Eval style): score every generated refutation on Logical Validity
and Scientific Rigor (1-5 each). Uses an independent judge model (Llama-3.1-8B,
NOT one of the 3 models under test) to avoid self-evaluation bias.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n mj2 python 09_llm_judge.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.rag_templates import JUDGE_SYSTEM, JUDGE_USER
from parse_rag_response import parse_judge
from vllm_utils import load_llm, generate_with_retry

JUDGE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
SEED = 42
ROOT = Path(__file__).parent.parent
ARMS = ["model_a", "model_b", "model_c"]


def load_all_records(split: str) -> list[dict]:
    records = []
    for arm_dir in ARMS:
        d = ROOT / "results" / arm_dir
        if not d.exists():
            continue
        for path in d.glob(f"*_{split}.jsonl"):
            for line in open(path):
                r = json.loads(line)
                if r.get("refutation"):
                    records.append(r)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="full", choices=["pilot", "full"])
    p.add_argument("--gpu-mem-util", type=float, default=0.6)
    args = p.parse_args()
    records = load_all_records(args.split)
    print(f"Loaded {len(records)} refutations to judge")

    llm = load_llm(JUDGE_MODEL, seed=SEED, gpu_mem=args.gpu_mem_util)
    tokenizer = llm.get_tokenizer()

    messages = [[{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user", "content": JUDGE_USER.format(claim=r["claim"], refutation=r["refutation"])}]
                for r in records]

    results = generate_with_retry(llm, tokenizer, messages, parse_judge, seed=SEED, max_tokens=150)

    out_path = ROOT / f"results/llm_judge_scores_{args.split}.jsonl"
    n_ok = 0
    with open(out_path, "w") as fout:
        for r, res in zip(records, results):
            logical, rigor = res["parsed"]
            n_ok += int(res["parse_success"])
            fout.write(json.dumps({
                "claim_id": r["claim_id"], "model": r["model"], "arm": r["arm"],
                "cards_category": r.get("cards_category"),
                "logical_validity": logical, "scientific_rigor": rigor,
                "judge_raw": res["raw"], "judge_model": JUDGE_MODEL,
            }, ensure_ascii=False) + "\n")

    print(f"Parse success: {n_ok}/{len(records)} -> {out_path}")


if __name__ == "__main__":
    main()
