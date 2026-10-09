"""
Native reimplementation of RAGAS's 3 core metrics for Model B / Model C outputs:
  - Faithfulness: fraction of refutation statements directly supported by context
  - Answer Relevance: cosine sim(claim, LLM-generated question the refutation answers)
  - Context Precision: rank-weighted precision of retrieved chunks vs the claim

Scored with an independent judge model (Llama-3.1-8B, not one of the 3 models
under test) to avoid self-evaluation bias, run directly through vLLM (see
prompts/ragas_lite_templates.py for why we don't use the `ragas` package here).

Run: CUDA_VISIBLE_DEVICES=0 conda run -n mj2 python 10_eval_ragas_lite.py
"""
import os
os.environ.setdefault("USE_TF", "0")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.ragas_lite_templates import (
    SPLIT_STATEMENTS_SYSTEM, SPLIT_STATEMENTS_USER,
    FAITHFULNESS_SYSTEM, FAITHFULNESS_USER,
    GEN_QUESTION_SYSTEM, GEN_QUESTION_USER,
    CONTEXT_PRECISION_SYSTEM, CONTEXT_PRECISION_USER,
)
from parse_rag_response import parse_subclaims  # reused: same "1. ... 2. ..." list format
from vllm_utils import load_llm, generate_with_retry

import re

JUDGE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
SEED = 42
ROOT = Path(__file__).parent.parent
ARMS_WITH_RETRIEVAL = ["model_b", "model_c"]


def load_records(split: str) -> list[dict]:
    records = []
    for arm_dir in ARMS_WITH_RETRIEVAL:
        d = ROOT / "results" / arm_dir
        for path in d.glob(f"*_{split}.jsonl"):
            for line in open(path):
                r = json.loads(line)
                if not r.get("refutation"):
                    continue
                if arm_dir == "model_b":
                    chunks = r["retrieved_chunks"]
                else:  # model_c: flatten subclaim-grouped chunks, dedup by chunk_id, keep order
                    seen, chunks = set(), []
                    for grp in r["retrieved_chunks"]:
                        for c in grp["chunks"]:
                            if c["chunk_id"] not in seen:
                                seen.add(c["chunk_id"])
                                chunks.append(c)
                r["_flat_chunks"] = chunks
                records.append(r)
    return records


def parse_supported(text: str):
    m = re.search(r"Supported\s*:\s*(\S+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("[]").lower().startswith("y")
    # fallback: model sometimes answers with a bare Yes/No, no "Supported:" label
    s = text.strip().strip("[]").lower()
    if s.startswith("yes"):
        return True
    if s.startswith("no"):
        return False
    return None


def parse_relevant(text: str):
    m = re.search(r"Relevant\s*:\s*(\S+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("[]").lower().startswith("y")
    s = text.strip().strip("[]").lower()
    if s.startswith("yes"):
        return True
    if s.startswith("no"):
        return False
    return None


def parse_question(text: str):
    m = re.search(r"Question\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def context_precision_at_k(relevances: list[bool]) -> float:
    total_relevant = sum(relevances)
    if total_relevant == 0:
        return 0.0
    running_hits, score = 0, 0.0
    for k, rel in enumerate(relevances, 1):
        if rel:
            running_hits += 1
            score += running_hits / k
    return score / total_relevant


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="full", choices=["pilot", "full"])
    p.add_argument("--gpu-mem-util", type=float, default=0.6)
    args = p.parse_args()
    records = load_records(args.split)
    print(f"Loaded {len(records)} model_b/model_c refutations with retrieved context")

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu")  # fail fast; encode on CPU (GPU is shared)

    import pickle
    cache_path = ROOT / f"results/_ragas_cache_{args.split}.pkl"
    cache = pickle.load(open(cache_path, "rb")) if cache_path.exists() else {}

    def save_cache():
        pickle.dump(cache, open(cache_path, "wb"))

    llm = load_llm(JUDGE_MODEL, seed=SEED, max_model_len=4096, gpu_mem=args.gpu_mem_util)
    tokenizer = llm.get_tokenizer()

    def context_block(chunks):
        return "\n\n".join(f"[{i}] {c['text'][:500]}" for i, c in enumerate(chunks, 1))

    # ── Stage 1: split each refutation into statements ───────────────────────
    print("Splitting refutations into statements ...")
    split_msgs = [[{"role": "system", "content": SPLIT_STATEMENTS_SYSTEM},
                    {"role": "user", "content": SPLIT_STATEMENTS_USER.format(refutation=r["refutation"])}]
                   for r in records]
    if "statements" not in cache:
        split_results = generate_with_retry(llm, tokenizer, split_msgs, lambda t: (parse_subclaims(t) or None,),
                                              seed=SEED, max_tokens=250)
        cache["statements"] = [r["parsed"][0] or [rec["refutation"]] for rec, r in zip(records, split_results)]
        save_cache()
    statements_per_record = cache["statements"]

    # ── Stage 2: faithfulness -- judge each statement against its context ────
    print("Judging faithfulness (statement supported by context?) ...")
    faith_msgs, faith_index = [], []
    for i, (rec, stmts) in enumerate(zip(records, statements_per_record)):
        ctx = context_block(rec["_flat_chunks"])
        for s in stmts:
            faith_msgs.append([{"role": "system", "content": FAITHFULNESS_SYSTEM},
                                 {"role": "user", "content": FAITHFULNESS_USER.format(context=ctx, statement=s)}])
            faith_index.append(i)
    if "faith" not in cache:
        faith_results = generate_with_retry(llm, tokenizer, faith_msgs, lambda t: (parse_supported(t),),
                                              seed=SEED, max_tokens=40)
        faith_scores = [[] for _ in records]
        for i, r in zip(faith_index, faith_results):
            faith_scores[i].append(r["parsed"][0])
        cache["faith"] = faith_scores
        save_cache()
    faith_scores = cache["faith"]

    # ── Stage 3: answer relevance -- generate question, embed, cosine sim ────
    print("Generating implied questions for answer relevance ...")
    q_msgs = [[{"role": "system", "content": GEN_QUESTION_SYSTEM},
                {"role": "user", "content": GEN_QUESTION_USER.format(refutation=r["refutation"])}] for r in records]
    if "questions" not in cache:
        q_results = generate_with_retry(llm, tokenizer, q_msgs, lambda t: (parse_question(t),), seed=SEED, max_tokens=80)
        cache["questions"] = [r["parsed"][0] or rec["claim"] for rec, r in zip(records, q_results)]
        save_cache()
    generated_questions = cache["questions"]

    print("Embedding claims + generated questions (bge-m3) ...")
    claim_embs = embed_model.encode([r["claim"] for r in records], normalize_embeddings=True, convert_to_numpy=True)
    q_embs = embed_model.encode(generated_questions, normalize_embeddings=True, convert_to_numpy=True)
    answer_relevance = (claim_embs * q_embs).sum(axis=1)  # cosine sim (both normalized)

    # ── Stage 4: context precision -- judge each retrieved chunk vs claim ────
    print("Judging context precision (chunk relevant to claim?) ...")
    cp_msgs, cp_index = [], []
    for i, rec in enumerate(records):
        for c in rec["_flat_chunks"]:
            cp_msgs.append([{"role": "system", "content": CONTEXT_PRECISION_SYSTEM},
                              {"role": "user", "content": CONTEXT_PRECISION_USER.format(claim=rec["claim"], chunk_text=c["text"][:500])}])
            cp_index.append(i)
    if "cp" not in cache:
        cp_results = generate_with_retry(llm, tokenizer, cp_msgs, lambda t: (parse_relevant(t),), seed=SEED, max_tokens=40)
        cp_relevances = [[] for _ in records]
        for i, r in zip(cp_index, cp_results):
            cp_relevances[i].append(bool(r["parsed"][0]))
        cache["cp"] = cp_relevances
        save_cache()
    cp_relevances = cache["cp"]

    # ── write out ──────────────────────────────────────────────────────────
    out_path = ROOT / f"results/ragas_lite_scores_{args.split}.jsonl"
    with open(out_path, "w") as fout:
        for i, rec in enumerate(records):
            supported = [s for s in faith_scores[i] if s is not None]
            faithfulness = sum(supported) / max(1, len(supported)) if supported else None
            fout.write(json.dumps({
                "claim_id": rec["claim_id"], "model": rec["model"], "arm": rec["arm"],
                "cards_category": rec.get("cards_category"),
                "faithfulness": faithfulness,
                "answer_relevance": float(answer_relevance[i]),
                "context_precision": context_precision_at_k(cp_relevances[i]),
                "n_statements": len(statements_per_record[i]),
                "n_chunks": len(rec["_flat_chunks"]),
            }, ensure_ascii=False) + "\n")

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
