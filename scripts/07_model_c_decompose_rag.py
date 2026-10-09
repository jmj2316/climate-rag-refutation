"""
Model C (proposed): decompose claim into sub-claims -> retrieve per sub-claim ->
filter retrieved chunks for relevance -> generate verdict + refutation from the
filtered, sub-claim-grouped context.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mj2 python 07_model_c_decompose_rag.py \
      --model mistralai/Mistral-7B-Instruct-v0.3 --split pilot
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.rag_templates import (
    DECOMPOSE_SYSTEM, DECOMPOSE_USER,
    FILTER_SYSTEM, FILTER_USER,
    MODEL_C_SYSTEM, MODEL_C_USER,
)
from parse_rag_response import parse_subclaims, parse_relevance, parse_verdict_refutation, is_correct
from vllm_utils import load_llm, generate_with_retry, model_slug
from retriever import IPCCRetriever

ROOT = Path(__file__).parent.parent
SEED = 42
RETRIEVE_K = 5   # per sub-claim, before filtering
MAX_KEPT_PER_SUBCLAIM = 2  # cap after filtering (or top-score fallback)
CHUNK_WORDS_IN_CONTEXT = 250  # truncate each chunk in the final context to stay within max_model_len


def load_claims(split: str) -> list[dict]:
    fname = "pilot_sample.json" if split == "pilot" else "claims_with_cards.json"
    return json.load(open(ROOT / "data/climate_fever" / fname))


def format_context(subclaim_chunks: list[tuple[str, list[dict]]]) -> str:
    lines = []
    counter = 1
    idx_map = []  # global citation number -> chunk
    for subclaim, chunks in subclaim_chunks:
        lines.append(f"Sub-claim: \"{subclaim}\"")
        if not chunks:
            lines.append("  (no relevant evidence found)")
        for c in chunks:
            text = " ".join(c["text"].split(" ")[:CHUNK_WORDS_IN_CONTEXT])
            lines.append(f"  [{counter}] ({c['source_label']}, p.{c['page_start']}-{c['page_end']}): {text}")
            idx_map.append(c)
            counter += 1
        lines.append("")
    return "\n".join(lines).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="pilot", choices=["pilot", "full"])
    p.add_argument("--gpu-mem-util", type=float, default=0.6)
    args = p.parse_args()

    claims = load_claims(args.split)
    print(f"Loaded {len(claims)} claims (split={args.split})")

    llm = load_llm(args.model, seed=SEED, max_model_len=8192, gpu_mem=args.gpu_mem_util)
    tokenizer = llm.get_tokenizer()

    # ── Stage 1: decompose ────────────────────────────────────────────────────
    print("Stage 1: decomposing claims into sub-claims ...")
    decomp_msgs = [[{"role": "system", "content": DECOMPOSE_SYSTEM},
                     {"role": "user", "content": DECOMPOSE_USER.format(claim=c["claim"])}] for c in claims]
    decomp_results = generate_with_retry(
        llm, tokenizer, decomp_msgs, lambda t: (parse_subclaims(t) or None,), seed=SEED, max_tokens=200,
    )
    subclaims_per_claim = []
    for c, r in zip(claims, decomp_results):
        subs = r["parsed"][0] or [c["claim"]]  # fallback: treat whole claim as one sub-claim
        subclaims_per_claim.append(subs)

    # ── Stage 2: retrieve per sub-claim ──────────────────────────────────────
    print("Stage 2: retrieving evidence per sub-claim ...")
    retriever = IPCCRetriever(device="cuda")
    flat_subclaims = [(i, sc) for i, subs in enumerate(subclaims_per_claim) for sc in subs]
    flat_retrieved = retriever.query_batch([sc for _, sc in flat_subclaims], k=RETRIEVE_K)

    # ── Stage 3: filter each (sub-claim, chunk) pair for relevance ──────────
    print("Stage 3: filtering retrieved chunks for relevance ...")
    filter_msgs, filter_index = [], []  # filter_index[j] = (claim_i, subclaim_text, chunk)
    for (claim_i, sc), chunks in zip(flat_subclaims, flat_retrieved):
        for chunk in chunks:
            filter_msgs.append([
                {"role": "system", "content": FILTER_SYSTEM},
                {"role": "user", "content": FILTER_USER.format(subclaim=sc, chunk_text=chunk["text"][:800])},
            ])
            filter_index.append((claim_i, sc, chunk))

    filter_results = generate_with_retry(
        llm, tokenizer, filter_msgs, lambda t: (parse_relevance(t),), seed=SEED, max_tokens=60,
    )

    # group filtered (kept) chunks by (claim_i, subclaim)
    kept = {}  # (claim_i, sc) -> list[chunk]
    for (claim_i, sc, chunk), r in zip(filter_index, filter_results):
        is_rel = r["parsed"][0]
        key = (claim_i, sc)
        kept.setdefault(key, {"relevant": [], "all": []})
        kept[key]["all"].append(chunk)
        if is_rel:
            kept[key]["relevant"].append(chunk)

    # ── Stage 4: build final context + generate verdict/refutation ──────────
    print("Stage 4: generating final verdict + refutation ...")
    contexts, trace_per_claim = [], []
    for claim_i, subs in enumerate(subclaims_per_claim):
        subclaim_chunks = []
        for sc in subs:
            bucket = kept.get((claim_i, sc), {"relevant": [], "all": []})
            chosen = bucket["relevant"] if bucket["relevant"] else sorted(
                bucket["all"], key=lambda c: -c["score"])[:MAX_KEPT_PER_SUBCLAIM]
            chosen = chosen[:MAX_KEPT_PER_SUBCLAIM]
            subclaim_chunks.append((sc, chosen))
        contexts.append(format_context(subclaim_chunks))
        trace_per_claim.append(subclaim_chunks)

    final_msgs = [[{"role": "system", "content": MODEL_C_SYSTEM},
                    {"role": "user", "content": MODEL_C_USER.format(claim=c["claim"], context=ctx)}]
                   for c, ctx in zip(claims, contexts)]
    final_results = generate_with_retry(llm, tokenizer, final_msgs, parse_verdict_refutation, seed=SEED, max_tokens=400)

    out_dir = ROOT / "results/model_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_slug(args.model)}_{args.split}.jsonl"

    n_ok = 0
    with open(out_path, "w") as fout:
        for c, subs, trace, r in zip(claims, subclaims_per_claim, trace_per_claim, final_results):
            verdict, refutation = r["parsed"]
            correct = is_correct(c["true_label"], verdict)
            n_ok += int(r["parse_success"])
            fout.write(json.dumps({
                "claim_id": c["claim_id"], "claim": c["claim"], "true_label": c["true_label"],
                "cards_category": c.get("cards_category"), "model": args.model, "arm": "C_decompose_filtered_rag",
                "subclaims": subs,
                "retrieved_chunks": [
                    {"subclaim": sc, "chunks": [{"chunk_id": ch["chunk_id"], "source": ch["source"],
                                                  "score": ch["score"], "text": ch["text"]} for ch in chunks]}
                    for sc, chunks in trace
                ],
                "raw_response": r["raw"], "verdict": verdict, "refutation": refutation,
                "correct": correct, "parse_success": r["parse_success"],
            }, ensure_ascii=False) + "\n")

    print(f"Parse success: {n_ok}/{len(claims)} -> {out_path}")


if __name__ == "__main__":
    main()
