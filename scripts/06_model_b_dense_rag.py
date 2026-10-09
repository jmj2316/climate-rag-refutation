"""
Model B: standard dense RAG. Embed the claim as-is, retrieve top-k=3 IPCC chunks,
generate verdict + refutation grounded in those chunks.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mj2 python 06_model_b_dense_rag.py \
      --model mistralai/Mistral-7B-Instruct-v0.3 --split pilot
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.rag_templates import MODEL_B_SYSTEM, MODEL_B_USER
from parse_rag_response import parse_verdict_refutation, is_correct
from vllm_utils import load_llm, generate_with_retry, model_slug
from retriever import IPCCRetriever

ROOT = Path(__file__).parent.parent
SEED = 42
TOP_K = 3


def load_claims(split: str) -> list[dict]:
    fname = "pilot_sample.json" if split == "pilot" else "claims_with_cards.json"
    return json.load(open(ROOT / "data/climate_fever" / fname))


def format_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] ({c['source_label']}, p.{c['page_start']}-{c['page_end']}): {c['text']}")
    return "\n\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="pilot", choices=["pilot", "full"])
    p.add_argument("--gpu-mem-util", type=float, default=0.6)
    args = p.parse_args()

    claims = load_claims(args.split)
    print(f"Loaded {len(claims)} claims (split={args.split})")

    print("Retrieving top-k chunks for each claim ...")
    retriever = IPCCRetriever(device="cuda")
    retrieved = retriever.query_batch([c["claim"] for c in claims], k=TOP_K)

    out_dir = ROOT / "results/model_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_slug(args.model)}_{args.split}.jsonl"

    llm = load_llm(args.model, seed=SEED, gpu_mem=args.gpu_mem_util)
    tokenizer = llm.get_tokenizer()

    messages = []
    for c, chunks in zip(claims, retrieved):
        context = format_context(chunks)
        messages.append([
            {"role": "system", "content": MODEL_B_SYSTEM},
            {"role": "user", "content": MODEL_B_USER.format(claim=c["claim"], context=context)},
        ])

    results = generate_with_retry(llm, tokenizer, messages, parse_verdict_refutation, seed=SEED, max_tokens=400)

    n_ok = 0
    with open(out_path, "w") as fout:
        for c, chunks, r in zip(claims, retrieved, results):
            verdict, refutation = r["parsed"]
            correct = is_correct(c["true_label"], verdict)
            n_ok += int(r["parse_success"])
            fout.write(json.dumps({
                "claim_id": c["claim_id"], "claim": c["claim"], "true_label": c["true_label"],
                "cards_category": c.get("cards_category"), "model": args.model, "arm": "B_dense_rag",
                "retrieved_chunks": [{"chunk_id": ch["chunk_id"], "source": ch["source"], "score": ch["score"],
                                       "text": ch["text"]} for ch in chunks],
                "raw_response": r["raw"], "verdict": verdict, "refutation": refutation,
                "correct": correct, "parse_success": r["parse_success"],
            }, ensure_ascii=False) + "\n")

    print(f"Parse success: {n_ok}/{len(claims)} -> {out_path}")


if __name__ == "__main__":
    main()
