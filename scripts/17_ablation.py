"""
Ablation ladder between dense RAG (B) and decompose+filter RAG (C), full 1,535 claims.

Arms (all single greedy pass, same prompts/parsers as 06/07):
  E1_decomp_nofilter : C's own sub-claims, top-2 by retrieval score per sub-claim (no LLM relevance
                       filter), C prompt/layout.            -> C vs E1 isolates the filter
  E2_raw_matchC       : raw claim query, n_i chunks where n_i = #chunks in C's context for that claim,
                       250-word truncation, B prompt.        -> C vs E2 = decompose+filter vs raw query
                                                                at matched context size
  E3_raw_matchE1      : raw claim query, n_i = #chunks in E1's context, 250-word truncation, B prompt.
                                                             -> E1 vs E3 isolates decomposition,
                                                                B vs E3 isolates context size/truncation

Run (one model per process):
  CUDA_VISIBLE_DEVICES=0 python -B 17_ablation.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.rag_templates import MODEL_B_SYSTEM, MODEL_B_USER, MODEL_C_SYSTEM, MODEL_C_USER
from parse_rag_response import parse_verdict_refutation, is_correct
from vllm_utils import load_llm, generate_with_retry, model_slug
from retriever import IPCCRetriever

ROOT = Path(__file__).parent.parent
SEED = 42
RETRIEVE_K = 5
MAX_KEPT = 2
WORDS = 250


def trunc(t):
    return " ".join(t.split(" ")[:WORDS])


def fmt_b(chunks):
    return "\n\n".join(f"[{i}] ({c['source_label']}, p.{c['page_start']}-{c['page_end']}): {trunc(c['text'])}"
                       for i, c in enumerate(chunks, 1))


def fmt_c(groups):
    lines, n = [], 1
    for sc, chunks in groups:
        lines.append(f"Sub-claim: \"{sc}\"")
        for c in chunks:
            lines.append(f"  [{n}] ({c['source_label']}, p.{c['page_start']}-{c['page_end']}): {trunc(c['text'])}")
            n += 1
        lines.append("")
    return "\n".join(lines).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=None, help="cap concurrent sequences (avoids KV-cache preemption)")
    ap.add_argument("--device", default="cuda", help="device for the query encoder (cpu for --prep-only)")
    ap.add_argument("--prep-only", action="store_true", help="build and cache the prompts, skip generation")
    args = ap.parse_args()

    claims = json.load(open(ROOT / "data/climate_fever/claims_with_cards.json"))
    cpath = ROOT / "results/model_c" / f"{model_slug(args.model)}_full.jsonl"
    crec = {}
    for line in open(cpath):
        r = json.loads(line)
        crec[r["claim_id"]] = r
    if args.limit:
        claims = claims[:args.limit]

    prep = ROOT / "results" / f"_abl_prep_{model_slug(args.model)}.pkl"
    if prep.exists() and not args.limit:
        arms, used = pickle.load(open(prep, "rb"))
        print("loaded cached prompts", prep.name, flush=True)
    else:
        ret = IPCCRetriever(device=args.device)
        encode, memo = ret.model.encode, {}

        def cached_encode(texts, **kw):
            miss = [t for t in texts if t not in memo]
            if miss:
                memo.update(zip(miss, encode(miss, **kw)))
            return np.stack([memo[t] for t in texts])

        ret.model.encode = cached_encode
        subs = [crec[c["claim_id"]]["subclaims"] for c in claims]
        n_c = [sum(len(g["chunks"]) for g in crec[c["claim_id"]]["retrieved_chunks"]) for c in claims]

        flat = [(i, s) for i, ss in enumerate(subs) for s in ss]
        flat_ret = ret.query_batch([s for _, s in flat], k=RETRIEVE_K)
        e1_groups = [[] for _ in claims]
        for (i, s), ch in zip(flat, flat_ret):
            e1_groups[i].append((s, sorted(ch, key=lambda c: -c["score"])[:MAX_KEPT]))
        n_e1 = [sum(len(g[1]) for g in gs) for gs in e1_groups]

        texts = [c["claim"] for c in claims]
        raw_by_n = {n: ret.query_batch(texts, k=n) for n in sorted(set(n_c) | set(n_e1))}
        raw_c = [raw_by_n[n][i] for i, n in enumerate(n_c)]
        raw_e1 = [raw_by_n[n][i] for i, n in enumerate(n_e1)]
        arms = {
            "E1_decomp_nofilter": [
                [{"role": "system", "content": MODEL_C_SYSTEM},
                 {"role": "user", "content": MODEL_C_USER.format(claim=c["claim"], context=fmt_c(gs))}]
                for c, gs in zip(claims, e1_groups)],
            "E2_raw_matchC": [
                [{"role": "system", "content": MODEL_B_SYSTEM},
                 {"role": "user", "content": MODEL_B_USER.format(claim=c["claim"], context=fmt_b(r))}]
                for c, r in zip(claims, raw_c)],
            "E3_raw_matchE1": [
                [{"role": "system", "content": MODEL_B_SYSTEM},
                 {"role": "user", "content": MODEL_B_USER.format(claim=c["claim"], context=fmt_b(r))}]
                for c, r in zip(claims, raw_e1)],
        }
        used = {"E1_decomp_nofilter": [[ch for _, chs in gs for ch in chs] for gs in e1_groups],
                "E2_raw_matchC": raw_c, "E3_raw_matchE1": raw_e1}

        if not args.limit:
            pickle.dump((arms, used), open(prep, "wb"))
    if args.prep_only:
        return

    llm = load_llm(args.model, seed=SEED, max_model_len=8192, gpu_mem=args.gpu_mem_util, max_num_seqs=args.max_num_seqs)
    tok = llm.get_tokenizer()
    for arm, msgs in arms.items():
        out_dir = ROOT / "results/ablation" / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.limit and (out_dir / f"{model_slug(args.model)}_full.jsonl").exists():
            print(f"{arm}: output exists, skipped", flush=True)
            continue
        res = generate_with_retry(llm, tok, msgs, parse_verdict_refutation, seed=SEED, max_tokens=400)
        n_ok = 0
        with open(out_dir / f"{model_slug(args.model)}_full.jsonl", "w") as f:
            for c, r, ch in zip(claims, res, used[arm]):
                v, ref = r["parsed"]
                n_ok += int(r["parse_success"])
                f.write(json.dumps({
                    "claim_id": c["claim_id"], "claim": c["claim"], "true_label": c["true_label"],
                    "cards_category": c.get("cards_category"), "model": args.model, "arm": arm,
                    "retrieved_chunks": [{"chunk_id": x["chunk_id"], "score": x["score"], "text": x["text"]} for x in ch],
                    "raw_response": r["raw"], "verdict": v, "refutation": ref,
                    "correct": is_correct(c["true_label"], v), "parse_success": r["parse_success"],
                }, ensure_ascii=False) + "\n")
        print(f"{arm}: parse success {n_ok}/{len(claims)}", flush=True)


if __name__ == "__main__":
    main()
