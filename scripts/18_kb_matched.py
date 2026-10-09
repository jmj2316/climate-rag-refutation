"""
Matched 2x2 knowledge-base experiment: {summary KB, full KB} x {dense only, hybrid dense+BM25}.
Same 1,535 claims, same top-3 cut, same judge prompt (Llama-3.1-8B, CONTEXT_PRECISION template) as the
RAGAS-lite context-precision metric. Replaces the unmatched pilot comparison.

summary KB = *_spm (Summaries for Policymakers) + *_ts (Technical Summaries) + syr_full (Synthesis Report
longer report) = 517 chunks.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python -B 18_kb_matched.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.ragas_lite_templates import CONTEXT_PRECISION_SYSTEM, CONTEXT_PRECISION_USER
from retriever import IPCCRetriever, _tokenize, RRF_K, CANDIDATE_MULT
from vllm_utils import load_llm, generate_with_retry
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).parent.parent
SEED = 42
K = 3
JUDGE = "NousResearch/Meta-Llama-3.1-8B-Instruct"
SUMMARY_SOURCES = {"wg1_spm", "wg2_spm", "wg3_spm", "syr_spm", "wg2_ts", "wg3_ts", "syr_full"}


def parse_relevant(text):
    import re
    m = re.search(r"Relevant\s*:\s*(\S+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("[]").lower().startswith("y")
    s = text.strip().strip("[]").lower()
    if s.startswith("yes"):
        return True
    if s.startswith("no"):
        return False
    return None


def cp_at_k(rel):
    tot = sum(rel)
    if tot == 0:
        return 0.0
    hits, sc = 0, 0.0
    for k, r in enumerate(rel, 1):
        if r:
            hits += 1
            sc += hits / k
    return sc / tot


def main():
    claims = json.load(open(ROOT / "data/climate_fever/claims_with_cards.json"))
    texts = [c["claim"] for c in claims]
    ret = IPCCRetriever(device="cuda")
    chunks = ret.chunks
    N = len(chunks)
    all_emb = ret.index.reconstruct_n(0, N)
    q_emb = ret.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32).astype("float32")
    toks = [_tokenize(c["text"]) for c in chunks]
    q_toks = [_tokenize(t) for t in texts]

    sel = {"summary": np.array([i for i, c in enumerate(chunks) if c["source"] in SUMMARY_SOURCES]),
           "full": np.arange(N)}
    print({k: len(v) for k, v in sel.items()}, flush=True)
    assert len(sel["summary"]) == 517, len(sel["summary"])

    retrieved = {}
    for kb, ids in sel.items():
        emb = all_emb[ids]
        sims = q_emb @ emb.T
        dense_rank = np.argsort(-sims, axis=1)[:, :K * CANDIDATE_MULT]
        bm25 = BM25Okapi([toks[i] for i in ids])
        bm_rank = [np.argsort(-bm25.get_scores(q))[:K * CANDIDATE_MULT] for q in q_toks]
        dense_only, hybrid = [], []
        for d, b in zip(dense_rank, bm_rank):
            dense_only.append([int(ids[j]) for j in d[:K]])
            sc = {}
            for r, j in enumerate(d):
                sc[j] = sc.get(j, 0.0) + 1.0 / (RRF_K + r + 1)
            for r, j in enumerate(b):
                sc[j] = sc.get(j, 0.0) + 1.0 / (RRF_K + r + 1)
            hybrid.append([int(ids[j]) for j, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:K]])
        retrieved[(kb, "dense")] = dense_only
        retrieved[(kb, "hybrid")] = hybrid
        print("retrieved", kb, flush=True)

    # sanity: full+hybrid must reproduce the retrieval stored for pipeline B
    bpath = ROOT / "results/model_b/Qwen2_5-7B-Instruct_full.jsonl"
    stored = {json.loads(l)["claim_id"]: [c["chunk_id"] for c in json.loads(l)["retrieved_chunks"]] for l in open(bpath)}
    same = sum(stored[c["claim_id"]] == [chunks[i]["chunk_id"] for i in retrieved[("full", "hybrid")][n]]
               for n, c in enumerate(claims))
    print(f"full+hybrid reproduces stored B retrieval for {same}/{len(claims)} claims", flush=True)

    pairs = sorted({(n, i) for lst in retrieved.values() for n, row in enumerate(lst) for i in row})
    print("unique (claim, chunk) pairs to judge:", len(pairs), flush=True)
    llm = load_llm(JUDGE, seed=SEED, max_model_len=4096, gpu_mem=0.5)
    tok = llm.get_tokenizer()
    msgs = [[{"role": "system", "content": CONTEXT_PRECISION_SYSTEM},
             {"role": "user", "content": CONTEXT_PRECISION_USER.format(claim=texts[n], chunk_text=chunks[i]["text"][:500])}]
            for n, i in pairs]
    res = generate_with_retry(llm, tok, msgs, lambda t: (parse_relevant(t),), seed=SEED, max_tokens=40)
    rel = {p: bool(r["parsed"][0]) for p, r in zip(pairs, res)}
    pickle.dump(rel, open(ROOT / "results/_kb_matched_rel.pkl", "wb"))
    with open(ROOT / "results/kb_matched_raw.jsonl", "w") as f:
        for (kb, mode), lst in retrieved.items():
            for n, row in enumerate(lst):
                f.write(json.dumps({"claim_id": claims[n]["claim_id"], "kb": kb, "retrieval": mode,
                                    "chunk_ids": [chunks[i]["chunk_id"] for i in row],
                                    "relevant": [rel[(n, i)] for i in row]}) + "\n")

    rng = np.random.default_rng(0)
    out = []
    for (kb, mode), lst in retrieved.items():
        per = [[rel[(n, i)] for i in row] for n, row in enumerate(lst)]
        cp = np.array([cp_at_k(r) for r in per])
        hit = np.array([any(r) for r in per], dtype=float)
        frac = np.array([np.mean(r) for r in per])
        boots = rng.integers(0, len(cp), size=(10000, len(cp)))
        rec = {"kb": kb, "retrieval": mode, "n_chunks_in_kb": len(sel[kb]),
               "context_precision": float(cp.mean()), "cp_lo": float(np.percentile(cp[boots].mean(1), 2.5)),
               "cp_hi": float(np.percentile(cp[boots].mean(1), 97.5)),
               "any_relevant": float(hit.mean()), "hit_lo": float(np.percentile(hit[boots].mean(1), 2.5)),
               "hit_hi": float(np.percentile(hit[boots].mean(1), 97.5)),
               "frac_relevant": float(frac.mean())}
        out.append(rec)
        print(rec, flush=True)
    json.dump(out, open(ROOT / "results/kb_matched_2x2.json", "w"), indent=1)

    # paired contrasts on per-claim CP
    per_cp = {k: np.array([cp_at_k([rel[(n, i)] for i in row]) for n, row in enumerate(lst)]) for k, lst in retrieved.items()}
    contrasts = {"full vs summary (hybrid)": (("full", "hybrid"), ("summary", "hybrid")),
                 "full vs summary (dense)": (("full", "dense"), ("summary", "dense")),
                 "hybrid vs dense (full)": (("full", "hybrid"), ("full", "dense")),
                 "hybrid vs dense (summary)": (("summary", "hybrid"), ("summary", "dense"))}
    paired = []
    for name, (a, b) in contrasts.items():
        d = per_cp[a] - per_cp[b]
        bm = d[boots].mean(1)
        paired.append({"contrast": name, "delta_cp": float(d.mean()), "lo": float(np.percentile(bm, 2.5)),
                       "hi": float(np.percentile(bm, 97.5))})
        print(paired[-1], flush=True)
    json.dump(paired, open(ROOT / "results/kb_matched_paired.json", "w"), indent=1)


if __name__ == "__main__":
    main()
