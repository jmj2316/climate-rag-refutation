"""Hybrid retriever over the IPCC KB: dense (bge-m3 + FAISS) fused with sparse
(BM25) via Reciprocal Rank Fusion. Dense embeddings alone under-retrieve for
queries with specific named entities/numbers (place names, years, study names)
that CLIMATE-FEVER claims often contain -- BM25's exact-term matching covers
that gap; RRF combines the two rankings without needing to calibrate scores
onto a shared scale.
"""
import os
os.environ.setdefault("USE_TF", "0")

import json
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

KB_DIR = Path(__file__).parent.parent / "kb"
INDEX_PATH = KB_DIR / "faiss.index"
META_PATH = KB_DIR / "chunks_meta.json"

MODEL_NAME = "BAAI/bge-m3"
RRF_K = 60          # standard RRF damping constant
CANDIDATE_MULT = 8  # pull top (k * CANDIDATE_MULT) from each ranker before fusing


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class IPCCRetriever:
    def __init__(self, device: str = "cuda"):
        self.index = faiss.read_index(str(INDEX_PATH))
        self.chunks = json.load(open(META_PATH))
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"Building BM25 index over {len(self.chunks)} chunks ...")
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

    def _dense_ranked(self, emb, n: int) -> list[list[int]]:
        _, idxs = self.index.search(emb, n)
        return [[i for i in row if i >= 0] for row in idxs]

    def _bm25_ranked(self, queries: list[str], n: int) -> list[list[int]]:
        out = []
        for q in queries:
            scores = self.bm25.get_scores(_tokenize(q))
            top = scores.argsort()[::-1][:n]
            out.append([int(i) for i in top])
        return out

    def _fuse(self, dense_idxs: list[int], bm25_idxs: list[int], k: int) -> list[dict]:
        rrf_scores = {}
        for rank, idx in enumerate(dense_idxs):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_idxs):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        ranked = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:k]
        results = []
        for idx, score in ranked:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def query(self, text: str, k: int = 3) -> list[dict]:
        return self.query_batch([text], k=k)[0]

    def query_batch(self, texts: list[str], k: int = 3) -> list[list[dict]]:
        n = k * CANDIDATE_MULT
        embs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32).astype("float32")
        dense_lists = self._dense_ranked(embs, n)
        bm25_lists = self._bm25_ranked(texts, n)
        return [self._fuse(d, b, k) for d, b in zip(dense_lists, bm25_lists)]
