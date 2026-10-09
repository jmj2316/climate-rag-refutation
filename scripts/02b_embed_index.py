"""
Embed KB chunks with bge-m3 and build a FAISS index.
Run: CUDA_VISIBLE_DEVICES=2 USE_TF=0 conda run -n mj2 python 02b_embed_index.py
"""
import os
os.environ.setdefault("USE_TF", "0")

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KB_DIR = Path(__file__).parent.parent / "kb"
CHUNKS_PATH = KB_DIR / "chunks.jsonl"
INDEX_PATH = KB_DIR / "faiss.index"
META_PATH = KB_DIR / "chunks_meta.json"

MODEL_NAME = "BAAI/bge-m3"


def main():
    chunks = [json.loads(l) for l in open(CHUNKS_PATH)]
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME, device="cuda" if os_has_cuda() else "cpu")
    texts = [c["text"] for c in chunks]

    print("Embedding ...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim via normalized vectors + inner product
    index.add(embeddings)

    KB_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Index: {index.ntotal} vectors, dim={dim} -> {INDEX_PATH}")
    print(f"Meta -> {META_PATH}")


def os_has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
