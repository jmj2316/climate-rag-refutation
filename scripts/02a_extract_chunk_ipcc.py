"""
Extract text from IPCC PDFs and split into overlapping chunks.
Run: conda activate climate_rag && python 02a_extract_chunk_ipcc.py

Chunking: ~600 words per chunk (~800 tokens), 100-word overlap, sentence-boundary
aware (simple regex split -- good enough for report prose, not code/tables).
"""
import json
import re
from pathlib import Path
from pypdf import PdfReader

PDF_DIR = Path(__file__).parent.parent / "data/ipcc_pdfs"
OUT_PATH = Path(__file__).parent.parent / "kb/chunks.jsonl"

CHUNK_WORDS = 600
OVERLAP_WORDS = 100

SOURCE_LABELS = {
    "wg1_spm":  "IPCC AR6 WG1 Summary for Policymakers (Physical Science Basis)",
    "wg2_spm":  "IPCC AR6 WG2 Summary for Policymakers (Impacts, Adaptation, Vulnerability)",
    "wg2_ts":   "IPCC AR6 WG2 Technical Summary (Impacts, Adaptation, Vulnerability)",
    "wg3_spm":  "IPCC AR6 WG3 Summary for Policymakers (Mitigation)",
    "wg3_ts":   "IPCC AR6 WG3 Technical Summary (Mitigation)",
    "syr_spm":  "IPCC AR6 Synthesis Report Summary for Policymakers",
    "syr_full": "IPCC AR6 Synthesis Report (Longer Report)",
}


def clean_page_text(text: str) -> str:
    # drop page-number-only lines and collapse whitespace
    lines = [ln for ln in text.split("\n") if not re.fullmatch(r"\s*\d+\s*", ln)]
    text = " ".join(lines)
    # strip IPCC's inline cross-reference clusters, e.g. "{2.4.2, Box 2.1, Figure TS.5}"
    text = re.sub(r"\{[^{}]{0,200}\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_words(words: list[str], page_map: list[int]) -> list[dict]:
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunk_text = " ".join(words[start:end])
        page_start = page_map[start]
        page_end = page_map[end - 1]
        chunks.append({"text": chunk_text, "page_start": page_start, "page_end": page_end})
        if end == len(words):
            break
        start = end - OVERLAP_WORDS
    return chunks


def main():
    all_chunks = []
    chunk_id = 0
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        source = pdf_path.stem
        source_label = SOURCE_LABELS.get(source, source)
        print(f"Extracting {pdf_path.name} ...")
        reader = PdfReader(str(pdf_path))

        words, page_map = [], []
        for page_idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"  page {page_idx}: extract failed ({e})")
                continue
            text = clean_page_text(text)
            if not text:
                continue
            page_words = text.split(" ")
            words.extend(page_words)
            page_map.extend([page_idx + 1] * len(page_words))

        doc_chunks = chunk_words(words, page_map)
        for c in doc_chunks:
            all_chunks.append({
                "chunk_id":     f"{source}_{chunk_id}",
                "source":       source,
                "source_label": source_label,
                "page_start":   c["page_start"],
                "page_end":     c["page_end"],
                "text":         c["text"],
            })
            chunk_id += 1
        print(f"  {len(reader.pages)} pages -> {len(doc_chunks)} chunks")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nTotal chunks: {len(all_chunks)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
