# Does Retrieval Help Small LLMs Refute Climate Misinformation?

Code, prompts and results for *Does Retrieval Help Small LLMs Refute Climate Misinformation? A Benchmark of Zero-Shot, Dense RAG, and Decomposed RAG over IPCC AR6* (Minjong Cheon, manuscript under review).

The study compares three pipelines on 1,535 CLIMATE-FEVER claims with three 7B chat models (Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Zephyr-7B-beta) and the full IPCC AR6 reports as knowledge base: **A** zero-shot judgment, **B** dense RAG, **C** sub-claim decomposition with LLM-filtered retrieval. An ablation ladder (E1-E3) separates context size, decomposition and filtering. Generations are scored against the dataset labels and by an independent judge (Llama-3.1-8B-Instruct).

## Contents

| Path | What |
|---|---|
| `scripts/` | Numbered pipeline (see below) |
| `prompts/` | All prompt templates (generation, judge, RAGAS-style metrics) |
| `data/climate_fever/` | CLIMATE-FEVER claims as used, with CARDS category labels |
| `kb/chunk_index.csv` | Index of the 12,704 knowledge-base chunks (id, report, pages, length); the text is rebuilt from the public PDFs by `scripts/01_download_ipcc.py` and `scripts/02a_extract_chunk_ipcc.py`; a fresh rebuild gives identical ids, pages and text for 12,702 of the 12,704 chunks, and two chunks (`wg1_full_914`, `wg2_full_8862`) differ in one or two characters of PDF-font decoding inside figure text |
| `results/` | Judge scores, RAGAS-style scores, bootstrap CIs, paired tests, matched-knowledge-base experiment, format audit; every table and figure in the paper is generated from these files |
| Release asset `climate-rag-refutation_outputs_v1.0.0.zip` | Per-claim outputs of all arms (`results/model_a`, `model_b`, `model_c`, `results/ablation/E1_*`, `E2_*`, `E3_*`) and the pre-reparse copies in `results/_backup_preparse/` used by `14_extra_analysis.py`; unzip at the repository root |

## Regenerate the statistics, tables and figures (CPU)

```bash
pip install -r requirements.txt
# download climate-rag-refutation_outputs_v1.0.0.zip from the release page, then:
unzip climate-rag-refutation_outputs_v1.0.0.zip
cd scripts
python 08_eval_classification.py --split full
python 11_aggregate_results.py
python 12_stats_tests.py
python 14_extra_analysis.py
python 19_format_robustness.py
python 20_ablation_stats.py
python 21_decomp_quality.py
python 15_make_tables.py      # writes paper/tab_*.tex
python 16_make_figures.py     # writes paper/figs/*.pdf (needs matplotlib)
```

## Re-run the experiments (GPU)

Generation used vLLM 0.6.6.post1 with a fixed seed (42); one 24-48 GB GPU per 7B model is sufficient.

1. Data and knowledge base: `00_download_climate_fever.py`, `01_download_ipcc.py`, `02a_extract_chunk_ipcc.py`, `02b_embed_index.py` (BAAI/bge-m3 + FAISS), `03_cards_classify.py`, `04_select_pilot.py`.
2. Pipelines: `05_model_a_zeroshot.py`, `06_model_b_dense_rag.py`, `07_model_c_decompose_rag.py` (each `--model <hf-id> --split full`), then `13_reparse_refutations.py`.
3. Scoring: `09_llm_judge.py`, `10_eval_ragas_lite.py`.
4. Ablation and matched knowledge-base experiment: `17_ablation.py --model <hf-id>`, `18_kb_matched.py`.
5. Statistics, tables and figures as above.

## Licenses

Code: MIT (`LICENSE`). CLIMATE-FEVER claims and the IPCC AR6 reports keep their original licenses and terms; the released per-claim outputs contain short retrieved excerpts of the reports for evaluation purposes. Model weights are obtained from their original Hugging Face repositories under their own licenses.
