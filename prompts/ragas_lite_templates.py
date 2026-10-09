"""
Prompt templates for a self-contained re-implementation of the 3 RAGAS metrics
(Faithfulness, Answer Relevance, Context Precision), scored by an independent
judge LLM (not one of the 3 models under test) run directly through vLLM.

We reimplement these natively instead of depending on the `ragas` package because
`ragas` requires a live langchain LLM connection, which would mean standing up a
cross-process OpenAI-compatible server between the GPU (vLLM) and CPU (ragas/
langchain) conda environments on this cluster -- the metric definitions below
follow the published RAGAS methodology (Es et al. 2023) directly against vLLM.
"""

# ── Faithfulness: does each statement in the refutation follow from the context? ─
SPLIT_STATEMENTS_SYSTEM = "You split a paragraph into its individual factual statements."
SPLIT_STATEMENTS_USER = """\
Paragraph:
"{refutation}"

Break this into a numbered list of individual, standalone factual statements \
(one claim per line, no more than 6 statements).

Respond in exactly this format (one per line):
1. [statement 1]
2. [statement 2 (if any)]
..."""

FAITHFULNESS_SYSTEM = (
    "You judge whether a statement is directly supported by (can be inferred from) "
    "a given set of source excerpts."
)
FAITHFULNESS_USER = """\
Source excerpts:
{context}

Statement: "{statement}"

Is this statement directly supported by the source excerpts above (not by outside \
knowledge)?

Respond in exactly this format:
Supported: [Yes / No]"""

# ── Answer relevance: does the refutation stay on-topic for the claim? ──────────
GEN_QUESTION_SYSTEM = "You generate the question that a given answer is most directly responding to."
GEN_QUESTION_USER = """\
Answer: "{refutation}"

Write ONE question that this answer directly and completely responds to (as if \
this answer were given without seeing the original claim).

Respond in exactly this format:
Question: [your question]"""

# ── Context precision: is each retrieved chunk actually relevant to the claim? ──
CONTEXT_PRECISION_SYSTEM = (
    "You judge whether a retrieved document excerpt is relevant evidence for "
    "fact-checking a given claim."
)
CONTEXT_PRECISION_USER = """\
Claim: "{claim}"

Excerpt: "{chunk_text}"

Is this excerpt relevant evidence for fact-checking the claim?

Respond in exactly this format:
Relevant: [Yes / No]"""
