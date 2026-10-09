"""Prompt templates for the RAG refutation benchmark (Models A/B/C + CARDS classifier + judge)."""

# ── CARDS taxonomy (Coan et al. 2021 / Skeptical Science super-claims) ──────────
CARDS_CATEGORIES = {
    "1_not_happening":       "It's not happening -- climate change / global warming is not occurring, or the data is faked/unreliable.",
    "2_not_us":               "It's not us -- humans / CO2 are not the cause; warming is natural (sun, volcanoes, natural cycles).",
    "3_not_bad":               "It's not bad -- warming's impacts are minimal, beneficial, or exaggerated; species/ecosystems will adapt fine.",
    "4_solutions_wont_work":   "Solutions won't work -- renewable energy / policy / mitigation is too expensive, unreliable, or futile.",
    "5_science_unreliable":    "Climate science / scientists are unreliable -- conspiracy, biased institutions, no real consensus, movement is political/alarmist.",
    "0_not_denial":            "Not a denialist claim -- consistent with mainstream climate science, or a neutral/factual statement.",
}

CARDS_SYSTEM = (
    "You are an expert annotator trained on the CARDS (Computer-Assisted Recognition "
    "of Denial and Skepticism) taxonomy for climate misinformation."
)

CARDS_USER = """\
Classify the following claim into exactly one of these categories:

0_not_denial: {cat0}
1_not_happening: {cat1}
2_not_us: {cat2}
3_not_bad: {cat3}
4_solutions_wont_work: {cat4}
5_science_unreliable: {cat5}

Claim: "{claim}"

Respond in exactly this format:
Category: [one of: 0_not_denial / 1_not_happening / 2_not_us / 3_not_bad / 4_solutions_wont_work / 5_science_unreliable]
Reason: [one short sentence]"""


def build_cards_prompt(claim: str) -> str:
    return CARDS_USER.format(
        claim=claim,
        cat0=CARDS_CATEGORIES["0_not_denial"],
        cat1=CARDS_CATEGORIES["1_not_happening"],
        cat2=CARDS_CATEGORIES["2_not_us"],
        cat3=CARDS_CATEGORIES["3_not_bad"],
        cat4=CARDS_CATEGORIES["4_solutions_wont_work"],
        cat5=CARDS_CATEGORIES["5_science_unreliable"],
    )


# ── Model A: pure LLM zero-shot (no retrieval) ───────────────────────────────────
MODEL_A_SYSTEM = (
    "You are a careful climate-science fact-checker. Judge claims using only your "
    "own training knowledge -- no external documents are provided."
)

MODEL_A_USER = """\
Claim: "{claim}"

Respond in exactly this format:
Verdict: [Supports / Refutes / Not enough information]
Refutation: [2-4 sentences. If the claim is inaccurate, explain why using your own \
scientific knowledge. If accurate, briefly confirm why.]"""


# ── Model B: standard dense RAG (top-k=3, claim embedded as-is) ─────────────────
MODEL_B_SYSTEM = (
    "You are a careful climate-science fact-checker. Judge the claim using ONLY the "
    "provided IPCC AR6 excerpts as evidence. Do not use outside knowledge."
)

MODEL_B_USER = """\
Claim: "{claim}"

IPCC AR6 excerpts:
{context}

Respond in exactly this format:
Verdict: [Supports / Refutes / Not enough information]
Refutation: [2-4 sentences, citing the excerpt(s) above by number, e.g. "[1]". \
If the excerpts don't address the claim, say so and answer "Not enough information".]"""


# ── Model C: sub-claim decomposition + query expansion + filtered RAG ───────────
DECOMPOSE_SYSTEM = (
    "You break down complex or compound claims about climate change into a list of "
    "atomic, independently-checkable factual sub-claims."
)

DECOMPOSE_USER = """\
Claim: "{claim}"

List 1-3 atomic sub-claims this breaks down into (a single claim may need only 1). \
Each sub-claim should be a self-contained factual statement that can be looked up \
in a scientific report, phrased as a search query.

Respond in exactly this format (one per line):
1. [sub-claim 1]
2. [sub-claim 2 (if any)]
3. [sub-claim 3 (if any)]"""

MODEL_C_SYSTEM = (
    "You are a careful climate-science fact-checker. Judge the claim using ONLY the "
    "provided IPCC AR6 excerpts, which were retrieved for each of its sub-claims and "
    "filtered for relevance. Do not use outside knowledge."
)

MODEL_C_USER = """\
Claim: "{claim}"

This claim decomposes into these sub-claims, each with its own retrieved evidence:
{context}

Respond in exactly this format:
Verdict: [Supports / Refutes / Not enough information]
Refutation: [2-4 sentences addressing each sub-claim, citing excerpts by number, \
e.g. "[1]". If the excerpts don't address a sub-claim, note that explicitly.]"""

# relevance filter applied to each retrieved chunk before it enters Model C's context
FILTER_SYSTEM = (
    "You judge whether a retrieved document excerpt is actually relevant evidence "
    "for a given sub-claim."
)

FILTER_USER = """\
Sub-claim: "{subclaim}"

Excerpt: "{chunk_text}"

Is this excerpt relevant evidence for judging the sub-claim (as opposed to being \
about a different, unrelated topic)?

Respond in exactly this format:
Relevant: [Yes / No]"""


# ── LLM-as-judge (G-Eval style rubric) ───────────────────────────────────────────
JUDGE_SYSTEM = (
    "You are an expert reviewer scoring AI-generated climate misinformation refutations "
    "against a strict rubric. Be critical -- do not default to high scores."
)

JUDGE_USER = """\
Original claim: "{claim}"

Generated refutation:
"{refutation}"

Score the refutation on two dimensions, each 1-5:

1. Logical Validity: Does it accurately identify and directly rebut (or correctly \
confirm) the specific factual premise of the claim, with sound reasoning?
   1 = does not engage with the claim's actual premise / non-sequitur
   3 = partially engages, some gaps in logic
   5 = precisely targets and correctly resolves the claim's core premise

2. Scientific Rigor: Does it reflect IPCC-style calibrated uncertainty language \
correctly (e.g. not overstating "high confidence" findings as certain, not \
understating well-established findings) and avoid factual errors?
   1 = confidently wrong, or fabricates certainty/uncertainty not supported by evidence
   3 = broadly correct but uncertainty language is loose or generic
   5 = precise, appropriately calibrated, no factual errors

Respond in exactly this format:
Logical Validity: [1-5]
Scientific Rigor: [1-5]
Justification: [1-2 sentences]"""
