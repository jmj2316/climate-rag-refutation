"""Parsers for CARDS classification, verdict+refutation, sub-claims, filter, and judge outputs."""
import re
from typing import Optional

VERDICT_MAP = {
    "supports": "supports",
    "support": "supports",
    "refutes": "refutes",
    "refute": "refutes",
    "not enough information": "nei",
    "not enough info": "nei",
    "nei": "nei",
    "insufficient": "nei",
}

CARDS_VALID = {"0_not_denial", "1_not_happening", "2_not_us", "3_not_bad",
               "4_solutions_wont_work", "5_science_unreliable"}

# CLIMATE-FEVER claim_label: 0 SUPPORTS / 1 REFUTES / 2 NOT_ENOUGH_INFO / 3 DISPUTED
CF_LABEL_TO_VERDICT = {0: "supports", 1: "refutes", 2: "nei", 3: "nei"}  # DISPUTED folded into nei for 3-way scoring


def parse_cards(text: str) -> Optional[str]:
    m = re.search(r"Category\s*:\s*(\S+)", text, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).strip().strip("[]").lower()
    for cat in CARDS_VALID:
        if raw == cat or raw in cat:
            return cat
    return None


def parse_verdict_refutation(text: str) -> tuple[Optional[str], Optional[str]]:
    vm = re.search(r"Verdict\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    verdict = None
    if vm:
        raw = vm.group(1).strip().lower().strip("[]")
        for k, v in VERDICT_MAP.items():
            if k in raw:
                verdict = v
                break
    rm = re.search(r"Refutation\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    refutation = rm.group(1).strip() if rm else None
    if not refutation and vm:
        # model dropped the "Refutation:" label: use the text after the Verdict line
        rest = text[vm.end():].strip()
        rest = re.sub(r"^(?:Explanation|Reasoning|Justification)\s*:\s*", "", rest, flags=re.IGNORECASE)
        refutation = rest or None
    return verdict, refutation


def parse_subclaims(text: str) -> list[str]:
    lines = re.findall(r"^\s*\d\.\s*(.+)$", text, re.MULTILINE)
    return [ln.strip() for ln in lines if ln.strip() and ln.strip().lower() not in ("(if any)", "n/a")]


def parse_relevance(text: str) -> Optional[bool]:
    m = re.search(r"Relevant\s*:\s*(\S+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("[]").lower().startswith("y")
    # fallback: model sometimes answers with a bare Yes/No, no "Relevant:" label
    stripped = text.strip().strip("[]").lower()
    if stripped.startswith("yes"):
        return True
    if stripped.startswith("no"):
        return False
    return None


def parse_judge(text: str) -> tuple[Optional[int], Optional[int]]:
    lv = re.search(r"Logical Validity\s*:\s*(\d)", text, re.IGNORECASE)
    sr = re.search(r"Scientific Rigor\s*:\s*(\d)", text, re.IGNORECASE)
    logical = int(lv.group(1)) if lv else None
    rigor = int(sr.group(1)) if sr else None
    return logical, rigor


def is_correct(true_label: int, predicted_verdict: Optional[str]) -> Optional[bool]:
    if predicted_verdict is None:
        return None
    return CF_LABEL_TO_VERDICT.get(true_label) == predicted_verdict
