"""
Re-parse refutation text from raw_response in existing Model A/B/C outputs after the
parser gained a fallback for responses that omit the "Refutation:" label. Verdicts are
unchanged (asserted). Originals are copied to results/_backup_preparse/ once.

Run: python 13_reparse_refutations.py   (stdlib only)
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_rag_response import parse_verdict_refutation

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
BK = RES / "_backup_preparse"


def main():
    for d in ["model_a", "model_b", "model_c"]:
        for path in sorted((RES / d).glob("*_full.jsonl")):
            bk = BK / d / path.name
            if not bk.exists():
                bk.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, bk)
            rows = [json.loads(l) for l in open(bk)]
            n_before = sum(bool(r.get("refutation")) for r in rows)
            for r in rows:
                v, ref = parse_verdict_refutation(r["raw_response"])
                assert v == r["verdict"], (path.name, r["claim_id"], v, r["verdict"])
                r["refutation"] = ref
            n_after = sum(bool(r.get("refutation")) for r in rows)
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{d}/{path.name}: refutation present {n_before} -> {n_after} / {len(rows)}")


if __name__ == "__main__":
    main()
