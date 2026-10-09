"""
Download IPCC AR6 SPM + Technical Summary PDFs (WG1/WG2/WG3 + Synthesis Report).
Run (has internet): conda activate climate_rag && python 01_download_ipcc.py

Note: WG1 has no standalone Technical Summary PDF under downloads/report/ (only
a chapter-HTML page and the 254MB full report), so WG1 TS is skipped -- SPM
coverage for WG1 plus TS for WG2/WG3 plus the Synthesis Report gives broad
enough physical+impacts+mitigation coverage for the KB.
"""
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data/ipcc_pdfs"

SOURCES = {
    "wg1_spm":  "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf",
    "wg2_spm":  "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_SummaryForPolicymakers.pdf",
    "wg2_ts":   "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_TechnicalSummary.pdf",
    "wg3_spm":  "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_SPM.pdf",
    "wg3_ts":   "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_TechnicalSummary.pdf",
    "syr_spm":  "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_SPM.pdf",
    "syr_full": "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf",
    # full chapter text: much broader/deeper coverage (regional chapters, specific
    # findings/studies) needed for CLIMATE-FEVER's diverse, often localized claims,
    # which the high-level SPM/TS documents above don't cover well.
    "wg1_full": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_FullReport.pdf",
    "wg2_full": "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_FullReport.pdf",
    "wg3_full": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_FullReport.pdf",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        out_path = OUT_DIR / f"{name}.pdf"
        if out_path.exists() and out_path.stat().st_size > 10_000:
            print(f"SKIP {name} (already downloaded, {out_path.stat().st_size} bytes)")
            continue
        print(f"Downloading {name} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(out_path, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"  -> {out_path} ({out_path.stat().st_size} bytes)")
    print("\nDone.")


if __name__ == "__main__":
    main()
