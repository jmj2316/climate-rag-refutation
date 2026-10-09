"""
Figures for the paper (Nature-family style: sans-serif, thin axes, panel letters, colour-blind-safe palette,
legends outside the axes). Drawn from the stats CSVs; writes paper/figs/fig*.pdf (+png preview).
Figures are sized to the 390 pt (5.4 in) text width and included at native size.

Run: python 16_make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "figs"
OUT.mkdir(exist_ok=True)
W = 5.4

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "ps.fonttype": 42, "mathtext.default": "regular", "legend.fontsize": 7, "axes.labelcolor": "#222222",
    "text.color": "#222222", "xtick.color": "#222222", "ytick.color": "#222222", "axes.edgecolor": "#444444"})

MODELS = ["HuggingFaceH4/zephyr-7b-beta", "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
SHORT = {MODELS[0]: "Zephyr", MODELS[1]: "Qwen", MODELS[2]: "Mistral"}
ARMS = ["A_zeroshot", "B_dense_rag", "C_decompose_filtered_rag"]
ALAB = {"A_zeroshot": "A  Zero-shot", "B_dense_rag": "B  Dense RAG", "C_decompose_filtered_rag": "C  Decomposition + filter"}
ACOL = {"A_zeroshot": "#7C8896", "B_dense_rag": "#E69F00", "C_decompose_filtered_rag": "#0072B2"}
MMARK = {MODELS[0]: "o", MODELS[1]: "s", MODELS[2]: "^"}
SIG, NS, GRID, BAND = "#1F2A44", "#9AA3AD", "#E3E6EA", "#F4F5F7"
CA_COL, CB_COL = "#009E73", "#CC79A7"
VER_COL = {"supports": "#009E73", "refutes": "#D55E00", "nei": "#B9C0C8"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def head(ax, letter, title="", dx=0.0):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 5), textcoords="offset points",
                fontsize=9.5, fontweight="bold", ha="left", va="bottom")
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(dx + 11, 5.5), textcoords="offset points",
                    fontsize=7.5, ha="left", va="bottom")


def band_rows(ax, n, start=0):
    for i in range(n):
        if (i + start) % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)


def dot(ax, x, y, lo, hi, marker, color, filled=True, ms=5.2, lw=1.1, zorder=3):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=zorder)
    ax.plot(x, y, marker=marker, ms=ms, mfc=color if filled else "white", mec=color if not filled else "white",
            mew=1.0 if not filled else 0.6, ls="none", zorder=zorder + 1)


def fmt(v, d=3):
    return f"{v:+.{d}f}".replace("-", "\u2212")


# ---------------------------------------------------------------------------------------------- overview
def fig_overview():
    H = 3.85
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    GREY, DK = "#6B7280", "#374151"
    fills = {"A": ("#ECEEF1", "#7C8896"), "B": ("#FDF0D1", "#C98600"), "C": ("#D9EAF6", "#0072B2")}

    def box(x, y, w, h, text, fc="white", ec=GREY, fs=5.7, lw=0.6, weight="normal", color="#222222", rs=0.045):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}", fc=fc, ec=ec, lw=lw, zorder=2))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.18, zorder=3,
                    fontweight=weight, color=color)

    def arrow(p, q, color=GREY, lw=0.6, ms=5):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, zorder=1, shrinkA=0, shrinkB=0))

    def chip(x, y, letter, key):
        ax.add_patch(Circle((x, y), 0.095, fc=fills[key][1], ec="white", lw=0.8, zorder=4))
        ax.text(x, y, letter, ha="center", va="center", color="white", fontsize=7.5, fontweight="bold", zorder=5)

    top = H - 0.10
    for x, t in [(0.32, "Input"), (2.12, "Pipelines (run with each of 3 models)"), (3.98, "Output"), (4.95, "Evaluation")]:
        ax.text(x, top, t, ha="center", va="center", fontsize=6.8, color=GREY, fontweight="bold")

    yA, yB, yC = 3.20, 2.58, 1.30
    hA, hB, hC = 0.46, 0.46, 0.66
    c = [0.98, 1.78, 2.58]
    cw = 0.70
    xL, wL = 3.40, 0.46

    box(0.02, 0.88, 0.56, 2.62, "CLIMATE-\nFEVER\nclaim\n\n$n$ = 1,535", fc="#F7F7F5", ec=GREY, fs=6.2)
    kb_y = 1.80
    box(c[1], kb_y, cw, 0.34, "IPCC AR6\n12,704 chunks", fc="#E8F1E4", ec="#5E8C4F", fs=5.6)

    for key, yc, h in (("A", yA, hA), ("B", yB, hB), ("C", yC, hC)):
        chip(0.80, yc, key, key)
        arrow((0.58, yc), (0.70, yc), lw=0.5)
        box(xL, yc - h / 2, wL, h, "LLM\njudge", fc=fills[key][0], ec=fills[key][1], fs=6.2, weight="bold")
        arrow((xL + wL, yc), (3.96, yc), lw=0.5)

    arrow((0.90, yA), (xL, yA))
    ax.text((0.90 + xL) / 2, yA + 0.07, "claim only, no retrieval", ha="center", va="bottom", fontsize=5.6, color=GREY, style="italic")
    arrow((0.90, yB), (c[1], yB))
    box(c[1], yB - hB / 2, cw, hB, "Hybrid retrieval\ntop-3 chunks", fc=fills["B"][0], ec=fills["B"][1])
    arrow((c[1] + cw, yB), (xL, yB))
    ax.text((c[1] + cw + xL) / 2, yB + 0.07, "raw-claim query", ha="center", va="bottom", fontsize=5.6, color=GREY, style="italic")
    arrow((c[1] + cw / 2, kb_y + 0.34), (c[1] + cw / 2, yB - hB / 2), lw=0.5, color="#5E8C4F")
    arrow((0.90, yC), (c[0], yC))
    stages = ["Decompose\ninto 1\u20133\nsub-claims", "Hybrid retrieval\ntop-5 per\nsub-claim", "LLM relevance\nfilter: \u22642 chunks\nper sub-claim"]
    for i, t in enumerate(stages):
        box(c[i], yC - hC / 2, cw, hC, t, fc=fills["C"][0], ec=fills["C"][1])
    arrow((c[0] + cw, yC), (c[1], yC))
    arrow((c[1] + cw, yC), (c[2], yC))
    arrow((c[2] + cw, yC), (xL, yC))
    arrow((c[1] + cw / 2, kb_y), (c[1] + cw / 2, yC + hC / 2), lw=0.5, color="#5E8C4F")
    ax.text(c[2] + cw + 0.06, yC - hC / 2 - 0.04, "sub-claim queries, grouped context", ha="right", va="top", fontsize=5.6, color=GREY, style="italic")

    box(3.96, 0.88, 0.46, 2.62, "Verdict\n\nsupports\nrefutes\nNEI\n\n+\n\nrefutation\n(2\u20134\nsentences)", fc="#F7F7F5", ec=GREY, fs=5.6)
    ev = [("Classification", "accuracy,\nmacro-F1 vs.\ngold label", 2.70), ("LLM judge", "Llama-3.1-8B:\nlogical validity,\nhedging", 1.89),
          ("RAGAS-style", "faithfulness,\ncontext precision\n(B, C only)", 1.08)]
    for t, s_, y0 in ev:
        h = 0.72
        ax.add_patch(FancyBboxPatch((4.56, y0), 0.82, h, boxstyle="round,pad=0,rounding_size=0.045", fc="white", ec=GREY, lw=0.6, zorder=2))
        ax.text(4.56 + 0.41, y0 + h - 0.11, t, ha="center", va="center", fontsize=6.0, fontweight="bold", zorder=3)
        ax.text(4.56 + 0.41, y0 + 0.28, s_, ha="center", va="center", fontsize=5.4, linespacing=1.15, zorder=3)
        arrow((4.42, 2.19), (4.56, y0 + h / 2), lw=0.5)

    # ablation strip
    ax.plot([0.02, W - 0.02], [0.68, 0.68], color=GRID, lw=0.8)
    ax.text(0.02, 0.30, "Ablation\narms", ha="left", va="center", fontsize=6.4, fontweight="bold", color=DK, linespacing=1.1)
    chips = [("B", "raw claim,\ntop-3 chunks", "#FDF0D1", "#C98600"), ("E3, E2", "raw claim, chunks\nmatched to E1 / C", "#FBE3AE", "#C98600"),
             ("E1", "sub-claims,\nno filter", "#B9D8EE", "#0072B2"), ("C", "sub-claims,\nLLM filter", "#8BBDE0", "#0072B2")]
    wchip, step, x00 = 0.76, 1.25, 0.72
    for i, (lab, t, fc, ec) in enumerate(chips):
        x0 = x00 + i * step
        box(x0, 0.06, wchip, 0.50, "", fc=fc, ec=ec)
        ax.text(x0 + wchip / 2, 0.45, lab, ha="center", va="center", fontsize=6.2, fontweight="bold", color=DK, zorder=3)
        ax.text(x0 + wchip / 2, 0.23, t, ha="center", va="center", fontsize=5.1, linespacing=1.12, zorder=3)
    for i, lab in enumerate(["context size", "decomposition", "filter"]):
        x0 = x00 + i * step + wchip
        x1 = x00 + (i + 1) * step
        arrow((x0 + 0.015, 0.28), (x1 - 0.015, 0.28), lw=0.6, color=DK)
        ax.text((x0 + x1) / 2, 0.33, lab, ha="center", va="bottom", fontsize=5.0, color=GREY, style="italic")
    save(fig, "fig_overview")


# ---------------------------------------------------------------------------------------------- class
def fig_class():
    ci = pd.read_csv(RES / "stats_ci_full.csv")
    base = json.load(open(RES / "stats_baselines_full.json"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.45), sharey=True)
    panels = [("accuracy", "acc_lo", "acc_hi", base["majority_acc"], "Accuracy", (0.30, 0.54)),
              ("macro_f1", "f1_lo", "f1_hi", base["majority_macro_f1"], "Macro-F1", (0.17, 0.52))]
    for ax, (key, lo, hi, bl, title, xl) in zip(axes, panels):
        band_rows(ax, 3)
        for i, m in enumerate(MODELS):
            for j, a in enumerate(ARMS):
                r = ci[(ci.model == m) & (ci.arm == a)].iloc[0]
                dot(ax, r[key], i + (j - 1) * 0.25, r[lo], r[hi], MMARK[m], ACOL[a], ms=5.4)
        ax.axvline(bl, color="#333333", ls=(0, (3, 2)), lw=0.7, zorder=2)
        ax.text(bl + (xl[1] - xl[0]) * 0.012, 0.5, "majority\nclass", fontsize=6, color="#333333", ha="left", va="center",
                linespacing=1.05, zorder=4, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.8))
        ax.set_xlim(*xl)
        ax.set_ylim(2.5, -0.5)
        ax.set_xlabel(title)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels([SHORT[m] for m in MODELS])
    head(axes[0], "a", "Accuracy", dx=-2)
    head(axes[1], "b", "Macro-F1", dx=-2)
    h = [plt.Line2D([], [], marker="o", ms=5.5, color=ACOL[a], ls="", label=ALAB[a]) for a in ARMS]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), columnspacing=1.6, handletextpad=0.2)
    fig.subplots_adjust(wspace=0.08, left=0.1, right=0.985, bottom=0.17, top=0.86)
    save(fig, "fig_class")


# ---------------------------------------------------------------------------------------------- ablation
def fig_ablation():
    ci = pd.read_csv(RES / "ablation_ci.csv")
    pr = pd.read_csv(RES / "ablation_paired.csv")
    av = pd.read_csv(RES / "ablation_paired_modelavg.csv").set_index("contrast")
    fig = plt.figure(figsize=(W, 3.3))
    axa = fig.add_axes([0.07, 0.2, 0.31, 0.68])
    axb = fig.add_axes([0.46, 0.2, 0.30, 0.68])
    axc = fig.add_axes([0.835, 0.2, 0.155, 0.68])

    # (a) ladder of macro-F1
    order = ["A_zeroshot", "B_dense_rag", "E3_raw_matchE1", "E2_raw_matchC", "E1_decomp_nofilter", "C_decompose_filtered_rag"]
    pos = dict(zip(order, [0, 1.25, 2.1, 2.95, 4.2, 5.05]))
    names = ["A", "B", "E3", "E2", "E1", "C"]
    lo_y, hi_y = 0.225, 0.52
    axa.axvspan(0.75, 3.4, color="#FDF0D1", lw=0, zorder=0, alpha=0.7)
    axa.axvspan(3.75, 5.5, color="#D9EAF6", lw=0, zorder=0, alpha=0.8)
    tone = {MODELS[0]: "#2B3445", MODELS[1]: "#6F7C93", MODELS[2]: "#9BA8C0"}
    for m in MODELS:
        d = ci[ci.model == m].set_index("arm")
        axa.plot([pos[a] for a in order], [d.loc[a, "macro_f1"] for a in order], color=tone[m], lw=0.8, zorder=2)
        for a in order:
            r = d.loc[a]
            axa.plot([pos[a]] * 2, [r.f1_lo, r.f1_hi], color=tone[m], lw=1.0, zorder=2)
            axa.plot(pos[a], r.macro_f1, MMARK[m], ms=4.6, mfc=tone[m], mec="white", mew=0.5, zorder=3)
        axa.text(5.62, d.loc["C_decompose_filtered_rag", "macro_f1"], SHORT[m], fontsize=6.4, va="center", color=tone[m] if m != MODELS[2] else "#7B88A0")
    axa.set_xticks([pos[a] for a in order])
    axa.set_xticklabels(names)
    axa.set_xlim(-0.45, 6.55)
    axa.set_ylim(lo_y, hi_y)
    axa.set_ylabel("Macro-F1")
    yl = lo_y + 0.009
    axa.text(0, yl, "no\nretrieval", ha="center", va="bottom", fontsize=5.4, color="#555", linespacing=1.05)
    axa.text(2.07, yl, "raw-claim\nquery", ha="center", va="bottom", fontsize=5.6, color="#8A5A00", linespacing=1.05)
    axa.text(4.62, yl, "sub-claim\nquery", ha="center", va="bottom", fontsize=5.6, color="#004A75", linespacing=1.05)
    axa.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    head(axa, "a", "Pipeline ladder", dx=-2)

    # (b) forest of the telescoping contrasts
    steps = [("context size (B to E3)", "Context  (E3 \u2212 B)"), ("decomposition", "Decomposition  (E1 \u2212 E3)"),
             ("filter", "Filter  (C \u2212 E1)"), ("C vs B", "Total  (C \u2212 B)")]
    y = 0
    ticks, tlabs = [], []
    for si, (cname, lab) in enumerate(steps):
        y_head = y
        y += 1.15
        y0 = y
        for m in MODELS:
            r = pr[(pr.model == m) & (pr.contrast == cname)].iloc[0]
            sig = r.p_boot_f1_holm < 0.05
            dot(axb, r.delta_f1, y, r.d_lo, r.d_hi, MMARK[m], SIG if sig else NS, filled=sig, ms=4.4, lw=1.0)
            ticks.append(y)
            tlabs.append(SHORT[m])
            y += 1
        a = av.loc[cname]
        sig = a.lo > 0 or a.hi < 0
        col = "#C0392B" if sig else NS
        axb.plot([a.lo, a.hi], [y, y], color=col, lw=1.2, zorder=3)
        axb.plot(a.delta_f1_mean, y, "D", ms=4.2, mfc=col if sig else "white", mec=col, mew=0.9, zorder=4)
        if si % 2 == 0:
            axb.axhspan(y_head - 0.5, y + 0.55, color=BAND, lw=0, zorder=0)
        axb.text(0.012, y_head + 0.05, lab, transform=axb.get_yaxis_transform(), ha="left", va="center", fontsize=6.4, fontweight="bold")
        y += 1.55
    axb.set_ylim(y - 1.05, -0.55)
    axb.axvline(0, color="#333", lw=0.7, zorder=1)
    axb.set_xlim(-0.045, 0.165)
    axb.set_xticks([0, 0.05, 0.10, 0.15])
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    axb.set_xlabel("\u0394 Macro-F1")
    axb.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    head(axb, "b", "Where the gain comes from", dx=-2)

    # (c) dominant verdict shift E3 -> E1
    e3 = ci[ci.arm == "E3_raw_matchE1"].set_index("model")
    e1 = ci[ci.arm == "E1_decomp_nofilter"].set_index("model")
    cols = {"frac_nei": "NEI", "frac_refutes": "refutes", "frac_supports": "supports"}
    for i, m in enumerate(MODELS):
        k = e3.loc[m, list(cols)].astype(float).idxmax()
        v0, v1 = 100 * e3.loc[m, k], 100 * e1.loc[m, k]
        axc.annotate("", xy=(v1, i), xytext=(v0, i), arrowprops=dict(arrowstyle="-|>", color="#666", lw=0.9, mutation_scale=6, shrinkA=3.6, shrinkB=3.6))
        axc.plot(v0, i, "o", ms=5.4, mfc="#E69F00", mec="white", mew=0.6, zorder=3)
        axc.plot(v1, i, "o", ms=5.4, mfc="#0072B2", mec="white", mew=0.6, zorder=3)
        axc.text(v0, i + 0.22, f"{v0:.0f}", ha="center", va="top", fontsize=6.0, color="#8A5A00")
        axc.text(v1, i + 0.22, f"{v1:.0f}", ha="center", va="top", fontsize=6.0, color="#004A75")
        axc.text(31, i - 0.30, f"{SHORT[m]} \u00b7 {cols[k]}", ha="left", va="center", fontsize=6.3, color="#333")
    axc.set_yticks([])
    axc.set_ylim(2.65, -0.65)
    axc.set_xlim(30, 100)
    axc.set_xticks([40, 60, 80, 100])
    axc.set_xlabel("Share of claims (%)")
    axc.spines["left"].set_visible(False)
    axc.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    head(axc, "c", "Verdict shift", dx=-2)

    h = [plt.Line2D([], [], marker=MMARK[m], ms=5, color=SIG, ls="", label=SHORT[m]) for m in MODELS]
    h += [plt.Line2D([], [], marker="D", ms=4.5, color="#C0392B", ls="", label="Mean of models"),
          plt.Line2D([], [], marker="o", ms=5, color=SIG, ls="", label="Holm $p$ < 0.05"),
          plt.Line2D([], [], marker="o", ms=5, color=NS, mfc="white", ls="", label="n.s."),
          plt.Line2D([], [], marker="o", ms=5, color="#E69F00", ls="", label="E3 (raw query)"),
          plt.Line2D([], [], marker="o", ms=5, color="#0072B2", ls="", label="E1 (sub-claims)")]
    fig.legend(handles=h, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.025), columnspacing=0.85, handletextpad=0.1, fontsize=6.2)
    save(fig, "fig_ablation")


# ---------------------------------------------------------------------------------------------- paired
def fig_paired():
    d = pd.read_csv(RES / "stats_paired_full.csv")
    cons = [("B_dense_rag - A_zeroshot", "B vs A", "dense RAG vs zero-shot"),
            ("C_decompose_filtered_rag - B_dense_rag", "C vs B", "decomposition + filter vs dense RAG"),
            ("C_decompose_filtered_rag - A_zeroshot", "C vs A", "decomposition + filter vs zero-shot")]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    fig.subplots_adjust(left=0.1, right=0.66, bottom=0.17, top=0.9)
    y = 0.0
    ticks, tl = [], []
    for k, (c, t1, t2) in enumerate(cons):
        yh = y
        y += 1.0
        y0 = y
        for m in MODELS:
            r = d[(d.model == m) & (d.contrast == c)].iloc[0]
            sig = r.p_boot_f1_holm < 0.05
            dot(ax, r.delta_f1, y, r.d_lo, r.d_hi, MMARK[m], SIG if sig else NS, filled=sig, ms=5.0)
            ax.text(1.04, y, f"{fmt(r.delta_f1)}  [{fmt(r.d_lo)}, {fmt(r.d_hi)}]", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=6.6, color="#222" if sig else "#777")
            ticks.append(y)
            tl.append(SHORT[m])
            y += 1
        if k % 2 == 0:
            ax.axhspan(yh - 0.4, y - 0.45, color=BAND, lw=0, zorder=0)
        tt = ax.text(0.008, yh + 0.08, t1, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.0, fontweight="bold")
        ax.annotate(t2, xy=(1, 0.5), xycoords=tt, xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=6.2, color="#666")
        if k == 0:
            ax.text(1.04, yh + 0.08, "\u0394 Macro-F1  [95% CI]", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.4,
                    color="#666", fontweight="bold")
        y += 0.35
    ax.set_ylim(y - 0.55, -0.45)
    ax.axvline(0, color="#333", lw=0.7, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tl, fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.145, 0.165)
    ax.set_xlabel("\u0394 Macro-F1 (paired bootstrap, 95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    h = [plt.Line2D([], [], marker="o", ms=5, color=SIG, ls="", label="Holm $p$ < 0.05"),
         plt.Line2D([], [], marker="o", ms=5, color=NS, mfc="white", ls="", label="not significant")]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.38, 1.03), columnspacing=1.4, handletextpad=0.2)
    save(fig, "fig_paired")


# ---------------------------------------------------------------------------------------------- verdicts
def fig_verdicts():
    d = pd.read_csv(RES / "stats_verdict_distribution_full.csv")
    gt = {"supports": 0.426, "refutes": 0.165, "nei": 0.409}
    rows, y = [], 0
    rows.append(("Ground truth", None, gt, y))
    y = 1.35
    for a in ARMS:
        for m in MODELS:
            r = d[(d.arm == a) & (d.model == m)].iloc[0]
            rows.append((SHORT[m], a, {k: r[k] for k in gt}, y))
            y += 1
        y += 0.35
    fig, ax = plt.subplots(figsize=(4.6, 3.35))
    for lab, a, vals, yy in rows:
        left = 0
        for k in ("supports", "refutes", "nei"):
            v = vals[k]
            ax.barh(yy, v, left=left, height=0.74, color=VER_COL[k], edgecolor="white", linewidth=0.6, zorder=3)
            if v >= 0.075:
                ax.text(left + v / 2, yy, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.3,
                        color="#222" if k == "nei" else "white", zorder=4)
            left += v
    ax.set_yticks([r[3] for r in rows])
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.9)
    ax.set_ylim(rows[-1][3] + 0.6, -0.65)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("Share of claims (%)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    pos = {}
    for lab, a, vals, yy in rows:
        if a:
            pos.setdefault(a, []).append(yy)
    glab = {ARMS[0]: "A\nZero-shot", ARMS[1]: "B\nDense RAG", ARMS[2]: "C\nDecomposition\n+ filter"}
    for a, ys in pos.items():
        ax.annotate(glab[a], xy=(0.0, np.mean(ys)), xycoords=("figure fraction", "data"), xytext=(2, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=6.8, fontweight="bold", color="#222", linespacing=1.15)
    fig.subplots_adjust(left=0.30, right=0.985, bottom=0.12, top=0.92)
    h = [plt.Line2D([], [], marker="s", ms=6, color=VER_COL[k], ls="", label=l) for k, l in
         [("supports", "supports"), ("refutes", "refutes"), ("nei", "not enough info")]]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.62, 1.03), columnspacing=1.3, handletextpad=0.2)
    save(fig, "fig_verdicts")


# ---------------------------------------------------------------------------------------------- cards
def fig_cards():
    d = pd.read_csv(RES / "stats_cards_deltas_full.csv")
    names = {"0_not_denial": "Not denial", "1_not_happening": "It's not happening", "2_not_us": "It's not us",
             "3_not_bad": "It's not bad", "4_solutions_wont_work": "Solutions won't work", "5_science_unreliable": "Science unreliable"}
    fig, ax = plt.subplots(figsize=(W, 2.7))
    band_rows(ax, len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        for off, k, col in ((-0.17, "CA", CA_COL), (0.17, "CB", CB_COL)):
            v, lo, hi = 100 * r[f"d_{k}"], 100 * r[f"lo_{k}"], 100 * r[f"hi_{k}"]
            dot(ax, v, i + off, lo, hi, "o", col, filled=(lo > 0 or hi < 0), ms=5.0)
    ax.axvline(0, color="#333", lw=0.7, zorder=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{names[c]}\n($n$ = {int(n)})" for c, n in zip(d.cards, d.n_claims)], fontsize=6.8, linespacing=1.1)
    ax.set_ylim(len(d) - 0.5, -0.5)
    ax.set_xlim(-6, 17)
    ax.set_xlabel("Accuracy difference, mean over models (percentage points; 95% CI)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    h = [plt.Line2D([], [], marker="o", ms=5, color=CA_COL, ls="", label="C vs A"),
         plt.Line2D([], [], marker="o", ms=5, color=CB_COL, ls="", label="C vs B"),
         plt.Line2D([], [], marker="o", ms=5, color="#444", ls="", label="CI excludes 0"),
         plt.Line2D([], [], marker="o", ms=5, color="#444", mfc="white", ls="", label="CI includes 0")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03), columnspacing=1.4, handletextpad=0.2)
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.17, top=0.9)
    save(fig, "fig_cards")


# ---------------------------------------------------------------------------------------------- quality
def fig_quality():
    m = pd.read_csv(RES / "master_results_table_full.csv")
    amap = {"A: Zero-shot": ARMS[0], "B: Dense RAG": ARMS[1], "C: Decompose+Filtered RAG": ARMS[2]}
    m["arm"] = m.arm_name.map(amap)
    h = pd.read_csv(RES / "stats_length_hedge_full.csv").drop(columns="hedge_rate")
    hn = pd.read_csv(RES / "hedge_nosub_full.csv")
    hn["model"] = hn.model.map({v: k for k, v in SHORT.items()})
    hn["arm"] = hn.arm.map({"A": ARMS[0], "B": ARMS[1], "C": ARMS[2]})
    h = h.merge(hn[["model", "arm", "hedge_nosub"]], on=["model", "arm"]).rename(columns={"hedge_nosub": "hedge_rate"})
    ci = pd.read_csv(RES / "stats_ci_full.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.35))

    def draw(ax, getter, arms):
        for mod in MODELS:
            pts = [getter(mod, a) for a in arms]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#B8BEC6", lw=0.7, zorder=2,
                    ls={MODELS[0]: "-", MODELS[1]: (0, (3, 1.5)), MODELS[2]: (0, (1, 1.2))}[mod])
            for a, p in zip(arms, pts):
                ax.plot(p[0], p[1], MMARK[mod], ms=5.6, mfc=ACOL[a], mec="white", mew=0.6, zorder=3)
        ax.grid(color=GRID, lw=0.5, zorder=0)

    draw(axes[0], lambda mod, a: (m[(m.model == mod) & (m.arm == a)].logical_validity.iloc[0],
                                  ci[(ci.model == mod) & (ci.arm == a)].macro_f1.iloc[0]), ARMS)
    axes[0].set_xlabel("Judge logical validity (1\u20135)")
    axes[0].set_ylabel("Macro-F1")
    draw(axes[1], lambda mod, a: (h[(h.model == mod) & (h.arm == a)].hedge_rate.iloc[0] * 100,
                                  h[(h.model == mod) & (h.arm == a)].lv.iloc[0]), ARMS)
    axes[1].set_xlabel("Hedging rate (%)")
    axes[1].set_ylabel("Judge logical validity")
    draw(axes[2], lambda mod, a: (m[(m.model == mod) & (m.arm == a)].context_precision.iloc[0],
                                  m[(m.model == mod) & (m.arm == a)].faithfulness.iloc[0]), ARMS[1:])
    axes[2].set_xlim(0.185, 0.235)
    axes[2].set_ylim(0.2, 0.42)
    axes[2].set_xlabel("Context precision")
    axes[2].set_ylabel("Faithfulness")
    for ax, l, t in zip(axes, "abc", ["Judge vs. truth", "Hedging vs. judge", "Retrieval vs. faithfulness"]):
        head(ax, l, t, dx=-2)
    hs = [plt.Line2D([], [], marker="o", ms=5.5, color=ACOL[a], ls="", label=ALAB[a].split("  ")[0] + " " + ALAB[a].split("  ")[1]) for a in ARMS]
    hs += [plt.Line2D([], [], marker=MMARK[mod], ms=5, color="#444", mfc="#444", ls="", label=SHORT[mod]) for mod in MODELS]
    fig.legend(handles=hs, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.03), columnspacing=1.0, handletextpad=0.1, fontsize=6.6)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.88, wspace=0.42)
    save(fig, "fig_quality")


if __name__ == "__main__":
    for f in (fig_overview, fig_class, fig_ablation, fig_paired, fig_verdicts, fig_cards, fig_quality):
        f()
    print("saved", sorted(p.name for p in OUT.glob("*.pdf")))
