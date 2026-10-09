"""
Generate LaTeX tables for the paper from the stats CSVs (so no numbers are transcribed by hand).
Writes paper/tab_*.tex.

Run: conda run -n climate_rag python 15_make_tables.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper"
SHORT = {"HuggingFaceH4/zephyr-7b-beta": "Zephyr", "Qwen/Qwen2.5-7B-Instruct": "Qwen",
         "mistralai/Mistral-7B-Instruct-v0.3": "Mistral"}
MODELS = list(SHORT)
ARM = {"A_zeroshot": r"A: \arma", "B_dense_rag": r"B: \armb", "C_decompose_filtered_rag": r"C: \armc"}


def f3(x):
    return f"{x:.3f}"


def ci(v, lo, hi):
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name, s):
    (OUT / f"{name}.tex").write_text(s)


def tab_class():
    ci_df = pd.read_csv(RES / "stats_ci_full.csv")
    base = json.load(open(RES / "stats_baselines_full.json"))
    L = [r"\begin{table}[!tbp]", r"\centering", r"\small",
         r"\caption{Veracity classification on 1{,}535 CLIMATE-FEVER claims (three-way; unparsed responses counted as errors). "
         r"Brackets: 95\% percentile bootstrap CI over claims (10{,}000 resamples). $\dagger$: accuracy significantly "
         r"\emph{above} the majority-class baseline (exact binomial test, $p<0.05$).}",
         r"\label{tab:class}", r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
         r"Pipeline & Model & Accuracy & Macro-F1 \\", r"\midrule",
         rf"Majority class & -- & {base['majority_acc']:.3f} & {base['majority_macro_f1']:.3f} \\", r"\midrule"]
    for a in ARM:
        for m in MODELS:
            r = ci_df[(ci_df.model == m) & (ci_df.arm == a)].iloc[0]
            dag = r"$^\dagger$" if (r.acc_vs_majority_p < 0.05 and r.acc_minus_majority > 0) else ""
            L.append(rf"{ARM[a]} & {SHORT[m]} & {ci(r.accuracy, r.acc_lo, r.acc_hi)}{dag} & {ci(r.macro_f1, r.f1_lo, r.f1_hi)} \\")
        if a != "C_decompose_filtered_rag":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_class", "\n".join(L) + "\n")


def tab_paired():
    d = pd.read_csv(RES / "stats_paired_full.csv")
    name = {"B_dense_rag - A_zeroshot": "B$-$A", "C_decompose_filtered_rag - B_dense_rag": "C$-$B",
            "C_decompose_filtered_rag - A_zeroshot": "C$-$A"}
    order = ["B_dense_rag - A_zeroshot", "C_decompose_filtered_rag - B_dense_rag", "C_decompose_filtered_rag - A_zeroshot"]
    L = [r"\begin{table}[!tbp]", r"\centering", r"\small",
         r"\caption{Paired comparisons between pipelines. $\Delta$Macro-F1 with 95\% paired-bootstrap CI and "
         r"two-sided bootstrap $p$; $\Delta$Acc in percentage points with exact McNemar $p$. Both $p$-values are "
         r"Holm-adjusted over the nine comparisons. Bold: adjusted $p<0.05$.}",
         r"\label{tab:paired}", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"Model & Contrast & $\Delta$Macro-F1 [95\% CI] & $p_{\mathrm{Holm}}$ & $\Delta$Acc (pp) & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for m in MODELS:
        for c in order:
            r = d[(d.model == m) & (d.contrast == c)].iloc[0]
            f1s = f"{r.delta_f1:+.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}]"
            pf, pa = pfmt(r.p_boot_f1_holm), pfmt(r.p_mcnemar_holm)
            if r.p_boot_f1_holm < 0.05:
                f1s, pf = rf"\textbf{{{f1s}}}", rf"\textbf{{{pf}}}"
            acc = f"{100 * r.delta_acc:+.1f}"
            if r.p_mcnemar_holm < 0.05:
                acc, pa = rf"\textbf{{{acc}}}", rf"\textbf{{{pa}}}"
            L.append(rf"{SHORT[m]} & {name[c]} & {f1s} & {pf} & {acc} & {pa} \\")
        if m != MODELS[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_paired", "\n".join(L) + "\n")


def tab_gen():
    m = pd.read_csv(RES / "master_results_table_full.csv")
    arm_map = {"A: Zero-shot": "A_zeroshot", "B: Dense RAG": "B_dense_rag", "C: Decompose+Filtered RAG": "C_decompose_filtered_rag"}
    m["arm"] = m["arm_name"].map(arm_map)
    L = [r"\begin{table}[!tbp]", r"\centering", r"\small",
         r"\caption{Generation-quality metrics (means over 1{,}535 claims; every refutation scored by an independent "
         r"Llama-3.1-8B judge). Faith.\ = faithfulness, AR = answer relevance, CP = context precision (all 0--1); "
         r"LV = logical validity, SR = scientific rigor (1--5). Faith., AR and CP are undefined without retrieval.}",
         r"\label{tab:gen}", r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"Pipeline & Model & Faith. & AR & CP & LV & SR \\", r"\midrule"]
    for a in ARM:
        for mod in MODELS:
            r = m[(m.model == mod) & (m.arm == a)].iloc[0]
            def g(x):
                return "--" if pd.isna(x) else f"{x:.3f}"
            L.append(rf"{ARM[a]} & {SHORT[mod]} & {g(r.faithfulness)} & {g(r.answer_relevance)} & {g(r.context_precision)} & {r.logical_validity:.2f} & {r.scientific_rigor:.2f} \\")
        if a != "C_decompose_filtered_rag":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_gen", "\n".join(L) + "\n")


def tab_scores():
    d = pd.read_csv(RES / "stats_scores_paired_full.csv")
    cols = ["logical_validity", "scientific_rigor", "faithfulness", "answer_relevance", "context_precision"]
    hdr = ["$\\Delta$LV", "$\\Delta$SR", "$\\Delta$Faith.", "$\\Delta$AR", "$\\Delta$CP"]
    L = [r"\begin{table}[!tbp]", r"\centering", r"\small",
         r"\caption{Paired mean differences in judged and RAGAS-style metrics (Wilcoxon signed-rank, Holm-adjusted over "
         r"27 tests). Bold: adjusted $p<0.05$. Judge metrics are defined for all pipelines; faithfulness, answer relevance "
         r"and context precision only for B and C.}",
         r"\label{tab:scores}", r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         "Model & Contrast & " + " & ".join(hdr) + r" \\", r"\midrule"]
    for mod in MODELS:
        for c, label in [("B_dense_rag - A_zeroshot", "B$-$A"), ("C_decompose_filtered_rag - B_dense_rag", "C$-$B"),
                         ("C_decompose_filtered_rag - A_zeroshot", "C$-$A")]:
            cells = []
            for met in cols:
                r = d[(d.model == mod) & (d.contrast == c) & (d.metric == met)]
                if r.empty:
                    cells.append("--")
                    continue
                r = r.iloc[0]
                s = f"{r.mean_diff:+.2f}" if met in ("logical_validity", "scientific_rigor") else f"{r.mean_diff:+.3f}"
                cells.append(rf"\textbf{{{s}}}" if r.p_holm < 0.05 else s)
            L.append(rf"{SHORT[mod]} & {label} & " + " & ".join(cells) + r" \\")
        if mod != MODELS[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_scores", "\n".join(L) + "\n")


def tab_cards():
    d = pd.read_csv(RES / "stats_cards_deltas_full.csv")
    names = {"0_not_denial": "Not denial", "1_not_happening": "It's not happening", "2_not_us": "It's not us",
             "3_not_bad": "It's not bad", "4_solutions_wont_work": "Solutions won't work",
             "5_science_unreliable": "Science is unreliable"}

    def cell(v, lo, hi):
        s = f"{100 * v:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        return rf"\textbf{{{s}}}" if (lo > 0 or hi < 0) else s

    L = [r"\begin{table}[!tbp]", r"\centering", r"\small",
         r"\caption{Accuracy by CARDS category, averaged over the three models, with paired differences in percentage "
         r"points (95\% bootstrap CI over claims). Bold: CI excludes zero (not corrected for multiplicity). "
         r"CARDS labels are machine-assigned (63.7\% validated accuracy).}",
         r"\label{tab:cards}", r"\begin{tabular}{@{}lrcccll@{}}", r"\toprule",
         r"Category & $n$ & A & B & C & C$-$A (pp) & C$-$B (pp) \\", r"\midrule"]
    for _, r in d.iterrows():
        L.append(rf"{names[r.cards]} & {int(r.n_claims)} & {r.acc_A:.3f} & {r.acc_B:.3f} & {r.acc_C:.3f} & "
                 rf"{cell(r.d_CA, r.lo_CA, r.hi_CA)} & {cell(r.d_CB, r.lo_CB, r.hi_CB)} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_cards", "\n".join(L) + "\n")


def tab_verdicts():
    d = pd.read_csv(RES / "stats_verdict_distribution_full.csv")
    L = [r"\begin{table}[!tbp]", r"\centering", r"\small",
         r"\caption{Share of predicted verdicts (unparsed responses omitted, $\le$1.2\%). Ground truth in the 1{,}535 claims: "
         r"\textsc{supports} 0.426, \textsc{refutes} 0.165, \textsc{nei} 0.409.}",
         r"\label{tab:verdicts}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Pipeline & Model & \textsc{supports} & \textsc{refutes} & \textsc{nei} \\", r"\midrule"]
    for a in ARM:
        for mod in MODELS:
            r = d[(d.arm == a) & (d.model == mod)].iloc[0]
            L.append(rf"{ARM[a]} & {SHORT[mod]} & {r.supports:.3f} & {r.refutes:.3f} & {r.nei:.3f} \\")
        if a != "C_decompose_filtered_rag":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_verdicts", "\n".join(L) + "\n")


ABL_ARMS = [("A_zeroshot", r"A: \arma"), ("B_dense_rag", r"B: \armb"), ("E3_raw_matchE1", r"E3: raw, $n_{E1}$ chk."),
            ("E2_raw_matchC", r"E2: raw, $n_C$ chk."), ("E1_decomp_nofilter", r"E1: sub-claims, no filter"),
            ("C_decompose_filtered_rag", r"C: \armc")]


def tab_ablation():
    d = pd.read_csv(RES / "ablation_ci.csv")
    L = [r"\begin{table}[!tbp]", r"\centering", r"\footnotesize",
         r"\caption{Ablation ladder between dense RAG (B) and decomposition with filtering (C), 1{,}535 claims per cell. "
         r"E3 and E2 use the raw claim as query but give the model as many chunks (chk., 250-word truncation, B prompt) as E1 ($n_{E1}$) and C "
         r"($n_C$) receive; E1 keeps C's sub-claims and the two highest-scoring chunks per sub-claim without the LLM relevance filter. "
         r"Brackets: 95\% bootstrap CI.}",
         r"\label{tab:ablation}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Model & Arm & Accuracy & Macro-F1 & \textsc{nei} share \\", r"\midrule"]
    for m in MODELS:
        for a, lab in ABL_ARMS:
            r = d[(d.model == m) & (d.arm == a)]
            if r.empty:
                continue
            r = r.iloc[0]
            L.append(rf"{SHORT[m]} & {lab} & {ci(r.accuracy, r.acc_lo, r.acc_hi)} & {ci(r.macro_f1, r.f1_lo, r.f1_hi)} & {r.frac_nei:.3f} \\")
        if m != MODELS[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_ablation", "\n".join(L) + "\n")


def tab_ablation_paired():
    d = pd.read_csv(RES / "ablation_paired.csv")
    avg = pd.read_csv(RES / "ablation_paired_modelavg.csv").set_index("contrast")
    names = {"context size (B to E3)": r"E3$-$B (context)", "context size (B to E2)": r"E2$-$B (context)",
             "decomposition": r"E1$-$E3 (decomp.)", "filter": r"C$-$E1 (filter)",
             "decomp.+filter vs raw": r"C$-$E2 (both)", "C vs B": r"C$-$B (complete)"}
    L = [r"\begin{table}[!tbp]", r"\centering", r"\small",
         r"\caption{Paired contrasts along the ablation ladder: $\Delta$Macro-F1 with 95\% paired-bootstrap CI and Holm-adjusted bootstrap $p$ "
         r"(adjusted over all contrasts and models); $\Delta$Acc in percentage points. Bold: adjusted $p<0.05$. "
         r"The last block averages the three models' $\Delta$Macro-F1.}",
         r"\label{tab:ablation_paired}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Model & Contrast & $\Delta$Macro-F1 [95\% CI] & $p_{\mathrm{Holm}}$ & $\Delta$Acc (pp) \\", r"\midrule"]
    for m in MODELS:
        for k, lab in names.items():
            r = d[(d.model == m) & (d.contrast == k)]
            if r.empty:
                continue
            r = r.iloc[0]
            f1s = f"{r.delta_f1:+.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}]"
            pf = pfmt(r.p_boot_f1_holm)
            if r.p_boot_f1_holm < 0.05:
                f1s, pf = rf"\textbf{{{f1s}}}", rf"\textbf{{{pf}}}"
            L.append(rf"{SHORT[m]} & {lab} & {f1s} & {pf} & {100 * r.delta_acc:+.1f} \\")
        L.append(r"\midrule")
    for k, lab in names.items():
        if k in avg.index:
            r = avg.loc[k]
            L.append(rf"Mean & {lab} & {r.delta_f1_mean:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] & -- & -- \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_ablation_paired", "\n".join(L) + "\n")


def tab_format():
    d = pd.read_csv(RES / "format_counts_full.csv")
    h = pd.read_csv(RES / "hedge_nosub_full.csv")
    L = [r"\begin{table}[!tbp]", r"\centering", r"\small",
         r"\caption{Output format of the 1{,}535 responses per configuration. Labeled: a ``Refutation:'' field is present. "
         r"Prose: no label, plain explanatory text (fallback). Scaffold: no label, and the text echoes the prompt's "
         r"``Sub-claim:''/``Excerpt'' scaffolding (fallback). Multi: the verdict line names more than one verdict. "
         r"Hedging: share of refutations matching the hedging pattern, excluding the token ``sub-claim''.}",
         r"\label{tab:format}", r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"Model & Pipeline & Labeled & Prose & Scaffold & Multi & Hedging \\", r"\midrule"]
    for m in ["Mistral", "Zephyr", "Qwen"]:
        for a in "ABC":
            r = d[(d.model == m) & (d.arm == a)].iloc[0]
            hh = h[(h.model == m) & (h.arm == a)].iloc[0].hedge_nosub
            L.append(rf"{m} & {a} & {int(r.labeled):,} & {int(r.fallback_prose):,} & {int(r.fallback_structured):,} & "
                     f"{int(r.multi_verdict)} & {100 * hh:.0f}\\% \\\\")
        if m != "Qwen":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_format", "\n".join(L) + "\n")


def tab_lv_clean():
    d = pd.read_csv(RES / "lv_clean_subset_full.csv")
    full = pd.read_csv(RES / "stats_scores_paired_full.csv")
    cmap = {"B-A": "B_dense_rag - A_zeroshot", "C-B": "C_decompose_filtered_rag - B_dense_rag",
            "C-A": "C_decompose_filtered_rag - A_zeroshot"}
    sm = {"Zephyr": MODELS[0], "Qwen": MODELS[1], "Mistral": MODELS[2]}
    L = [r"\begin{table}[!tbp]", r"\centering", r"\small",
         r"\caption{Paired judge logical-validity contrasts on all claims and on the claims whose outputs are not "
         r"fallback-structured in either pipeline of the comparison (95\% bootstrap CI over claims).}",
         r"\label{tab:lv_clean}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Model & Contrast & $\Delta$LV, all claims & $n$ clean & $\Delta$LV, clean claims [95\% CI] \\", r"\midrule"]
    for m in ["Zephyr", "Qwen", "Mistral"]:
        for c in ["B-A", "C-B", "C-A"]:
            r = d[(d.model == m) & (d.contrast == c)].iloc[0]
            f = full[(full.model == sm[m]) & (full.contrast == cmap[c]) & (full.metric == "logical_validity")].iloc[0]
            L.append(rf"{m} & {c.replace('-', '$-$')} & {f.mean_diff:+.2f} & {int(r.n_clean_pairs):,} & "
                     rf"{r.delta_lv:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] \\")
        if m != "Mistral":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_lv_clean", "\n".join(L) + "\n")

def tab_kb():
    d = pd.read_json(RES / "kb_matched_2x2.json")
    L = [r"\begin{table}[!tbp]", r"\centering", r"\small",
         r"\caption{Matched knowledge-base comparison: top-3 retrieval for all 1{,}535 claims, every retrieved chunk judged by the same Llama-3.1-8B context-precision prompt. "
         r"CP = rank-weighted context precision; Any = share of claims with at least one relevant chunk in the top 3; Rel.\ = share of retrieved chunks judged relevant. Brackets: 95\% bootstrap CI over claims.}",
         r"\label{tab:kb}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"KB (chunks) & Retrieval & CP & Any & Rel. \\", r"\midrule"]
    for kb in ("summary", "full"):
        for mode in ("dense", "hybrid"):
            r = d[(d.kb == kb) & (d.retrieval == mode)].iloc[0]
            L.append(rf"{kb.capitalize()} ({int(r.n_chunks_in_kb):,}) & {'Dense' if mode == 'dense' else 'Hybrid'} & "
                     rf"{ci(r.context_precision, r.cp_lo, r.cp_hi)} & {ci(r.any_relevant, r.hit_lo, r.hit_hi)} & {r.frac_relevant:.3f} \\")
        if kb == "summary":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_kb", "\n".join(L) + "\n")


if __name__ == "__main__":
    for f in (tab_class, tab_paired, tab_gen, tab_scores, tab_cards, tab_verdicts):
        f()
    tab_format()
    tab_lv_clean()
    if (RES / "kb_matched_2x2.json").exists():
        tab_kb()
    if (RES / "ablation_ci.csv").exists():
        tab_ablation()
        tab_ablation_paired()
    print("wrote tables to", OUT)
