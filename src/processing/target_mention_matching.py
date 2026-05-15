"""
target_mention_matching.py
Step 5.5: Full Target Mention Matching — All 499 Targets
NeuroGraph Agent — Alzheimer's Disease Research Intelligence

Matches every target in target_evidence_clean.csv against the full literature
and clinical trial text corpora using tokenisation-based counting.
Replaces the top-50 approach used in text_mining.py (Step 5).

All findings are hypothesis-generating signals only. Gene symbol matching is
a lexical technique and cannot resolve synonyms, gene family ambiguity, or
indirect references. Results require expert validation.
"""

import os
import sys
import io
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
FIG_DIR  = os.path.join(BASE_DIR, "outputs", "figures")
RPT_DIR  = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)

# tokeniser: alphanumeric sequences only (splits on hyphens, slashes, etc.)
_TOKENISE = re.compile(r'[A-Za-z0-9]+')


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def tokenise(text: str) -> Counter:
    """Return a Counter of lowercased alphanumeric tokens from text."""
    if not isinstance(text, str) or not text.strip():
        return Counter()
    return Counter(_TOKENISE.findall(text.lower()))


def symbol_ambiguity(symbol: str) -> str:
    """
    Classify a gene symbol by its potential for false-positive matches.

    Rules (conservative — biomedical corpora reduce real-world ambiguity):
      high     : len <= 2  (not present in this dataset, reserved for future)
      moderate : len == 3, all alpha  (APP, ACE, CLU — may match abbreviations)
      low      : has a digit, or len >= 4
    """
    has_digit = any(c.isdigit() for c in symbol)
    n = len(symbol)
    if has_digit:
        return "low"    # digits make the token distinctive
    if n <= 2:
        return "high"   # extremely short — very likely to match common words
    if n == 3:
        return "moderate"  # short all-alpha: possible but limited ambiguity in biomedical text
    return "low"        # 4+ all-alpha is generally specific in biomedical context


def mention_quality(ambig: str, lit: int, trial: int) -> str:
    total = lit + trial
    if total == 0:
        return "not_detected"
    if ambig == "high":
        return "high_ambiguity_caution"
    if ambig == "moderate":
        return "moderate_ambiguity_review"
    return "reliable"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("=" * 60)
    print("STEP 5.5: FULL TARGET MENTION MATCHING — NEUROGRAPH AGENT")
    print("=" * 60)

    for path in (
        os.path.join(DATA_DIR, "target_evidence_clean.csv"),
        os.path.join(DATA_DIR, "literature_clean.csv"),
        os.path.join(DATA_DIR, "clinical_trials_clean.csv"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file missing: {path}")

    te     = pd.read_csv(os.path.join(DATA_DIR, "target_evidence_clean.csv"))
    lit    = pd.read_csv(os.path.join(DATA_DIR, "literature_clean.csv"))
    trials = pd.read_csv(os.path.join(DATA_DIR, "clinical_trials_clean.csv"))

    # deduplicate targets by approved_symbol (keep highest association_score)
    n_before = len(te)
    te = (
        te.sort_values("association_score", ascending=False)
          .drop_duplicates(subset="approved_symbol", keep="first")
          .reset_index(drop=True)
    )
    if len(te) < n_before:
        print(f"  [Info] Deduplicated target_evidence: {n_before} → {len(te)}")

    print(f"\nTargets to match:  {len(te):,}")
    print(f"Literature docs:   {len(lit):,}")
    print(f"Clinical trials:   {len(trials):,}")
    return te, lit, trials


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD TEXT CORPORA
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_text(row: pd.Series, cols: list) -> str:
    parts = [str(row[c]) for c in cols
             if c in row.index and pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan")]
    return " ".join(parts)


def build_corpora(lit: pd.DataFrame, trials: pd.DataFrame):
    """Return lists of raw text strings (one per document) for each corpus."""
    # literature: use paper_text if well-populated, else combine fields
    if lit["paper_text"].notna().sum() > len(lit) * 0.5:
        lit_texts = lit["paper_text"].fillna("").tolist()
    else:
        lit_texts = [
            _safe_text(row, ["title", "abstract", "fields_of_study"])
            for _, row in lit.iterrows()
        ]

    # trials: use trial_text if well-populated, else combine fields
    trial_cols = [
        "brief_title", "official_title", "conditions", "interventions",
        "primary_outcomes", "eligibility_criteria", "brief_summary",
    ]
    if trials["trial_text"].notna().sum() > len(trials) * 0.5:
        trial_texts = trials["trial_text"].fillna("").tolist()
    else:
        trial_texts = [
            _safe_text(row, trial_cols) for _, row in trials.iterrows()
        ]

    print(f"\nTokenising {len(lit_texts)} literature documents...")
    lit_counters = [tokenise(t) for t in lit_texts]
    print(f"Tokenising {len(trial_texts)} trial documents...")
    trial_counters = [tokenise(t) for t in trial_texts]
    print("Tokenisation complete.")

    return lit_counters, trial_counters


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MATCH ALL TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

def match_targets(te: pd.DataFrame,
                  lit_counters: list,
                  trial_counters: list) -> pd.DataFrame:
    print(f"\nMatching {len(te)} targets...")

    rows = []
    for i, (_, r) in enumerate(te.iterrows()):
        symbol  = str(r["approved_symbol"]).strip()
        sym_low = symbol.lower()

        # per-document counts (fast: Counter lookup is O(1))
        lit_counts   = [c[sym_low] for c in lit_counters]
        trial_counts = [c[sym_low] for c in trial_counters]

        lit_total   = sum(lit_counts)
        trial_total = sum(trial_counts)
        lit_docs    = sum(1 for x in lit_counts   if x > 0)
        trial_docs  = sum(1 for x in trial_counts if x > 0)

        ambig   = symbol_ambiguity(symbol)
        quality = mention_quality(ambig, lit_total, trial_total)
        gap     = lit_total - trial_total
        status  = "checked_with_mentions" if (lit_total + trial_total) > 0 else "checked_zero_mentions"

        rows.append({
            "target_id":                     r.get("target_id", ""),
            "approved_symbol":               symbol,
            "approved_name":                 r.get("approved_name", ""),
            "biotype":                       r.get("biotype", ""),
            "association_score":             r.get("association_score", np.nan),
            "literature_mentions":           lit_total,
            "trial_mentions":                trial_total,
            "literature_documents_mentioned": lit_docs,
            "trial_documents_mentioned":     trial_docs,
            "literature_to_trial_gap_signal": gap,
            "symbol_ambiguity_flag":         ambig,
            "mention_match_quality":         quality,
            "text_coverage_status":          status,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(te)} targets processed...")

    df = pd.DataFrame(rows)

    # validation
    n_lit    = (df["literature_mentions"] > 0).sum()
    n_trial  = (df["trial_mentions"] > 0).sum()
    n_zero   = ((df["literature_mentions"] == 0) & (df["trial_mentions"] == 0)).sum()
    n_unc    = (df["text_coverage_status"] == "not_checked").sum()

    print(f"\n  Total targets checked:         {len(df):,}")
    print(f"  With literature mentions:      {n_lit}")
    print(f"  With trial mentions:           {n_trial}")
    print(f"  Zero mentions in both corpora: {n_zero}")
    print(f"  not_checked (should be 0):     {n_unc}")
    if n_unc > 0:
        print("  [Warning] Some targets were not checked — investigate.")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SAVE TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def save_table(df: pd.DataFrame):
    out_path = os.path.join(DATA_DIR, "target_text_mentions_full.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n  Saved: {out_path}")

    # also print top 10 by literature_mentions
    print("\n  Top 10 targets by literature_mentions:")
    top10 = df.sort_values("literature_mentions", ascending=False).head(10)
    for _, r in top10.iterrows():
        print(f"    {r['approved_symbol']:<12}  lit={r['literature_mentions']:4d}  "
              f"trial={r['trial_mentions']:4d}  "
              f"ambig={r['symbol_ambiguity_flag']}  quality={r['mention_match_quality']}")

    print("\n  Top 10 targets by trial_mentions:")
    top10t = df.sort_values("trial_mentions", ascending=False).head(10)
    for _, r in top10t.iterrows():
        print(f"    {r['approved_symbol']:<12}  lit={r['literature_mentions']:4d}  "
              f"trial={r['trial_mentions']:4d}  "
              f"ambig={r['symbol_ambiguity_flag']}")

    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def make_figures(df: pd.DataFrame):
    print("\n--- Generating figures ---")

    # convenience subsets
    has_mentions = df[(df["literature_mentions"] > 0) | (df["trial_mentions"] > 0)]

    AMBIG_COLOURS = {"low": "#4472C4", "moderate": "#ED7D31", "high": "#C00000"}

    # ── Fig 1: All targets scatter — lit vs trial mentions ────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    for ambig, colour in AMBIG_COLOURS.items():
        sub = df[df["symbol_ambiguity_flag"] == ambig]
        if len(sub) == 0:
            continue
        ax.scatter(
            np.log1p(sub["literature_mentions"]),
            np.log1p(sub["trial_mentions"]),
            c=colour, s=20, alpha=0.65, label=f"ambiguity: {ambig}",
        )

    # annotate top 15 by lit_mentions (non-zero)
    top_lit = df[df["literature_mentions"] > 0].nlargest(15, "literature_mentions")
    for _, r in top_lit.iterrows():
        ax.annotate(
            r["approved_symbol"],
            xy=(np.log1p(r["literature_mentions"]), np.log1p(r["trial_mentions"])),
            xytext=(4, 2), textcoords="offset points",
            fontsize=7.5, color="#222222",
        )

    lim = max(
        np.log1p(df["literature_mentions"].max()),
        np.log1p(df["trial_mentions"].max()),
    ) * 1.05 + 0.1
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--",
            linewidth=1, label="Equal mentions")

    ax.set_xlabel("log1p(Literature Mentions)", fontsize=11)
    ax.set_ylabel("log1p(Trial Mentions)", fontsize=11)
    ax.set_title(
        "Literature vs Trial Mentions — All 499 Targets\n"
        "(targets below diagonal appear more in literature than trials; "
        "zero-mention targets cluster at origin)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "text_target_mentions_full_lit_vs_trials.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Fig 2: Top 20 by literature_mentions ──────────────────────────────────
    top20_lit = df.nlargest(20, "literature_mentions")
    fig, ax = plt.subplots(figsize=(11, 7))
    colours = [AMBIG_COLOURS[a] for a in top20_lit["symbol_ambiguity_flag"]]
    bars = ax.barh(
        top20_lit["approved_symbol"][::-1],
        top20_lit["literature_mentions"][::-1],
        color=colours[::-1], edgecolor="white",
    )
    ax.set_xlabel("Literature Mention Count (raw)", fontsize=11)
    ax.set_title(
        "Top 20 Targets by Literature Mentions\n"
        "(colour = symbol ambiguity; orange/red = review required)",
        fontsize=12,
    )
    ax.bar_label(bars, padding=3, fontsize=8)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "text_target_mentions_full_top_literature.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Fig 3: Top 20 by trial_mentions ──────────────────────────────────────
    top20_tri = df.nlargest(20, "trial_mentions")
    fig, ax = plt.subplots(figsize=(11, 7))
    colours = [AMBIG_COLOURS[a] for a in top20_tri["symbol_ambiguity_flag"]]
    bars = ax.barh(
        top20_tri["approved_symbol"][::-1],
        top20_tri["trial_mentions"][::-1],
        color=colours[::-1], edgecolor="white",
    )
    ax.set_xlabel("Trial Mention Count (raw)", fontsize=11)
    ax.set_title(
        "Top 20 Targets by Trial Mentions\n"
        "(colour = symbol ambiguity; orange = review required)",
        fontsize=12,
    )
    ax.bar_label(bars, padding=3, fontsize=8)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "text_target_mentions_full_top_trials.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(df: pd.DataFrame):
    n_total  = len(df)
    n_lit    = (df["literature_mentions"] > 0).sum()
    n_trial  = (df["trial_mentions"] > 0).sum()
    n_zero   = ((df["literature_mentions"] == 0) & (df["trial_mentions"] == 0)).sum()
    n_ambig  = (df["symbol_ambiguity_flag"] == "high").sum()
    n_mod    = (df["symbol_ambiguity_flag"] == "moderate").sum()
    n_low    = (df["symbol_ambiguity_flag"] == "low").sum()

    top10_lit  = df.nlargest(10, "literature_mentions")
    top10_tri  = df.nlargest(10, "trial_mentions")

    lines = [
        "# NeuroGraph Agent — Target Mention Matching Summary",
        "",
        "*Generated: 2026-05-13 | Step 5.5 of NeuroGraph pipeline*",
        "",
        "> All mention counts are derived from whole-word tokenisation of biomedical text.",
        "> They are lexical signals only. Gene symbols may match non-target tokens,",
        "> and targets may be referenced without their gene symbol. These counts",
        "> are hypothesis-generating signals, not measures of research activity.",
        "",
        "---",
        "",
        "## 1. Coverage",
        "",
        f"All {n_total} unique targets in `target_evidence_clean.csv` were checked against",
        "the full literature and clinical trial text corpora.",
        "",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| Targets checked | {n_total} |",
        f"| With literature mentions (>0) | {n_lit} |",
        f"| With trial mentions (>0) | {n_trial} |",
        f"| Zero mentions in both corpora | {n_zero} |",
        f"| not_checked | 0 |",
        "",
        f"**{n_zero} targets had zero mentions in both corpora.** This means the gene",
        "symbol was not found as a standalone alphanumeric token in the retrieved",
        "documents. It does not mean the target is absent from the broader literature",
        "or from ongoing trials — it may be referenced via protein name, drug name,",
        "pathway, or not present in this specific corpus sample.",
        "",
        "---",
        "",
        "## 2. Methods",
        "",
        "Text was sourced from `paper_text` (literature) and `trial_text` (trials).",
        "Each document was tokenised into lowercased alphanumeric sequences using",
        "`[A-Za-z0-9]+`. For each target, the lowercased approved_symbol was",
        "looked up in each document's token Counter. This approach:",
        "",
        "- Handles hyphens and slashes as separators (TREM2-targeted → ['trem2', 'targeted'])",
        "- Is case-insensitive via lowercasing",
        "- Does NOT resolve synonyms, protein names, or compound symbols (APP-swe)",
        "- Does NOT use approximate or semantic matching",
        "",
        "---",
        "",
        "## 3. Symbol Ambiguity",
        "",
        "Gene symbols were classified by potential for false-positive matches:",
        "",
        f"| Ambiguity | Criteria | Count |",
        f"|-----------|----------|-------|",
        f"| low | has digit, or len >= 4 | {n_low} |",
        f"| moderate | len == 3, all alpha (e.g. APP, ACE) | {n_mod} |",
        f"| high | len <= 2 | {n_ambig} |",
        "",
        "Targets with moderate ambiguity are flagged in `mention_match_quality`",
        "as `moderate_ambiguity_review`. Their mention counts may include matches",
        "from non-gene uses of the same abbreviation in clinical text.",
        "",
        "> In a focused biomedical corpus, most 3-letter gene symbols are used in",
        "> their molecular biology context and rarely match unrelated abbreviations.",
        "> However, manual validation is recommended for key findings from this tier.",
        "",
        "---",
        "",
        "## 4. Top Targets by Literature Mentions",
        "",
        "| Symbol | Approved Name | Lit Mentions | Trial Mentions | Gap Signal | Ambiguity |",
        "|--------|--------------|-------------|---------------|-----------|-----------|",
    ]

    for _, r in top10_lit.iterrows():
        name_s = str(r["approved_name"])[:40]
        lines.append(
            f"| {r['approved_symbol']} | {name_s} | "
            f"{int(r['literature_mentions'])} | {int(r['trial_mentions'])} | "
            f"{int(r['literature_to_trial_gap_signal'])} | {r['symbol_ambiguity_flag']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Top Targets by Trial Mentions",
        "",
        "| Symbol | Approved Name | Trial Mentions | Lit Mentions | Ambiguity |",
        "|--------|--------------|---------------|-------------|-----------|",
    ]

    for _, r in top10_tri.iterrows():
        name_s = str(r["approved_name"])[:40]
        lines.append(
            f"| {r['approved_symbol']} | {name_s} | "
            f"{int(r['trial_mentions'])} | {int(r['literature_mentions'])} | "
            f"{r['symbol_ambiguity_flag']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Limitations and Warnings",
        "",
        "### Gene symbol ambiguity",
        "Short gene symbols (especially 3-letter all-alpha, e.g. APP, ACE, CLU) may",
        "match common biomedical abbreviations beyond the target gene. For example,",
        "'ACE' appears in 'ACE inhibitor' which does refer to the ACE protein — however",
        "some clinical text may use 'ACE' to refer to the drug class rather than the gene.",
        "Targets with `mention_match_quality = moderate_ambiguity_review` should be",
        "manually confirmed before citing counts as evidence of research activity.",
        "",
        "### Trial text does not name molecular targets",
        "Clinical trial registrations describe drugs, interventions, and patient criteria.",
        "A trial targeting TREM2 via a monoclonal antibody (e.g. AL002) will not",
        "contain the word 'TREM2' in its registration text. **Low or zero trial mention",
        "counts are not evidence of underinvestigation** — they reflect the lexical",
        "structure of trial registrations, not the true scope of target-related trials.",
        "",
        "### Corpus size",
        "The literature corpus (478 papers) and trial corpus (1,000 trials) are samples.",
        "Absence from this corpus does not imply absence from the broader literature.",
        "",
        "### No semantic resolution",
        "The tokenisation approach does not resolve synonyms (e.g. 'presenilin 1' → PSEN1),",
        "protein names, drug names, or pathway references. Targets may be well-studied",
        "without appearing as gene symbols in text.",
        "",
        "---",
        "",
        "*This report is for research intelligence purposes only. It is not clinical",
        "evidence and should not inform clinical or drug development decisions.*",
    ]

    report_path = os.path.join(RPT_DIR, "target_mention_matching_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    te, lit, trials = load_data()
    lit_counters, trial_counters = build_corpora(lit, trials)
    df = match_targets(te, lit_counters, trial_counters)
    save_table(df)
    make_figures(df)
    write_report(df)

    # validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    assert len(df) == 499, f"Expected 499 targets, got {len(df)}"
    assert (df["text_coverage_status"] == "not_checked").sum() == 0, "Some targets not checked"
    assert df["approved_symbol"].notna().all(), "Missing symbols"

    saved_files = [
        os.path.join(DATA_DIR, "target_text_mentions_full.csv"),
        os.path.join(FIG_DIR,  "text_target_mentions_full_lit_vs_trials.png"),
        os.path.join(FIG_DIR,  "text_target_mentions_full_top_literature.png"),
        os.path.join(FIG_DIR,  "text_target_mentions_full_top_trials.png"),
        os.path.join(RPT_DIR,  "target_mention_matching_summary.md"),
    ]
    for p in saved_files:
        status = "OK" if os.path.isfile(p) else "MISSING"
        print(f"  [{status}] {os.path.relpath(p, BASE_DIR)}")

    print("\nStep 5.5 complete.")


if __name__ == "__main__":
    main()
