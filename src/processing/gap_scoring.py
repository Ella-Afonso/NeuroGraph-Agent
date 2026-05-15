"""
gap_scoring.py
Step 6: Research Gap Scoring (revised using full 499-target mention matching)
NeuroGraph Agent — Alzheimer's Disease Research Intelligence

Combines Open Targets association scores, literature mention counts, and
clinical trial mention counts — all 499 targets checked — to produce a
ranked research gap score. Uses a literature-aware gap component so that
targets with zero literature signal do not receive inflated underexploration
scores. All scores are hypothesis-generating signals only.
"""

import os
import sys
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
FIG_DIR  = os.path.join(BASE_DIR, "outputs", "figures")
RPT_DIR  = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)

# ── scoring weights ────────────────────────────────────────────────────────────
W_EVIDENCE     = 0.40
W_LITERATURE   = 0.30
W_LIT_GAP      = 0.30   # literature_trial_gap_norm component

W_CONS_EVIDENCE  = 0.50
W_CONS_LIT       = 0.20
W_CONS_LIT_GAP   = 0.30

# ── category thresholds (percentile-based) ────────────────────────────────────
PCT_HIGH     = 90   # top 10%
PCT_MODERATE = 70   # next 20%


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("=" * 60)
    print("STEP 6: RESEARCH GAP SCORING (FULL MENTION MATCHING)")
    print("=" * 60)

    full_path = os.path.join(DATA_DIR, "target_text_mentions_full.csv")
    if not os.path.isfile(full_path):
        raise FileNotFoundError(
            f"Required file missing: {full_path}\n"
            "Run src/processing/target_mention_matching.py first."
        )

    df = pd.read_csv(full_path)
    print(f"\nLoaded target_text_mentions_full:  {len(df):,} targets")
    print(f"  With literature mentions:  {(df['literature_mentions'] > 0).sum()}")
    print(f"  With trial mentions:       {(df['trial_mentions'] > 0).sum()}")
    print(f"  Zero in both corpora:      {((df['literature_mentions'] == 0) & (df['trial_mentions'] == 0)).sum()}")

    # optional topic files (for report context only)
    topic_data = {}
    for key, fname in [
        ("lit_topics",   "literature_topics.csv"),
        ("trial_topics", "clinical_trial_topics.csv"),
    ]:
        p = os.path.join(DATA_DIR, fname)
        topic_data[key] = pd.read_csv(p) if os.path.isfile(p) else None

    return df, topic_data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE SCORES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Computing gap scores ---")
    scaler = MinMaxScaler()

    # evidence_strength_norm: rescale association_score to [0,1]
    df["evidence_strength_norm"] = scaler.fit_transform(
        df[["association_score"]]
    ).flatten()

    # literature_signal_norm: log1p → scale
    df["_lit_log"] = np.log1p(df["literature_mentions"])
    df["literature_signal_norm"] = scaler.fit_transform(
        df[["_lit_log"]]
    ).flatten()

    # trial_signal_norm: log1p → scale
    df["_trial_log"] = np.log1p(df["trial_mentions"])
    df["trial_signal_norm"] = scaler.fit_transform(
        df[["_trial_log"]]
    ).flatten()

    # underexploration_norm (kept for reference / backward compat)
    df["underexploration_norm"] = 1.0 - df["trial_signal_norm"]

    # literature_trial_gap_norm: high only if BOTH lit signal exists AND trial signal is low
    # = 0 when lit_signal = 0 (no lit → no gap claim)
    # = 0 when trial_signal = 1 (fully covered in trials → no gap claim)
    # = max when lit_signal = 1 AND trial_signal = 0
    df["literature_trial_gap_norm"] = (
        df["literature_signal_norm"] * (1.0 - df["trial_signal_norm"])
    )

    # primary gap_score
    df["gap_score"] = (
        W_EVIDENCE   * df["evidence_strength_norm"]
        + W_LITERATURE * df["literature_signal_norm"]
        + W_LIT_GAP    * df["literature_trial_gap_norm"]
    ).round(6)

    # conservative gap_score
    df["conservative_gap_score"] = (
        W_CONS_EVIDENCE * df["evidence_strength_norm"]
        + W_CONS_LIT    * df["literature_signal_norm"]
        + W_CONS_LIT_GAP * df["literature_trial_gap_norm"]
    ).round(6)

    # validate range
    for col in ["evidence_strength_norm", "literature_signal_norm",
                "trial_signal_norm", "literature_trial_gap_norm",
                "gap_score", "conservative_gap_score"]:
        lo, hi = df[col].min(), df[col].max()
        if lo < -1e-6 or hi > 1 + 1e-6:
            print(f"  [Warning] {col} out of [0,1]: {lo:.4f}–{hi:.4f}")

    print(f"  gap_score range:              {df['gap_score'].min():.4f} – {df['gap_score'].max():.4f}")
    print(f"  conservative_gap_score range: {df['conservative_gap_score'].min():.4f} – {df['conservative_gap_score'].max():.4f}")
    print(f"  Targets with lit_trial_gap > 0: {(df['literature_trial_gap_norm'] > 0).sum()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GAP CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

def assign_categories(df: pd.DataFrame):
    high_thresh     = np.percentile(df["gap_score"], PCT_HIGH)
    moderate_thresh = np.percentile(df["gap_score"], PCT_MODERATE)

    def categorise(score):
        if score >= high_thresh:
            return "High-priority gap signal"
        if score >= moderate_thresh:
            return "Moderate gap signal"
        return "Low gap signal"

    df["gap_category"] = df["gap_score"].apply(categorise)
    method_note = (
        f"Percentile-based: top {100-PCT_HIGH}% = High-priority "
        f"(>= {high_thresh:.4f}), "
        f"next {PCT_HIGH-PCT_MODERATE}% = Moderate "
        f"(>= {moderate_thresh:.4f})"
    )
    print(f"\n  Category method: {method_note}")
    for cat, n in df["gap_category"].value_counts().items():
        print(f"    {cat}: {n}")

    return df, method_note, high_thresh, moderate_thresh


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INTERPRETATION TIERS
# ═══════════════════════════════════════════════════════════════════════════════

def assign_tiers(df: pd.DataFrame, high_thresh: float):
    # threshold for "trial-covered": 90th pct of trial_mentions, min 1
    trial_90th = max(float(df["trial_mentions"].quantile(0.90)), 1.0)
    print(f"\n  Trial mentions 90th pct (all targets): {trial_90th:.1f}")

    def tier(row):
        lit    = int(row["literature_mentions"])
        trial  = int(row["trial_mentions"])
        score  = row["gap_score"]
        ambig  = row["symbol_ambiguity_flag"]

        # 1. Established / trial-covered (regardless of ambiguity)
        if trial > trial_90th:
            return "Established / trial-covered target"

        # 2. High-ambiguity symbol
        if ambig == "high":
            return "Ambiguous symbol; manual validation required"

        # 3. Strong gap: lit signal present, low trials, high score
        if lit > 0 and trial <= trial_90th and score >= high_thresh:
            return "Strong text-supported gap signal"

        # 4. Detected in neither corpus
        if lit == 0 and trial == 0:
            return "Checked-zero text signal"

        # 5. Everything else
        return "Low-priority / weak gap signal"

    df["gap_interpretation_tier"] = df.apply(tier, axis=1)

    tier_counts = df["gap_interpretation_tier"].value_counts()
    print("  Interpretation tier counts:")
    for t, n in tier_counts.items():
        print(f"    {t}: {n}")

    return df, trial_90th


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTERPRETATION NOTES
# ═══════════════════════════════════════════════════════════════════════════════

def make_note(row) -> str:
    lit_abs   = int(row["literature_mentions"])
    trial_abs = int(row["trial_mentions"])
    ev        = row["evidence_strength_norm"]
    tier      = row["gap_interpretation_tier"]
    ambig     = row["symbol_ambiguity_flag"]

    if tier == "Strong text-supported gap signal":
        return (
            f"Literature mentions detected ({lit_abs}), with low or zero clinical trial "
            f"text presence ({trial_abs} mentions). Combined with Open Targets evidence "
            f"(norm={ev:.2f}), this may suggest a research-to-clinic gap for this target. "
            f"Requires expert validation; trial text may not name molecular targets directly."
        )
    if tier == "Established / trial-covered target":
        if ambig == "moderate":
            return (
                f"High trial mention count ({trial_abs}), but symbol ambiguity is 'moderate'. "
                f"Some trial mentions may reflect non-gene uses of '{row['approved_symbol']}'. "
                f"Manual validation recommended. Not classified as a gap candidate."
            )
        return (
            f"Substantial clinical trial text presence ({trial_abs} trial mentions). "
            f"This target appears investigated in clinical settings within this corpus. "
            f"Not classified as a gap candidate based on this signal."
        )
    if tier == "Checked-zero text signal":
        return (
            f"Not detected in literature or trial text in this corpus. "
            f"This is a lexical absence — the gene symbol was not found as a standalone "
            f"token. The target may be referenced via protein name, drug name, or pathway, "
            f"or may simply not appear in this sample of 478 papers and 1,000 trials."
        )
    if tier == "Ambiguous symbol; manual validation required":
        return (
            f"Gene symbol is very short (ambiguity=high). Mention counts may include "
            f"false positives from non-gene uses. Manual literature search recommended."
        )
    if lit_abs > 0 and trial_abs > 0:
        return (
            f"Detected in both literature ({lit_abs}) and trial text ({trial_abs}). "
            f"Does not meet the threshold for strong gap signal in this scoring framework."
        )
    return (
        "Low text signal; gap score primarily reflects Open Targets evidence strength. "
        "Does not represent a confirmed gap candidate in this framework."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RANK AND FINALISE
# ═══════════════════════════════════════════════════════════════════════════════

def rank_and_finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("gap_score", ascending=False).reset_index(drop=True)
    df["gap_rank"] = df.index + 1
    df["interpretation_note"] = df.apply(make_note, axis=1)

    assert df["gap_rank"].nunique() == len(df), "gap_rank not unique"
    assert df["approved_symbol"].notna().all(), "Missing approved_symbol"

    output_cols = [
        "gap_rank", "approved_symbol", "approved_name", "target_id",
        "biotype", "association_score",
        "literature_mentions", "trial_mentions",
        "literature_documents_mentioned", "trial_documents_mentioned",
        "literature_to_trial_gap_signal",
        "evidence_strength_norm", "literature_signal_norm",
        "trial_signal_norm", "underexploration_norm",
        "literature_trial_gap_norm",
        "gap_score", "conservative_gap_score",
        "gap_category", "gap_interpretation_tier",
        "text_coverage_status", "symbol_ambiguity_flag",
        "mention_match_quality", "interpretation_note",
    ]
    # only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    return df[output_cols].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def save_tables(df_out: pd.DataFrame):
    # full ranked table
    main_path = os.path.join(DATA_DIR, "research_gap_scores.csv")
    df_out.to_csv(main_path, index=False, encoding="utf-8")
    print(f"\n  Saved: {main_path}")

    # top 25 overall
    top25_path = os.path.join(DATA_DIR, "top_research_gap_candidates.csv")
    df_out.head(25).to_csv(top25_path, index=False, encoding="utf-8")
    print(f"  Saved: {top25_path}")

    # strong text-supported only
    strong_df = df_out[df_out["gap_interpretation_tier"] == "Strong text-supported gap signal"]
    strong_path = os.path.join(DATA_DIR, "strong_text_supported_gap_candidates.csv")
    strong_df.to_csv(strong_path, index=False, encoding="utf-8")
    print(f"  Saved: {strong_path}  ({len(strong_df)} candidates)")

    # top text-supported (strong + low-priority-with-lit)
    ts_df = df_out[df_out["literature_mentions"] > 0].head(25)
    ts_path = os.path.join(DATA_DIR, "top_text_supported_gap_candidates.csv")
    ts_df.to_csv(ts_path, index=False, encoding="utf-8")
    print(f"  Saved: {ts_path}")

    # score components summary
    summary_cols = [
        "approved_symbol", "gap_rank",
        "evidence_strength_norm", "literature_signal_norm",
        "trial_signal_norm", "literature_trial_gap_norm",
        "gap_score", "conservative_gap_score",
        "gap_category", "gap_interpretation_tier",
        "text_coverage_status", "symbol_ambiguity_flag",
    ]
    summary_cols = [c for c in summary_cols if c in df_out.columns]
    comp_path = os.path.join(DATA_DIR, "gap_score_components_summary.csv")
    df_out[summary_cols].to_csv(comp_path, index=False, encoding="utf-8")
    print(f"  Saved: {comp_path}")

    return main_path, top25_path, strong_path, ts_path, comp_path


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def make_figures(df_out: pd.DataFrame, high_thresh: float, moderate_thresh: float,
                 trial_90th: float):
    print("\n--- Generating figures ---")

    top20     = df_out.head(20)
    has_lit   = df_out[df_out["literature_mentions"] > 0]
    zero_both = df_out[(df_out["literature_mentions"] == 0) & (df_out["trial_mentions"] == 0)]

    # ── Fig 1: Top 20 gap candidates ──────────────────────────────────────────
    TIER_COLOURS = {
        "Strong text-supported gap signal":          "#C00000",
        "Established / trial-covered target":        "#ED7D31",
        "Low-priority / weak gap signal":            "#4472C4",
        "Checked-zero text signal":                  "#BFBFBF",
        "Ambiguous symbol; manual validation required": "#7030A0",
    }
    colours = [TIER_COLOURS.get(t, "#4472C4") for t in top20["gap_interpretation_tier"]]
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top20["approved_symbol"][::-1], top20["gap_score"][::-1],
                   color=colours[::-1], edgecolor="white", linewidth=0.5)
    ax.axvline(high_thresh,     color="#C00000", linestyle="--", lw=1,
               label=f"High threshold ({high_thresh:.4f})")
    ax.axvline(moderate_thresh, color="#ED7D31", linestyle=":",  lw=1,
               label=f"Moderate threshold ({moderate_thresh:.4f})")
    ax.set_xlabel("Gap Score", fontsize=11)
    ax.set_title(
        "Top 20 Gap Candidates — Gap Score (full 499-target matching)\n"
        "(computationally derived; not clinical evidence)",
        fontsize=11,
    )
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "gap_01_top_gap_candidates.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Fig 2: Score distribution ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, colour, label in [
        (axes[0], "gap_score",              "#4472C4", "Gap Score"),
        (axes[1], "conservative_gap_score", "#ED7D31", "Conservative Gap Score"),
    ]:
        ax.hist(df_out[col], bins=30, color=colour, edgecolor="white", linewidth=0.4)
        ax.axvline(df_out[col].median(), color="black",   linestyle="--", lw=1.2,
                   label=f"Median {df_out[col].median():.4f}")
        ax.axvline(df_out[col].mean(),   color="#C00000", linestyle=":",  lw=1.2,
                   label=f"Mean {df_out[col].mean():.4f}")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Number of Targets", fontsize=11)
        ax.set_title(f"Distribution: {label}", fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Gap Score Distributions (all 499 targets, full mention matching)\n"
        "Bimodal shape expected: zero-lit targets cluster at left; lit-signal targets at right",
        fontsize=10,
    )
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "gap_02_score_distribution.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Fig 3: Evidence score vs trial mentions scatter ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        has_lit["association_score"],
        has_lit["_trial_log"] if "_trial_log" in has_lit.columns else np.log1p(has_lit["trial_mentions"]),
        c=has_lit["gap_score"], cmap="RdYlGn_r",
        s=60, alpha=0.85, zorder=3, label="Has lit mentions",
    )
    ax.scatter(
        zero_both["association_score"],
        np.zeros(len(zero_both)),
        c="lightgrey", s=10, alpha=0.3, zorder=2, label="Zero mentions in both",
    )
    plt.colorbar(sc, ax=ax, label="Gap Score")
    for _, row in df_out.head(12).iterrows():
        y = (row["_trial_log"] if "_trial_log" in row.index
             else np.log1p(row["trial_mentions"]))
        ax.annotate(row["approved_symbol"], xy=(row["association_score"], y),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Open Targets Association Score", fontsize=11)
    ax.set_ylabel("log1p(Trial Mentions)", fontsize=11)
    ax.set_title(
        "Evidence Score vs Trial Text Mentions\n"
        "(annotated = top 12 by gap score; grey = zero-mention targets at y=0)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "gap_03_evidence_vs_trial_mentions.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Fig 4: Literature vs trial mentions scatter ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    sc2 = ax.scatter(
        np.log1p(has_lit["literature_mentions"]),
        np.log1p(has_lit["trial_mentions"]),
        c=has_lit["gap_score"], cmap="RdYlGn_r",
        s=70, alpha=0.85,
    )
    plt.colorbar(sc2, ax=ax, label="Gap Score")
    for _, row in has_lit.head(20).iterrows():
        ax.annotate(row["approved_symbol"],
                    xy=(np.log1p(row["literature_mentions"]), np.log1p(row["trial_mentions"])),
                    xytext=(4, 2), textcoords="offset points", fontsize=8)
    lim = max(np.log1p(has_lit["literature_mentions"].max()),
              np.log1p(has_lit["trial_mentions"].max())) * 1.05 + 0.1
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", lw=1, label="Equal mentions")
    ax.set_xlabel("log1p(Literature Mentions)", fontsize=11)
    ax.set_ylabel("log1p(Trial Mentions)", fontsize=11)
    ax.set_title(
        "Literature vs Trial Text Mentions (targets with lit > 0 only)\n"
        "(below diagonal = more literature than trial coverage)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "gap_04_literature_vs_trial_mentions.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Fig 5: Score component heatmap (top 20 by gap_score) ──────────────────
    hm_cols = ["evidence_strength_norm", "literature_signal_norm",
               "literature_trial_gap_norm", "gap_score"]
    hm_labels = ["Evidence\nStrength", "Literature\nSignal",
                 "Lit-Trial\nGap Norm", "Gap\nScore"]
    hm_data = top20.set_index("approved_symbol")[hm_cols]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(hm_data.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Normalised Score (0–1)", fraction=0.03, pad=0.04)
    ax.set_xticks(range(len(hm_cols)))
    ax.set_xticklabels(hm_labels, fontsize=10)
    ax.set_yticks(range(len(hm_data)))
    ax.set_yticklabels(hm_data.index, fontsize=9)
    for i in range(len(hm_data)):
        for j in range(len(hm_cols)):
            val = hm_data.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if val > 0.65 else "black")
    ax.set_title("Score Components — Top 20 Gap Candidates\n(provisional)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "gap_05_score_component_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Fig 6: Interpretation tier breakdown ──────────────────────────────────
    tier_order = [
        "Strong text-supported gap signal",
        "Established / trial-covered target",
        "Low-priority / weak gap signal",
        "Checked-zero text signal",
        "Ambiguous symbol; manual validation required",
    ]
    tier_counts = df_out["gap_interpretation_tier"].value_counts()
    counts_ordered = [int(tier_counts.get(t, 0)) for t in tier_order]
    short_labels = [
        "Strong gap\n(text-supported)",
        "Trial-covered",
        "Low-priority /\nweak signal",
        "Checked-zero\n(not detected)",
        "Ambiguous\nsymbol",
    ]
    tier_colours = ["#C00000", "#ED7D31", "#4472C4", "#BFBFBF", "#7030A0"]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(short_labels, counts_ordered, color=tier_colours, edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_ylabel("Number of Targets", fontsize=11)
    ax.set_title(
        "Gap Interpretation Tier Distribution — All 499 Targets\n"
        "(full mention matching; zero-signal targets are 'checked-zero', not 'evidence-only')",
        fontsize=11,
    )
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "gap_06_interpretation_tier_breakdown.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(df_out: pd.DataFrame, method_note: str,
                 high_thresh: float, moderate_thresh: float,
                 trial_90th: float, topic_data: dict):

    n_total   = len(df_out)
    n_lit     = int((df_out["literature_mentions"] > 0).sum())
    n_trial   = int((df_out["trial_mentions"] > 0).sum())
    n_zero    = int(((df_out["literature_mentions"] == 0) &
                     (df_out["trial_mentions"] == 0)).sum())
    n_strong  = int((df_out["gap_interpretation_tier"] == "Strong text-supported gap signal").sum())
    n_covered = int((df_out["gap_interpretation_tier"] == "Established / trial-covered target").sum())
    n_checked_zero = int((df_out["gap_interpretation_tier"] == "Checked-zero text signal").sum())
    high_n    = int((df_out["gap_category"] == "High-priority gap signal").sum())
    mod_n     = int((df_out["gap_category"] == "Moderate gap signal").sum())
    low_n     = int((df_out["gap_category"] == "Low gap signal").sum())

    strong_df = df_out[df_out["gap_interpretation_tier"] == "Strong text-supported gap signal"]
    covered_df = df_out[df_out["gap_interpretation_tier"] == "Established / trial-covered target"]
    top15 = df_out.head(15)

    lines = [
        "# NeuroGraph Agent — Research Gap Scoring Summary",
        "",
        "*Generated: 2026-05-13 | Step 6 (revised) of NeuroGraph pipeline*",
        "",
        "> **Important notice:** This report contains computationally derived research",
        "> gap signals. All findings are hypothesis-generating only. No clinical claims",
        "> are made. All candidates require expert validation.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "This step ranks all 499 Alzheimer's disease targets by a transparent,",
        "multi-component research gap score. Step 5.5 performed full text-mention",
        "matching across all 499 targets (not just the top 50). The scoring now uses",
        "a **literature-aware gap component** so that targets with zero literature",
        "signal cannot receive inflated underexploration scores.",
        "",
        "---",
        "",
        "## 2. Data Inputs and Coverage",
        "",
        f"| Source | Records |",
        f"|--------|---------|",
        f"| target_text_mentions_full.csv | {n_total} targets (all checked in Step 5.5) |",
        f"| Literature corpus | 478 papers |",
        f"| Trial corpus | 1,000 trials |",
        "",
        f"| Mention coverage | Count |",
        f"|-----------------|-------|",
        f"| Targets with literature mentions (> 0) | {n_lit} |",
        f"| Targets with trial mentions (> 0) | {n_trial} |",
        f"| Targets with zero mentions in both | {n_zero} |",
        "",
        "### Important: Zero-mention targets",
        "",
        f"> **{n_zero} targets had zero mentions in both corpora.** This means the",
        "> approved gene symbol was not found as a standalone alphanumeric token in",
        "> the retrieved documents. This is a **lexical absence**, not evidence of",
        "> absence from science or clinical trials. The target may be referenced via",
        "> protein name, drug name, or pathway, or may simply not appear in this",
        "> specific corpus of 478 papers and 1,000 trials.",
        "> These targets receive a `gap_interpretation_tier` of **'Checked-zero text signal'**,",
        "> not 'evidence-only'. Their gap score is driven by Open Targets evidence alone.",
        "",
        "---",
        "",
        "## 3. Scoring Formula",
        "",
        "### Primary Gap Score",
        "",
        "```",
        "gap_score = 0.40 x evidence_strength_norm",
        "          + 0.30 x literature_signal_norm",
        "          + 0.30 x literature_trial_gap_norm",
        "```",
        "",
        "| Component | Weight | Derivation |",
        "|-----------|--------|-----------|",
        "| evidence_strength_norm | 0.40 | MinMaxScaler(association_score) |",
        "| literature_signal_norm | 0.30 | MinMaxScaler(log1p(literature_mentions)) |",
        "| literature_trial_gap_norm | 0.30 | literature_signal_norm x (1 - trial_signal_norm) |",
        "",
        "### Why literature_trial_gap_norm replaces underexploration_norm",
        "",
        "The previous formula used `underexploration_norm = 1 - trial_signal_norm`, which",
        "gave maximum underexploration credit to *any* target with zero trial mentions —",
        "including targets with zero literature signal. This inflated gap scores for targets",
        "that are simply absent from both corpora.",
        "",
        "The new component `literature_trial_gap_norm = literature_signal_norm x (1 - trial_signal_norm)`",
        "is **zero when literature_signal is zero**. A target only receives a positive gap",
        "contribution when it has measurable literature presence *and* relatively low trial",
        "text presence. Targets with zero literature signal get gap_score = 0.40 x evidence_strength_norm only.",
        "",
        "### Conservative Gap Score",
        "",
        "```",
        "conservative_gap_score = 0.50 x evidence_strength_norm",
        "                       + 0.20 x literature_signal_norm",
        "                       + 0.30 x literature_trial_gap_norm",
        "```",
        "",
        f"### Category Assignment",
        f"Categories use percentile-based thresholds ({method_note}).",
        "",
        f"| Category | Count | Threshold |",
        f"|----------|-------|-----------|",
        f"| High-priority gap signal | {high_n} | gap_score >= {high_thresh:.4f} |",
        f"| Moderate gap signal | {mod_n} | gap_score >= {moderate_thresh:.4f} |",
        f"| Low gap signal | {low_n} | gap_score < {moderate_thresh:.4f} |",
        "",
        "### Interpretation Tiers",
        "",
        "| Tier | Criteria | Count |",
        "|------|----------|-------|",
        f"| Strong text-supported gap signal | lit > 0, trial <= {trial_90th:.0f} (90th pct), high gap_score, low/mod ambiguity | {n_strong} |",
        f"| Established / trial-covered target | trial > {trial_90th:.0f} (90th pct) | {n_covered} |",
        f"| Checked-zero text signal | lit = 0 AND trial = 0 | {n_checked_zero} |",
        "| Ambiguous symbol; manual validation required | symbol_ambiguity_flag = high | 0 |",
        "| Low-priority / weak gap signal | all other | — |",
        "",
        "---",
        "",
        "## 4. Top Gap Candidates",
        "",
        "### 4a. All targets — top 15 by gap score",
        "",
        "Note: many targets near the top of this list have zero text mentions.",
        "Their ranking reflects evidence strength, not text-supported gap signals.",
        "See Section 4b for the more defensible subset.",
        "",
        "| Rank | Symbol | Name | Assoc. | Lit | Trial | Gap Score | Tier |",
        "|------|--------|------|--------|-----|-------|-----------|------|",
    ]

    for _, r in top15.iterrows():
        name_s = str(r["approved_name"])[:32] + ("..." if len(str(r["approved_name"])) > 32 else "")
        tier_s = str(r["gap_interpretation_tier"])[:28] + "..."
        lines.append(
            f"| {int(r['gap_rank'])} | {r['approved_symbol']} | {name_s} | "
            f"{r['association_score']:.3f} | {int(r['literature_mentions'])} | "
            f"{int(r['trial_mentions'])} | {r['gap_score']:.4f} | {tier_s} |"
        )

    lines += [
        "",
        "### 4b. Strong text-supported gap candidates",
        "",
        "These candidates have measurable literature presence, low trial text presence,",
        "and high gap scores. They represent the most defensible subset of gap signals",
        "in this framework.",
        "",
        "| Rank | Symbol | Approved Name | Assoc. | Lit | Trial | Gap Score | Ambiguity |",
        "|------|--------|--------------|--------|-----|-------|-----------|-----------|",
    ]

    for _, r in strong_df.iterrows():
        name_s = str(r["approved_name"])[:38]
        lines.append(
            f"| {int(r['gap_rank'])} | {r['approved_symbol']} | {name_s} | "
            f"{r['association_score']:.3f} | {int(r['literature_mentions'])} | "
            f"{int(r['trial_mentions'])} | {r['gap_score']:.4f} | "
            f"{r['symbol_ambiguity_flag']} |"
        )

    if len(strong_df) == 0:
        lines.append("*No targets met the Strong text-supported gap signal criteria.*")

    lines += [
        "",
        "### 4c. Established / trial-covered targets",
        "",
        "These targets have high trial text presence and are less likely to represent",
        "underexplored gaps based on this signal. Targets with moderate symbol ambiguity",
        "in this list should be manually validated for false-positive trial mentions.",
        "",
        "| Symbol | Approved Name | Lit | Trial | Ambiguity | Note |",
        "|--------|--------------|-----|-------|-----------|------|",
    ]

    for _, r in covered_df.iterrows():
        name_s = str(r["approved_name"])[:30]
        ambig_note = "review mentions" if r["symbol_ambiguity_flag"] == "moderate" else "reliable"
        lines.append(
            f"| {r['approved_symbol']} | {name_s} | "
            f"{int(r['literature_mentions'])} | {int(r['trial_mentions'])} | "
            f"{r['symbol_ambiguity_flag']} | {ambig_note} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Score Distribution",
        "",
        f"Gap scores across all {n_total} targets:",
        "",
        f"- Mean:   {df_out['gap_score'].mean():.4f}",
        f"- Median: {df_out['gap_score'].median():.4f}",
        f"- Std:    {df_out['gap_score'].std():.4f}",
        f"- Min:    {df_out['gap_score'].min():.4f}",
        f"- Max:    {df_out['gap_score'].max():.4f}",
        "",
        f"The distribution is strongly right-skewed / positively skewed: {n_zero} targets",
        "have zero literature mentions and therefore zero literature_trial_gap_norm,",
        f"clustering at lower scores, while a smaller set of {n_strong} text-supported",
        "candidates forms the upper tail.",
        "",
        "---",
        "",
        "## 6. Interpretation",
        "",
        "### 6a. Strong text-supported gap candidates",
        "",
    ]

    if len(strong_df) > 0:
        lines.append(
            f"The following {len(strong_df)} targets have literature text signal, "
            f"low trial text presence (trial_mentions <= {trial_90th:.0f}), "
            "and high gap scores. They *may* represent areas where preclinical research "
            "attention has not been matched by clinical trial activity, though this "
            "requires manual validation."
        )
        lines.append("")
        for _, r in strong_df.iterrows():
            ambig_note = (f" [symbol ambiguity = {r['symbol_ambiguity_flag']}; "
                          "review mention counts]"
                          if r["symbol_ambiguity_flag"] == "moderate" else "")
            lines.append(
                f"- **{r['approved_symbol']}** ({str(r['approved_name'])[:50]}): "
                f"lit={int(r['literature_mentions'])}, trial={int(r['trial_mentions'])}, "
                f"assoc={r['association_score']:.3f}, gap={r['gap_score']:.4f}{ambig_note}"
            )
    else:
        lines.append("*No targets met the strong text-supported criteria.*")

    lines += [
        "",
        "### 6b. Established / trial-covered targets",
        "",
        "These targets should not be described as underexplored based on this corpus.",
        "Note that APP, SCD, and ACE have moderate symbol ambiguity — their high trial",
        "mention counts may include false positives (e.g. 'APP' matching 'application',",
        "'SCD' matching 'sickle cell disease', 'ACE' matching 'ACE inhibitor' class).",
        "Manual validation of these counts is recommended before citing them as evidence.",
        "",
        "---",
        "",
        "## 7. Limitations",
        "",
        "- **471 targets had zero text mentions**: This is a lexical absence from this",
        "  specific corpus, not evidence of absent research. The corpus (478 papers,",
        "  1,000 trials) is a sample. These targets receive gap scores driven by",
        "  evidence_strength_norm alone and should not be treated as confirmed gaps.",
        "",
        "- **Gene symbol ambiguity**: Short 3-letter all-alpha symbols (APP, ACE, SCD,",
        "  CLU) may match non-gene uses in clinical text. APP could match 'application',",
        "  SCD could match 'sickle cell disease', ACE could match 'ACE inhibitor' as a",
        "  drug class reference. Mention counts for these symbols should be manually",
        "  verified before citing as evidence of gene-specific research activity.",
        "",
        "- **Trial text does not name molecular targets**: A clinical trial targeting",
        "  TREM2 via AL002 will not contain 'TREM2' in its registration text. Low trial",
        "  mention counts do not imply absence of relevant trials.",
        "",
        "- **Corpus sample**: 478 papers and 1,000 trials are a limited sample of the",
        "  global literature and trial landscape.",
        "",
        "- **Open Targets scores**: association_score summarises curation breadth, not",
        "  therapeutic tractability. High scores do not imply valid drug targets.",
        "",
        "- **No causal inference**: This framework is purely lexical and statistical.",
        "  It is not clinical decision support.",
        "",
        "---",
        "",
        "## 8. Recommended Step 7",
        "",
        "**Step 7: Knowledge Graph Construction**",
        "",
        "```",
        "Disease -> Target -> Paper -> Trial -> Theme",
        "```",
        "",
        "Inputs from Steps 5, 5.5, and 6:",
        "- `research_gap_scores.csv` — gap score, tier, and ambiguity as node attributes",
        "- `strong_text_supported_gap_candidates.csv` — prioritised gap candidates",
        "- `literature_topic_assignments.csv` — paper-to-theme edges",
        "- `clinical_trial_topic_assignments.csv` — trial-to-theme edges",
        "- `target_evidence_clean.csv` — target-to-disease edges with evidence weights",
        "",
        "**Node colouring:** Use `gap_interpretation_tier` as the primary node-colouring",
        "attribute rather than `gap_category` alone. `gap_interpretation_tier` better",
        "separates strong text-supported gap candidates (red) from established /",
        "trial-covered targets (orange), checked-zero targets (grey), and low-priority",
        "candidates (blue), making the knowledge graph immediately interpretable.",
        "",
        "---",
        "",
        "*All outputs are computationally derived. No clinical conclusions should be",
        "drawn from this report without expert domain review.*",
    ]

    report_path = os.path.join(RPT_DIR, "gap_scoring_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. load
    df, topic_data = load_data()

    # 2. compute scores
    df = compute_scores(df)

    # 3. gap categories
    df, method_note, high_thresh, moderate_thresh = assign_categories(df)

    # 4. interpretation tiers
    df, trial_90th = assign_tiers(df, high_thresh)

    # 5. rank + interpretation notes
    df_out = rank_and_finalise(df)

    # 6. save tables
    save_tables(df_out)

    # 7. figures — attach log columns for scatter plots
    df_out = df_out.merge(
        df[["approved_symbol", "_lit_log", "_trial_log"]],
        on="approved_symbol", how="left",
    )
    make_figures(df_out, high_thresh, moderate_thresh, trial_90th)

    # 8. report
    write_report(df_out, method_note, high_thresh, moderate_thresh,
                 trial_90th, topic_data)

    # ── validation ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total targets scored:        {len(df_out):,}")
    print(f"  gap_rank unique:             {df_out['gap_rank'].nunique() == len(df_out)}")
    print(f"  Missing approved_symbol:     {df_out['approved_symbol'].isna().sum()}")

    norm_cols = ["evidence_strength_norm", "literature_signal_norm",
                 "trial_signal_norm", "literature_trial_gap_norm",
                 "gap_score", "conservative_gap_score"]
    norm_cols = [c for c in norm_cols if c in df_out.columns]
    all_in_range = all(df_out[c].between(0, 1).all() for c in norm_cols)
    print(f"  All score cols in [0,1]:     {all_in_range}")

    tiers = df_out["gap_interpretation_tier"].value_counts()
    print(f"\n  Tier counts:")
    for t, n in tiers.items():
        print(f"    {t}: {n}")

    # top 10 overall
    print(f"\n  Top 10 overall gap candidates:")
    hdr = f"  {'Rank':>4}  {'Symbol':<10}  {'Assoc':>7}  {'Lit':>5}  {'Trial':>5}  {'GapScore':>9}  Tier"
    print(hdr)
    print("  " + "-" * 80)
    for _, r in df_out.head(10).iterrows():
        print(
            f"  {int(r['gap_rank']):>4}  {r['approved_symbol']:<10}  "
            f"{r['association_score']:>7.3f}  "
            f"{int(r['literature_mentions']):>5}  {int(r['trial_mentions']):>5}  "
            f"{r['gap_score']:>9.4f}  {r['gap_interpretation_tier'][:30]}"
        )

    # top 10 strong text-supported
    strong_df = df_out[df_out["gap_interpretation_tier"] == "Strong text-supported gap signal"]
    print(f"\n  Top 10 STRONG TEXT-SUPPORTED gap candidates ({len(strong_df)} total):")
    print(hdr)
    print("  " + "-" * 80)
    for _, r in strong_df.head(10).iterrows():
        print(
            f"  {int(r['gap_rank']):>4}  {r['approved_symbol']:<10}  "
            f"{r['association_score']:>7.3f}  "
            f"{int(r['literature_mentions']):>5}  {int(r['trial_mentions']):>5}  "
            f"{r['gap_score']:>9.4f}  {r['gap_interpretation_tier'][:30]}"
        )

    saved_files = [
        os.path.join(DATA_DIR, "research_gap_scores.csv"),
        os.path.join(DATA_DIR, "top_research_gap_candidates.csv"),
        os.path.join(DATA_DIR, "strong_text_supported_gap_candidates.csv"),
        os.path.join(DATA_DIR, "top_text_supported_gap_candidates.csv"),
        os.path.join(DATA_DIR, "gap_score_components_summary.csv"),
        os.path.join(FIG_DIR,  "gap_01_top_gap_candidates.png"),
        os.path.join(FIG_DIR,  "gap_02_score_distribution.png"),
        os.path.join(FIG_DIR,  "gap_03_evidence_vs_trial_mentions.png"),
        os.path.join(FIG_DIR,  "gap_04_literature_vs_trial_mentions.png"),
        os.path.join(FIG_DIR,  "gap_05_score_component_heatmap.png"),
        os.path.join(FIG_DIR,  "gap_06_interpretation_tier_breakdown.png"),
        os.path.join(RPT_DIR,  "gap_scoring_summary.md"),
    ]
    print("\n  Output files:")
    for p in saved_files:
        status = "OK" if os.path.isfile(p) else "MISSING"
        print(f"    [{status}] {os.path.relpath(p, BASE_DIR)}")

    print("\nStep 6 (full-match revision) complete.")


if __name__ == "__main__":
    main()
