"""
NeuroGraph Agent — Exploratory Data Analysis
----------------------------------------------
Reads  : Data/processed/clinical_trials_clean.csv
         Data/processed/literature_clean.csv
         Data/processed/target_evidence_clean.csv

Writes : outputs/figures/   (15 PNG charts)
         outputs/reports/eda_summary.md

Usage:
    python src/processing/run_eda.py
    python -m src.processing.run_eda
"""

import os
import sys
import textwrap
from collections import Counter

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Windows Unicode fix ─────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_PROCESSED_DIR

# ── Output paths ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.dirname(__file__), "..", "..")
FIGURES_DIR  = os.path.abspath(os.path.join(BASE_DIR, "outputs", "figures"))
REPORTS_DIR  = os.path.abspath(os.path.join(BASE_DIR, "outputs", "reports"))
REPORT_FILE  = os.path.join(REPORTS_DIR, "eda_summary.md")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Colour palette (matplotlib named colours, no seaborn) ──────────────────────
C_BLUE   = "#2166ac"
C_TEAL   = "#35978f"
C_ORANGE = "#d6604d"
C_PURPLE = "#762a83"
C_GREY   = "#878787"
C_GREEN  = "#4dac26"


# ── Unicode-safe print helper ───────────────────────────────────────────────────
def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(str(text).encode("ascii", errors="replace").decode("ascii"))


# ── Helpers ─────────────────────────────────────────────────────────────────────

def fig_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, name)


def save_fig(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: outputs/figures/{filename}")


def col(df: pd.DataFrame, name: str, default=None) -> pd.Series:
    """Return a column if it exists, else a Series of the default value."""
    if name in df.columns:
        return df[name]
    print(f"    [WARN] Column '{name}' not found — skipping.")
    return pd.Series([default] * len(df), name=name)


def split_pipe(series: pd.Series) -> Counter:
    """Explode a '|'-delimited string column into a flat Counter."""
    counts: Counter = Counter()
    for val in series.dropna():
        for item in str(val).split("|"):
            item = item.strip()
            if item:
                counts[item] += 1
    return counts


def missingness_bar(df: pd.DataFrame, title: str, filename: str) -> dict:
    """Plot a horizontal bar chart of missing-value percentages."""
    miss_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    miss_pct = miss_pct[miss_pct > 0]

    if miss_pct.empty:
        print(f"    No missing values in {title} — skipping missingness chart.")
        return {}

    fig, ax = plt.subplots(figsize=(9, max(3, len(miss_pct) * 0.45)))
    colours = [C_ORANGE if v > 30 else C_TEAL for v in miss_pct.values]
    ax.barh(miss_pct.index, miss_pct.values, color=colours)
    ax.set_xlabel("% Missing", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(30, color=C_GREY, linestyle="--", linewidth=0.8, label="30% threshold")
    ax.legend(fontsize=9)
    for i, (label, val) in enumerate(zip(miss_pct.index, miss_pct.values)):
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=8)
    save_fig(filename)
    return miss_pct.to_dict()


# ════════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple:
    print("\n[EDA] Loading datasets...")

    ct_path  = os.path.join(DATA_PROCESSED_DIR, "clinical_trials_clean.csv")
    lit_path = os.path.join(DATA_PROCESSED_DIR, "literature_clean.csv")
    ot_path  = os.path.join(DATA_PROCESSED_DIR, "target_evidence_clean.csv")

    for path in [ct_path, lit_path, ot_path]:
        assert os.path.exists(path), f"File not found: {path}"

    df_ct  = pd.read_csv(ct_path,  encoding="utf-8")
    df_lit = pd.read_csv(lit_path, encoding="utf-8")
    df_ot  = pd.read_csv(ot_path,  encoding="utf-8")

    # Parse dates in clinical trials
    for dcol in ["start_date", "completion_date"]:
        if dcol in df_ct.columns:
            df_ct[dcol] = pd.to_datetime(df_ct[dcol], errors="coerce")

    print(f"  Clinical Trials : {df_ct.shape[0]:,} rows x {df_ct.shape[1]} columns")
    print(f"  Literature      : {df_lit.shape[0]:,} rows x {df_lit.shape[1]} columns")
    print(f"  Target Evidence : {df_ot.shape[0]:,} rows x {df_ot.shape[1]} columns")

    return df_ct, df_lit, df_ot


# ════════════════════════════════════════════════════════════════════════════════
# 2. CLINICAL TRIALS EDA
# ════════════════════════════════════════════════════════════════════════════════

def eda_clinical_trials(df: pd.DataFrame) -> dict:
    print("\n[EDA] Clinical Trials...")
    stats = {}

    # ── 2a. Trial status distribution ─────────────────────────────────────────
    status = col(df, "overall_status").fillna("Unknown").value_counts()
    stats["status"] = status.to_dict()

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = [C_TEAL if s == "COMPLETED" else
               C_BLUE if s == "RECRUITING" else
               C_ORANGE if s in ("TERMINATED", "WITHDRAWN") else
               C_GREY for s in status.index]
    ax.bar(status.index, status.values, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_title("Clinical Trial Status Distribution\n(Alzheimer's Disease, ClinicalTrials.gov)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Status", fontsize=11)
    ax.set_ylabel("Number of Trials", fontsize=11)
    ax.tick_params(axis="x", rotation=35)
    for i, v in enumerate(status.values):
        ax.text(i, v + 3, str(v), ha="center", fontsize=9)
    save_fig("ct_01_status_distribution.png")

    # ── 2b. Phase distribution ─────────────────────────────────────────────────
    phase_raw = col(df, "phases").fillna("Not Specified")
    phase_raw = phase_raw.replace("", "Not Specified")
    phase_counts = phase_raw.value_counts()
    stats["phases"] = phase_counts.to_dict()

    phase_label_map = {
        "PHASE1": "Phase 1", "PHASE2": "Phase 2", "PHASE3": "Phase 3",
        "PHASE4": "Phase 4", "PHASE1|PHASE2": "Phase 1/2",
        "PHASE2|PHASE3": "Phase 2/3", "EARLY_PHASE1": "Early Phase 1",
        "NA": "Not Applicable", "Not Specified": "Not Specified",
    }
    phase_counts.index = [phase_label_map.get(p, p) for p in phase_counts.index]
    phase_counts = phase_counts.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colours = [C_BLUE if "Phase" in str(p) and p != "Not Applicable" else C_GREY
                   for p in phase_counts.index]
    ax.bar(phase_counts.index, phase_counts.values,
           color=bar_colours, edgecolor="white", linewidth=0.5)
    ax.set_title("Trial Phase Distribution\n(Alzheimer's Disease, ClinicalTrials.gov)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Phase", fontsize=11)
    ax.set_ylabel("Number of Trials", fontsize=11)
    ax.tick_params(axis="x", rotation=35)
    for i, v in enumerate(phase_counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontsize=9)
    save_fig("ct_02_phase_distribution.png")

    # ── 2c. Sponsor class distribution ────────────────────────────────────────
    if "sponsor_class" in df.columns:
        sponsor_class = df["sponsor_class"].fillna("Unknown").value_counts()

        label_map = {
            "OTHER": "Academic / Non-Profit", "INDUSTRY": "Industry",
            "NIH": "NIH", "OTHER_GOV": "Other Government",
            "FED": "Federal Agency", "NETWORK": "Network",
        }
        sponsor_class.index = [label_map.get(s, s) for s in sponsor_class.index]
        # Store AFTER relabelling so write_report can look up "Industry", "Academic / Non-Profit" etc.
        stats["sponsor_class"] = sponsor_class.to_dict()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(sponsor_class.index, sponsor_class.values,
               color=[C_TEAL, C_ORANGE, C_BLUE, C_PURPLE, C_GREY, C_GREEN][:len(sponsor_class)],
               edgecolor="white", linewidth=0.5)
        ax.set_title("Sponsor Type Distribution\n(Alzheimer's Disease Trials)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Sponsor Class", fontsize=11)
        ax.set_ylabel("Number of Trials", fontsize=11)
        ax.tick_params(axis="x", rotation=25)
        for i, v in enumerate(sponsor_class.values):
            ax.text(i, v + 3, str(v), ha="center", fontsize=9)
        save_fig("ct_03_sponsor_class.png")

    # ── 2d. Top countries ──────────────────────────────────────────────────────
    if "countries" in df.columns:
        country_counts = split_pipe(df["countries"])
        top_countries = pd.Series(country_counts).nlargest(20)
        stats["top_countries"] = top_countries.to_dict()

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top_countries.index[::-1], top_countries.values[::-1],
                color=C_BLUE, edgecolor="white", linewidth=0.5)
        ax.set_title("Top 20 Countries by Trial Count\n(Alzheimer's Disease Trials)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Trials", fontsize=11)
        ax.set_ylabel("Country", fontsize=11)
        for i, v in enumerate(top_countries.values[::-1]):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
        save_fig("ct_04_top_countries.png")

    # ── 2e. Enrolment distribution ────────────────────────────────────────────
    if "enrollment_count" in df.columns:
        enrol = df["enrollment_count"].dropna()
        enrol = enrol[enrol > 0]
        stats["enrollment"] = {
            "count": int(enrol.count()), "min": int(enrol.min()),
            "median": int(enrol.median()), "mean": float(enrol.mean()),
            "max": int(enrol.max()),
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(enrol.clip(upper=1000), bins=40, color=C_TEAL, edgecolor="white")
        axes[0].set_title("Enrolment Count Distribution\n(clipped at 1,000)",
                          fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Enrolment Count", fontsize=10)
        axes[0].set_ylabel("Number of Trials", fontsize=10)

        axes[1].hist(np.log10(enrol + 1), bins=40, color=C_ORANGE, edgecolor="white")
        axes[1].set_title("Enrolment Count (log10 scale)",
                          fontsize=11, fontweight="bold")
        axes[1].set_xlabel("log10(Enrolment + 1)", fontsize=10)
        axes[1].set_ylabel("Number of Trials", fontsize=10)

        plt.suptitle("Trial Enrolment Distribution — Alzheimer's Disease",
                     fontsize=13, fontweight="bold", y=1.02)
        save_fig("ct_05_enrollment_distribution.png")

    # ── 2f. Trial start year trend ────────────────────────────────────────────
    if "start_date" in df.columns:
        start_years = df["start_date"].dt.year.dropna().astype(int)
        year_counts = start_years.value_counts().sort_index()
        year_counts = year_counts[(year_counts.index >= 1990) & (year_counts.index <= 2026)]
        stats["start_years"] = year_counts.to_dict()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(year_counts.index, year_counts.values, color=C_PURPLE, edgecolor="white", linewidth=0.5)
        ax.set_title("Alzheimer's Clinical Trials by Start Year\n(ClinicalTrials.gov registered trials)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Start Year", fontsize=11)
        ax.set_ylabel("Number of Trials", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        save_fig("ct_06_trials_by_year.png")

    # ── 2g. Missingness summary ────────────────────────────────────────────────
    key_cols = ["nct_id", "brief_title", "overall_status", "start_date",
                "completion_date", "phases", "primary_purpose", "sponsor_name",
                "sponsor_class", "interventions", "countries", "enrollment_count",
                "minimum_age", "maximum_age"]
    key_cols_present = [c for c in key_cols if c in df.columns]
    miss_stats = missingness_bar(
        df[key_cols_present],
        "Missing Values — Clinical Trials Dataset\n(Key columns only)",
        "ct_07_missingness.png",
    )
    stats["missingness"] = miss_stats

    return stats


# ════════════════════════════════════════════════════════════════════════════════
# 3. LITERATURE EDA
# ════════════════════════════════════════════════════════════════════════════════

def eda_literature(df: pd.DataFrame) -> dict:
    print("\n[EDA] Literature...")
    stats = {}

    # ── 3a. Papers by year ────────────────────────────────────────────────────
    if "year" in df.columns:
        year_counts = df["year"].value_counts().sort_index()
        year_counts = year_counts[(year_counts.index >= 2015) & (year_counts.index <= 2026)]
        stats["papers_by_year"] = year_counts.to_dict()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(year_counts.index, year_counts.values, color=C_BLUE, edgecolor="white", linewidth=0.5)
        ax.set_title("Alzheimer's Research Papers by Year\n(Semantic Scholar, 2015-2025)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Publication Year", fontsize=11)
        ax.set_ylabel("Number of Papers", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        for i, (yr, cnt) in enumerate(zip(year_counts.index, year_counts.values)):
            ax.text(yr, cnt + 0.5, str(cnt), ha="center", fontsize=8)
        save_fig("lit_01_papers_by_year.png")

    # ── 3b. Citation count distribution ──────────────────────────────────────
    if "citation_count" in df.columns:
        cit = df["citation_count"].dropna()
        cit = cit[cit >= 0]
        stats["citations"] = {
            "min": int(cit.min()), "max": int(cit.max()),
            "mean": round(float(cit.mean()), 1),
            "median": int(cit.median()),
            "p25": int(cit.quantile(0.25)),
            "p75": int(cit.quantile(0.75)),
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(cit.clip(upper=2000), bins=40, color=C_TEAL, edgecolor="white")
        axes[0].set_title("Citation Count Distribution\n(clipped at 2,000)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Citation Count", fontsize=10)
        axes[0].set_ylabel("Number of Papers", fontsize=10)

        axes[1].hist(np.log10(cit + 1), bins=40, color=C_ORANGE, edgecolor="white")
        axes[1].set_title("Citation Count (log10 scale)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("log10(Citations + 1)", fontsize=10)
        axes[1].set_ylabel("Number of Papers", fontsize=10)

        plt.suptitle("Alzheimer's Paper Citation Distributions — Semantic Scholar",
                     fontsize=13, fontweight="bold", y=1.02)
        save_fig("lit_02_citation_distribution.png")

    # ── 3c. Top venues ────────────────────────────────────────────────────────
    if "venue" in df.columns:
        venue_counts = df["venue"].dropna().replace("", np.nan).dropna().value_counts().head(15)
        stats["top_venues"] = venue_counts.to_dict()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(venue_counts.index[::-1], venue_counts.values[::-1],
                color=C_BLUE, edgecolor="white", linewidth=0.5)
        ax.set_title("Top 15 Publication Venues\n(Alzheimer's Disease Papers, Semantic Scholar)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Papers", fontsize=11)
        for i, v in enumerate(venue_counts.values[::-1]):
            ax.text(v + 0.2, i, str(v), va="center", fontsize=8)
        save_fig("lit_03_top_venues.png")

    # ── 3d. Fields of study ───────────────────────────────────────────────────
    if "fields_of_study" in df.columns:
        field_counts = split_pipe(df["fields_of_study"])
        top_fields = pd.Series(field_counts).nlargest(10)
        stats["fields_of_study"] = top_fields.to_dict()

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(top_fields.index[::-1], top_fields.values[::-1],
                color=C_PURPLE, edgecolor="white", linewidth=0.5)
        ax.set_title("Fields of Study Distribution\n(Alzheimer's Disease Papers, Semantic Scholar)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Papers", fontsize=11)
        for i, v in enumerate(top_fields.values[::-1]):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
        save_fig("lit_04_fields_of_study.png")

    # ── 3e. Most cited papers table (printed, not plotted) ────────────────────
    if "citation_count" in df.columns and "title" in df.columns:
        top_papers = df.nlargest(10, "citation_count")[
            ["title", "year", "citation_count", "venue"]
        ].copy()
        top_papers["title"] = top_papers["title"].apply(
            lambda t: textwrap.shorten(str(t), width=60, placeholder="...") if pd.notna(t) else ""
        )
        stats["top_cited_papers"] = top_papers.to_dict(orient="records")
        safe_print("\n  Top 10 most cited papers:")
        for _, r in top_papers.iterrows():
            safe_print(
                f"    [{int(r['year']) if pd.notna(r['year']) else '?'}] "
                f"({int(r['citation_count'])} cit.) {r['title']}"
            )

    # ── 3f. Missingness summary ────────────────────────────────────────────────
    key_cols = ["paper_id", "title", "abstract", "year",
                "citation_count", "venue", "doi", "pmid", "authors", "fields_of_study"]
    key_cols_present = [c for c in key_cols if c in df.columns]
    miss_stats = missingness_bar(
        df[key_cols_present],
        "Missing Values — Literature Dataset\n(Key columns only)",
        "lit_05_missingness.png",
    )
    stats["missingness"] = miss_stats

    return stats


# ════════════════════════════════════════════════════════════════════════════════
# 4. OPEN TARGETS EDA
# ════════════════════════════════════════════════════════════════════════════════

def eda_open_targets(df: pd.DataFrame) -> dict:
    print("\n[EDA] Open Targets...")
    stats = {}

    ds_cols = [c for c in df.columns if c.startswith("ds_")]

    ds_label_map = {
        "ds_europepmc":          "Europe PMC (literature)",
        "ds_crispr_screen":      "CRISPR Screen",
        "ds_gwas_credible_sets": "GWAS Credible Sets",
        "ds_expression_atlas":   "Expression Atlas",
        "ds_clinical_precedence":"Clinical Precedence",
        "ds_reactome":           "Reactome (pathways)",
        "ds_eva":                "EVA (clinical variants)",
        "ds_impc":               "IMPC (mouse models)",
        "ds_uniprot_variants":   "UniProt Variants",
        "ds_uniprot_literature": "UniProt Literature",
        "ds_genomics_england":   "Genomics England",
        "ds_orphanet":           "Orphanet",
        "ds_gene_burden":        "Gene Burden",
    }

    # ── 4a. Top targets by association score ──────────────────────────────────
    if "approved_symbol" in df.columns and "association_score" in df.columns:
        top_n = 25
        top_targets = df.nlargest(top_n, "association_score").copy()
        stats["top_targets"] = top_targets[
            ["approved_symbol", "approved_name", "association_score"]
        ].to_dict(orient="records")

        safe_print("\n  Top 10 targets by association score:")
        for _, r in top_targets.head(10).iterrows():
            safe_print(
                f"    {r['approved_symbol']:<12} {r['association_score']:.4f}  {r['approved_name']}"
            )

        fig, ax = plt.subplots(figsize=(10, 8))
        colours_bar = [C_ORANGE if s >= 0.65 else C_BLUE if s >= 0.45 else C_TEAL
                       for s in top_targets["association_score"]]
        ax.barh(
            top_targets["approved_symbol"][::-1],
            top_targets["association_score"][::-1],
            color=colours_bar[::-1], edgecolor="white", linewidth=0.5,
        )
        ax.set_title(f"Top {top_n} Gene/Protein Targets — Alzheimer's Disease\n"
                     "(Open Targets Platform, association score)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Association Score (0-1)", fontsize=11)
        ax.set_xlim(0, 1.0)
        ax.axvline(0.65, color=C_ORANGE, linestyle="--", linewidth=0.8,
                   label="High evidence (>=0.65)")
        ax.axvline(0.45, color=C_BLUE, linestyle="--", linewidth=0.8,
                   label="Moderate evidence (>=0.45)")
        ax.legend(fontsize=9, loc="lower right")
        save_fig("ot_01_top_targets.png")

    # ── 4b. Association score distribution ────────────────────────────────────
    if "association_score" in df.columns:
        scores = df["association_score"].dropna()
        stats["score_stats"] = {
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "mean": round(float(scores.mean()), 4),
            "median": round(float(scores.median()), 4),
            "p75": round(float(scores.quantile(0.75)), 4),
            "p90": round(float(scores.quantile(0.90)), 4),
        }

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(scores, bins=40, color=C_TEAL, edgecolor="white")
        ax.axvline(scores.median(), color=C_ORANGE, linestyle="--", linewidth=1.5,
                   label=f"Median: {scores.median():.3f}")
        ax.axvline(scores.mean(), color=C_BLUE, linestyle="--", linewidth=1.5,
                   label=f"Mean: {scores.mean():.3f}")
        ax.set_title("Association Score Distribution — Alzheimer's Disease Targets\n"
                     "(Open Targets Platform)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Association Score (0-1)", fontsize=11)
        ax.set_ylabel("Number of Targets", fontsize=11)
        ax.legend(fontsize=10)
        save_fig("ot_02_score_distribution.png")

    # ── 4c. Datasource coverage ────────────────────────────────────────────────
    if ds_cols:
        coverage_pct = pd.Series(
            {c: 100 * df[c].notna().sum() / len(df) for c in ds_cols}
        ).sort_values(ascending=True)

        coverage_pct.index = [ds_label_map.get(c, c.replace("ds_", "")) for c in coverage_pct.index]
        stats["datasource_coverage"] = coverage_pct.round(1).to_dict()

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_cols = [C_ORANGE if v >= 50 else C_BLUE if v >= 20 else C_GREY
                    for v in coverage_pct.values]
        ax.barh(coverage_pct.index, coverage_pct.values,
                color=bar_cols, edgecolor="white", linewidth=0.5)
        ax.set_title("Evidence Source Coverage\n"
                     "(% of 500 Alzheimer's Targets with score from each source)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("% Targets with Evidence from Source", fontsize=11)
        ax.axvline(50, color=C_GREY, linestyle="--", linewidth=0.8, label="50% threshold")
        ax.legend(fontsize=9)
        for i, v in enumerate(coverage_pct.values):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
        save_fig("ot_03_datasource_coverage.png")

    # ── 4d. Datasource heatmap — top 20 targets x key datasources ─────────────
    heatmap_ds = [c for c in [
        "ds_europepmc", "ds_crispr_screen", "ds_gwas_credible_sets",
        "ds_expression_atlas", "ds_clinical_precedence",
        "ds_reactome", "ds_eva",
    ] if c in df.columns]

    if heatmap_ds and "approved_symbol" in df.columns:
        top20 = df.nlargest(20, "association_score").set_index("approved_symbol")
        heat_data = top20[heatmap_ds].copy()
        heat_data.columns = [ds_label_map.get(c, c.replace("ds_", "")) for c in heat_data.columns]

        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.imshow(heat_data.values.astype(float), aspect="auto",
                       cmap="Blues", vmin=0, vmax=1)

        ax.set_xticks(range(len(heat_data.columns)))
        ax.set_xticklabels(heat_data.columns, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(len(heat_data.index)))
        ax.set_yticklabels(heat_data.index, fontsize=9)
        ax.set_title("Datasource Evidence Heatmap — Top 20 Alzheimer's Targets\n"
                     "(Blue intensity = evidence score; white = no evidence from that source)",
                     fontsize=12, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Datasource Score (0-1)", fraction=0.03, pad=0.04)

        for i in range(len(heat_data.index)):
            for j in range(len(heat_data.columns)):
                val = heat_data.values[i, j]
                if not np.isnan(val):
                    text_colour = "white" if val > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=text_colour)

        save_fig("ot_04_datasource_heatmap.png")

    # ── 4e. Missingness summary ────────────────────────────────────────────────
    core_cols = ["target_id", "approved_symbol", "approved_name", "biotype",
                 "association_score"] + ds_cols
    core_cols_present = [c for c in core_cols if c in df.columns]
    miss_stats = missingness_bar(
        df[core_cols_present],
        "Missing Values — Target Evidence Dataset\n(Core + datasource columns)",
        "ot_05_missingness.png",
    )
    stats["missingness"] = miss_stats

    # ── 4f. Read total_available from raw JSON if present ─────────────────────
    import json as _json
    raw_ot_path = os.path.join(BASE_DIR, "Data", "raw", "open_targets_raw.json")
    stats["total_available"] = None
    stats["total_fetched"] = len(df)
    if os.path.exists(raw_ot_path):
        try:
            with open(raw_ot_path, encoding="utf-8") as _f:
                _raw = _json.load(_f)
            stats["total_available"] = _raw.get("total_available")
        except Exception:
            pass

    return stats


# ════════════════════════════════════════════════════════════════════════════════
# 5. WRITE EDA SUMMARY REPORT
# ════════════════════════════════════════════════════════════════════════════════

def write_report(df_ct, df_lit, df_ot, ct_stats, lit_stats, ot_stats) -> None:
    print("\n[EDA] Writing summary report...")

    ct_status       = ct_stats.get("status", {})
    ct_phases       = ct_stats.get("phases", {})
    ct_sponsors     = ct_stats.get("sponsor_class", {})
    ct_enrol        = ct_stats.get("enrollment", {})

    lit_years       = lit_stats.get("papers_by_year", {})
    lit_cit         = lit_stats.get("citations", {})
    lit_venues      = lit_stats.get("top_venues", {})
    lit_top_papers  = lit_stats.get("top_cited_papers", [])

    ot_score        = ot_stats.get("score_stats", {})
    ot_coverage     = ot_stats.get("datasource_coverage", {})
    ot_top_targets  = ot_stats.get("top_targets", [])
    ot_total_avail  = ot_stats.get("total_available")   # may be None if raw JSON unavailable
    ot_total_fetched = ot_stats.get("total_fetched", len(df_ot))

    completed_pct   = round(100 * ct_status.get("COMPLETED", 0) / len(df_ct), 1)
    recruiting_pct  = round(100 * ct_status.get("RECRUITING", 0) / len(df_ct), 1)
    terminated_pct  = round(100 * ct_status.get("TERMINATED", 0) / len(df_ct), 1)
    industry_n      = ct_sponsors.get("Industry", 0)
    academic_n      = ct_sponsors.get("Academic / Non-Profit", 0)
    industry_pct    = round(100 * industry_n / len(df_ct), 1)
    academic_pct    = round(100 * academic_n / len(df_ct), 1)

    has_abstract    = int((df_lit["abstract"].fillna("").str.strip() != "").sum())
    missing_abstract = len(df_lit) - has_abstract
    abstract_pct    = round(100 * has_abstract / len(df_lit), 1)
    missing_abstract_pct = round(100 * missing_abstract / len(df_lit), 1)

    top_target_rows = ot_top_targets[:5]

    top_venue_name  = list(lit_venues.keys())[0] if lit_venues else "N/A"
    top_venue_count = list(lit_venues.values())[0] if lit_venues else "N/A"

    crispr_cov      = ot_coverage.get("CRISPR Screen", 0)
    europepmc_cov   = ot_coverage.get("Europe PMC (literature)", 0)
    clin_prec_cov   = ot_coverage.get("Clinical Precedence", 0)

    # Dynamic peak year from literature data
    peak_year = max(lit_years, key=lit_years.get) if lit_years else "N/A"
    lit_year_min = min(lit_years.keys()) if lit_years else "N/A"
    lit_year_max = max(lit_years.keys()) if lit_years else "N/A"

    # Phase counts (raw keys before relabelling)
    phase_not_specified = ct_phases.get("Not Specified", 0)
    phase2_n  = ct_phases.get("PHASE2", 0)
    phase1_n  = ct_phases.get("PHASE1", 0)
    phase3_n  = ct_phases.get("PHASE3", 0)
    phase_interventional = sum(
        v for k, v in ct_phases.items()
        if k not in ("Not Specified", "NA", "Not Applicable")
    )

    # Dynamic OT truncation note
    if ot_total_avail is not None:
        ot_truncation_note = (
            f"Only the top {ot_total_fetched:,} associations by score were retrieved "
            f"out of {ot_total_avail:,} available in Open Targets for this disease. "
            "Lower-scoring targets that may represent novel hypotheses are not included in this analysis."
        )
    else:
        ot_truncation_note = (
            f"Only the first {ot_total_fetched:,} target associations retrieved by the API were analysed. "
            "Lower-scoring targets that may represent novel hypotheses are not included in this analysis."
        )

    # Top target rows as Markdown table rows
    target_table_rows = "".join(
        f"| {i+1} | {r.get('approved_symbol','?')} | "
        f"{r.get('approved_name','?')} | "
        f"{r.get('association_score', 0):.4f} |\n"
        for i, r in enumerate(top_target_rows)
    )

    report = f"""# NeuroGraph Agent — Exploratory Data Analysis Summary

## 1. Project Context

NeuroGraph Agent is an agentic biomedical research intelligence system focused on Alzheimer's disease. This EDA combines three independently collected public datasets:

- **ClinicalTrials.gov** — registered clinical trials for Alzheimer's disease
- **Semantic Scholar** — scholarly literature records on Alzheimer's disease
- **Open Targets Platform** — gene/protein target-disease association evidence

The outputs here are exploratory and descriptive only. This project does not make clinical claims, does not constitute clinical decision support, and all findings should be treated as hypothesis-generating research intelligence requiring further validation.

---

## 2. Datasets Analysed

| Dataset | Rows | Columns | Source |
|---|---|---|---|
| Clinical Trials | {len(df_ct):,} | {df_ct.shape[1]} | ClinicalTrials.gov API v2 |
| Literature | {len(df_lit):,} | {df_lit.shape[1]} | Semantic Scholar API |
| Target Evidence | {len(df_ot):,} | {df_ot.shape[1]} | Open Targets GraphQL API |

---

## 3. Clinical Trial Landscape

### Status Distribution

Of {len(df_ct):,} trials in the dataset:
- **{ct_status.get('COMPLETED', 0):,} ({completed_pct}%) are COMPLETED** — indicating a historically active research landscape
- **{ct_status.get('RECRUITING', 0):,} ({recruiting_pct}%) are RECRUITING** — representing the active trial pipeline
- **{ct_status.get('TERMINATED', 0):,} ({terminated_pct}%) are TERMINATED** — this attrition rate may reflect challenges in Alzheimer's drug development, though termination reasons vary and require further investigation
- **{ct_status.get('UNKNOWN', 0):,} trials have UNKNOWN status** — a data quality limitation worth noting in downstream analysis

### Phase Distribution

Phase information is not recorded for {phase_not_specified:,} of {len(df_ct):,} trials ({round(100*phase_not_specified/len(df_ct),1)}%) — these are predominantly observational studies, registries, and expanded-access records. Among the {phase_interventional:,} trials with a recorded interventional phase, Phase 2 is the most common ({phase2_n} trials), followed by Phase 1 ({phase1_n}). The smaller Phase 3 count ({phase3_n}) compared to Phase 2 is consistent with the high attrition rate typically observed between phases in Alzheimer's drug development, though this dataset alone is insufficient to quantify that attrition.

### Sponsor Type

Academic institutions and non-profits represent the largest funder class ({academic_n:,} trials, {academic_pct}%), with industry sponsors accounting for {industry_n:,} trials ({industry_pct}%). This pattern suggests that early exploratory research may be largely publicly funded, while commercial sponsors could be more concentrated at later-stage trials — though this requires further validation against phase-stratified data.

### Geographic Distribution

The United States leads in trial count by a significant margin. The geographic spread indicates global research interest, but may also reflect ClinicalTrials.gov registration biases towards US-based research.

### Enrolment

Median trial enrolment is {ct_enrol.get('median', 'N/A')}. The distribution is highly right-skewed — a small number of large Phase 3 trials drive the upper tail.

### Data Quality Issues

- `start_date` missing in {df_ct['start_date'].isna().sum()} of {len(df_ct):,} trials ({round(100*df_ct['start_date'].isna().sum()/len(df_ct),1)}%)
- `completion_date` missing in {df_ct['completion_date'].isna().sum()} trials ({round(100*df_ct['completion_date'].isna().sum()/len(df_ct),1)}%)
- `phases` missing in {df_ct['phases'].isna().sum()} trials ({round(100*df_ct['phases'].isna().sum()/len(df_ct),1)}%) — predominantly observational studies

---

## 4. Literature Landscape

### Publication Year Trends

Papers in this dataset range from {lit_year_min} to {lit_year_max}. Publication volume in this sample peaks at {peak_year} ({lit_years.get(peak_year, 'N/A')} records). Counts for recent years ({int(lit_year_max) - 1}–{lit_year_max}) are notably lower, which may reflect indexing lag in the Semantic Scholar API — recently published papers take time to be indexed and have had less time to accumulate citations. This should be treated as a retrieval artefact rather than a genuine decline in research activity.

### Citation Patterns

| Statistic | Value |
|---|---|
| Minimum | {lit_cit.get('min', 'N/A'):,} |
| Maximum | {lit_cit.get('max', 'N/A'):,} |
| Mean | {lit_cit.get('mean', 'N/A'):,} |
| Median | {lit_cit.get('median', 'N/A'):,} |
| 25th Percentile | {lit_cit.get('p25', 'N/A'):,} |
| 75th Percentile | {lit_cit.get('p75', 'N/A'):,} |

The citation distribution is strongly right-skewed. The relatively high median ({lit_cit.get('median', 'N/A')}) suggests this API sample may be skewed toward highly cited and prominent papers, which could reflect Semantic Scholar's retrieval or ranking behaviour rather than the full distribution of Alzheimer's research output. This dataset does not represent the complete Alzheimer's literature landscape and citation counts should be interpreted with that context in mind.

### Venue Patterns

`{top_venue_name}` is the most represented venue ({top_venue_count} papers), reflecting the field's dedicated journal infrastructure. High-impact general journals (Nature, Lancet Neurology, Nature Medicine) are also well-represented, indicating that Alzheimer's research reaches broad scientific audiences.

### Abstract Missingness

{has_abstract} of {len(df_lit)} papers ({abstract_pct}%) have an abstract. The {missing_abstract} ({missing_abstract_pct}%) without abstracts will contribute reduced signal in downstream NLP analysis. These records can still contribute year, venue, and citation data to bibliometric analysis.

### Fields of Study

Medicine dominates the field classification, with a small number of papers classified under Biology, Computer Science, and Chemistry. This suggests the dataset is primarily clinical and translational rather than basic science — relevant to note for TF-IDF theme extraction.

---

## 5. Target Evidence Landscape

### Top Targets / Genes

The highest-scoring targets by Alzheimer's association evidence are:

| Rank | Gene | Full Name | Association Score |
|---|---|---|---|
{target_table_rows}
APP, PSEN1, and PSEN2 are the canonical amyloid pathway genes, while APOE represents the strongest common genetic risk factor. Their top-ranking positions are consistent with known Alzheimer's research themes and suggest the Open Targets query is returning well-supported targets. However, a high association score indicates that a target is well-represented in available evidence — it does not imply therapeutic success or causal proof.

### Association Score Distribution

| Statistic | Value |
|---|---|
| Minimum | {ot_score.get('min', 'N/A')} |
| Maximum | {ot_score.get('max', 'N/A')} |
| Mean | {ot_score.get('mean', 'N/A')} |
| Median | {ot_score.get('median', 'N/A')} |
| 75th Percentile | {ot_score.get('p75', 'N/A')} |
| 90th Percentile | {ot_score.get('p90', 'N/A')} |

The distribution is right-skewed with the majority of targets clustering near the lower bound. Only a small proportion of targets have strong multi-source evidence. This skew is expected — it indicates a core of targets that are well-supported by available evidence, and a long tail of lower-confidence candidates that may represent underexplored research opportunities requiring further validation.

### Datasource Evidence Patterns

The most prevalent evidence source is **Europe PMC** ({europepmc_cov:.1f}% of targets), indicating that most associations are literature-derived. **CRISPR Screen** data is the second most common ({crispr_cov:.1f}%), reflecting growing use of genome-wide screens in neurodegeneration research. **Clinical Precedence** — which indicates an approved or investigational drug — is present for only {clin_prec_cov:.1f}% of targets, highlighting the large gap between genetic/literature evidence and clinical translatability.

---

## 6. Cross-Dataset Observations

These three datasets are designed to complement each other in later analysis stages:

- **Disease to Targets**: Open Targets provides ranked gene/protein candidates for Alzheimer's disease
- **Disease to Trials**: ClinicalTrials.gov provides the intervention and phase landscape
- **Disease to Papers**: Semantic Scholar captures the research theme and citation landscape
- **Targets to Papers**: Gene symbols (e.g. APP, APOE) can be matched to paper abstracts to assess literature density per target
- **Targets to Trials**: Target gene names and associated drugs can be mapped to trial interventions to identify which targets are already in clinical testing vs. underexplored
- **Trial interventions to Research Themes**: Text mining of intervention names and outcome measures can reveal thematic clusters (amyloid, tau, neuroinflammation, etc.)

---

## 7. Limitations

- **API completeness**: The Semantic Scholar dataset is capped at 500 papers and may not represent the full publication landscape.
- **Abstract missingness**: {missing_abstract_pct}% of papers lack abstracts, reducing coverage for NLP-based theme extraction.
- **ClinicalTrials.gov scope**: Only registered trials appear in this dataset. Unregistered studies, observational research, and trials registered outside ClinicalTrials.gov are absent.
- **Open Targets truncation**: {ot_truncation_note}
- **Association scores are not therapeutic proof**: Open Targets scores reflect the breadth and strength of available evidence, not clinical validation. A high score indicates that a target is well-studied, not that it is a proven drug target.
- **This project is not clinical decision support.** All outputs are for research intelligence and hypothesis generation only.

---

## 8. Recommended Next Steps

Based on the EDA, the following analytical steps are recommended:

1. **Text mining on literature** — Apply TF-IDF to paper abstracts to extract dominant and emerging research themes (amyloid, tau, neuroinflammation, synaptic dysfunction, etc.)
2. **Intervention text mining** — Extract and categorise intervention types from clinical trial text to build an intervention taxonomy
3. **Gap scoring** — Combine target association scores, literature density per target, and trial saturation to identify underexplored candidates
4. **Target-theme-trial linking** — Map gene symbols across all three datasets to build a linked evidence network
5. **Knowledge graph construction** — Build a NetworkX graph with Disease, Target, Paper, Trial, and Theme nodes for visual and algorithmic exploration
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved -> {REPORT_FILE}")


# ════════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  NeuroGraph Agent -- Exploratory Data Analysis")
    print("=" * 60)

    df_ct, df_lit, df_ot = load_data()

    ct_stats  = eda_clinical_trials(df_ct)
    lit_stats = eda_literature(df_lit)
    ot_stats  = eda_open_targets(df_ot)

    write_report(df_ct, df_lit, df_ot, ct_stats, lit_stats, ot_stats)

    print("\n" + "=" * 60)
    print("  EDA complete.")
    print(f"  Figures  -> outputs/figures/  ({len(os.listdir(FIGURES_DIR))} files)")
    print(f"  Report   -> outputs/reports/eda_summary.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
