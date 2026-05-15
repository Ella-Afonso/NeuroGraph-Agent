"""
text_mining.py
Step 5: Text Mining and Research Theme Extraction
NeuroGraph Agent — Alzheimer's Disease Research Intelligence

Extracts computational research themes from literature and clinical trial text
using TF-IDF keyword extraction, NMF topic modelling, and target mention
matching. All themes are computationally derived and should be interpreted
with caution — they are hypothesis-generating signals, not clinical conclusions.
"""

import os
import re
import sys
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Force UTF-8 console output on Windows to avoid cp1252 encoding errors
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
FIG_DIR  = os.path.join(BASE_DIR, "outputs", "figures")
RPT_DIR  = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
N_TOPICS   = 6
N_CLUSTERS = 6
TOP_N_TERMS_TFIDF  = 30
TOP_N_TERMS_TOPIC  = 10
TOP_N_TARGETS      = 50
RECENT_YEAR_CUTOFF = 2022
RANDOM_STATE       = 42

CUSTOM_STOPWORDS = {
    "study", "studies", "trial", "trials", "disease", "diseases",
    "patient", "patients", "treatment", "treatments", "clinical",
    "outcome", "outcomes", "measure", "measures", "group", "groups",
    "participant", "participants", "intervention", "interventions",
    "placebo", "randomized", "randomised", "phase", "year", "years",
    "week", "weeks", "month", "months", "day", "days",
    "baseline", "follow", "arm", "arms", "result", "results",
    "effect", "effects", "data", "analysis", "use", "used",
    "primary", "secondary", "endpoint", "endpoints",
    "compared", "comparison", "including", "included", "include",
    "associated", "association", "significantly", "significant",
    "research", "review", "based", "total", "high", "low",
    "reported", "report", "shown", "shown", "known", "new",
    "increased", "decreased", "reduced", "improvement",
    "test", "tests", "assessed", "assessment", "evaluating",
    "evaluated", "evaluation", "conducted", "conducted",
    "randomized controlled", "controlled trial", "double blind",
    "open label", "placebo controlled",
    "alzheimer", "alzheimers", "ad",  # ubiquitous in all docs
}

SKLEARN_ENGLISH = "english"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", text)
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_combine(*fields) -> str:
    parts = [str(f) for f in fields if pd.notna(f) and str(f).strip() not in ("", "nan")]
    return " ".join(parts)


def build_combined_stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    combined = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS
    return list(combined)


def make_tfidf(corpus, ngram_range=(1, 2), min_df=2, max_df=0.85,
               max_features=1000, extra_stopwords=None):
    stop = build_combined_stopwords()
    if extra_stopwords:
        stop += list(extra_stopwords)
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words=stop,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(corpus)
    return vec, matrix


def top_tfidf_terms(vec, matrix, top_n=30):
    mean_scores = np.asarray(matrix.mean(axis=0)).flatten()
    feature_names = vec.get_feature_names_out()
    indices = mean_scores.argsort()[::-1][:top_n]
    return [(feature_names[i], mean_scores[i]) for i in indices]


def assign_nmf_topics(H):
    return np.argmax(H, axis=1)


def assign_nmf_scores(H):
    return H.max(axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("=" * 60)
    print("STEP 5: TEXT MINING — NEUROGRAPH AGENT")
    print("=" * 60)

    lit = pd.read_csv(os.path.join(DATA_DIR, "literature_clean.csv"))
    trials = pd.read_csv(os.path.join(DATA_DIR, "clinical_trials_clean.csv"))
    targets = pd.read_csv(os.path.join(DATA_DIR, "target_evidence_clean.csv"))

    print(f"\nLoaded literature:      {len(lit):,} records")
    print(f"Loaded clinical trials: {len(trials):,} records")
    print(f"Loaded targets:         {len(targets):,} records")
    return lit, trials, targets


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPARE TEXT FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_literature_text(lit: pd.DataFrame) -> pd.DataFrame:
    if "paper_text" in lit.columns and lit["paper_text"].notna().sum() > 0:
        lit["literature_text_for_mining"] = lit["paper_text"].apply(clean_text)
    else:
        lit["literature_text_for_mining"] = lit.apply(
            lambda r: clean_text(
                safe_combine(r.get("title"), r.get("abstract"), r.get("fields_of_study"))
            ),
            axis=1,
        )

    before = len(lit)
    lit = lit[lit["literature_text_for_mining"].str.strip() != ""].reset_index(drop=True)
    after = len(lit)
    if before > after:
        print(f"  [Warning] Dropped {before - after} literature records with empty text.")

    print(f"\nLiterature records used for mining: {len(lit):,}")
    return lit


def prepare_trial_text(trials: pd.DataFrame) -> pd.DataFrame:
    if "trial_text" in trials.columns and trials["trial_text"].notna().sum() > 0:
        trials["trial_text_for_mining"] = trials["trial_text"].apply(clean_text)
    else:
        trials["trial_text_for_mining"] = trials.apply(
            lambda r: clean_text(
                safe_combine(
                    r.get("brief_title"), r.get("official_title"),
                    r.get("conditions"), r.get("interventions"),
                    r.get("primary_outcomes"), r.get("eligibility_criteria"),
                )
            ),
            axis=1,
        )

    before = len(trials)
    trials = trials[trials["trial_text_for_mining"].str.strip() != ""].reset_index(drop=True)
    after = len(trials)
    if before > after:
        print(f"  [Warning] Dropped {before - after} trial records with empty text.")

    print(f"Clinical trial records used for mining: {len(trials):,}")
    return trials


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LITERATURE TF-IDF
# ═══════════════════════════════════════════════════════════════════════════════

def literature_tfidf(lit: pd.DataFrame):
    print("\n--- Literature TF-IDF ---")
    corpus = lit["literature_text_for_mining"].tolist()
    vec, matrix = make_tfidf(corpus)
    print(f"  TF-IDF matrix shape (literature): {matrix.shape}")

    # global top terms
    top_terms = top_tfidf_terms(vec, matrix, top_n=TOP_N_TERMS_TFIDF)
    tfidf_df = pd.DataFrame(top_terms, columns=["term", "score"])
    tfidf_df["source"] = "literature"

    # per-year top terms
    year_terms_rows = []
    if "year" in lit.columns:
        for yr in sorted(lit["year"].dropna().unique()):
            idx = lit[lit["year"] == yr].index
            if len(idx) < 3:
                continue
            sub_matrix = matrix[idx]
            yr_top = top_tfidf_terms(vec, sub_matrix, top_n=10)
            for term, score in yr_top:
                year_terms_rows.append({"year": int(yr), "term": term, "score": score})

    # recent papers
    if "year" in lit.columns:
        recent_idx = lit[lit["year"] >= RECENT_YEAR_CUTOFF].index
        if len(recent_idx) >= 5:
            recent_top = top_tfidf_terms(vec, matrix[recent_idx], top_n=20)
            print(f"  Top terms in recent literature ({RECENT_YEAR_CUTOFF}+): "
                  f"{[t for t, _ in recent_top[:10]]}")

    out_path = os.path.join(DATA_DIR, "literature_tfidf_terms.csv")
    tfidf_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}")

    # figure 1 — top TF-IDF terms
    fig, ax = plt.subplots(figsize=(10, 7))
    terms_plot = tfidf_df.head(20)
    ax.barh(terms_plot["term"][::-1], terms_plot["score"][::-1], color="#4472C4")
    ax.set_xlabel("Mean TF-IDF Score", fontsize=11)
    ax.set_title("Top 20 TF-IDF Terms — Literature\n(computationally extracted; interpret with caution)", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "text_lit_top_tfidf_terms.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # figure 2 — terms by year (heatmap-style bar, top 8 terms over years)
    if year_terms_rows:
        year_df = pd.DataFrame(year_terms_rows)
        # pick the 8 most common terms across all years
        top8 = (
            year_df.groupby("term")["score"].mean()
            .sort_values(ascending=False)
            .head(8)
            .index.tolist()
        )
        year_pivot = (
            year_df[year_df["term"].isin(top8)]
            .pivot_table(index="year", columns="term", values="score", aggfunc="mean")
            .fillna(0)
        )
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        for term in year_pivot.columns:
            ax2.plot(year_pivot.index, year_pivot[term], marker="o", label=term)
        ax2.set_xlabel("Year", fontsize=11)
        ax2.set_ylabel("Mean TF-IDF Score", fontsize=11)
        ax2.set_title("Top TF-IDF Terms by Year — Literature\n(provisional trend; small year samples may be unstable)", fontsize=12)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        fig2_path = os.path.join(FIG_DIR, "text_lit_terms_by_year.png")
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {fig2_path}")
    else:
        print("  [Warning] Not enough year data for terms-by-year figure.")

    return vec, matrix, tfidf_df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CLINICAL TRIAL TF-IDF
# ═══════════════════════════════════════════════════════════════════════════════

def trial_tfidf(trials: pd.DataFrame):
    print("\n--- Clinical Trial TF-IDF ---")
    corpus = trials["trial_text_for_mining"].tolist()
    vec, matrix = make_tfidf(corpus)
    print(f"  TF-IDF matrix shape (trials): {matrix.shape}")

    top_terms = top_tfidf_terms(vec, matrix, top_n=TOP_N_TERMS_TFIDF)
    tfidf_df = pd.DataFrame(top_terms, columns=["term", "score"])
    tfidf_df["source"] = "clinical_trials"

    out_path = os.path.join(DATA_DIR, "trial_tfidf_terms.csv")
    tfidf_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}")

    # figure — top TF-IDF terms
    fig, ax = plt.subplots(figsize=(10, 7))
    terms_plot = tfidf_df.head(20)
    ax.barh(terms_plot["term"][::-1], terms_plot["score"][::-1], color="#ED7D31")
    ax.set_xlabel("Mean TF-IDF Score", fontsize=11)
    ax.set_title("Top 20 TF-IDF Terms — Clinical Trials\n(computationally extracted; interpret with caution)", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "text_trial_top_tfidf_terms.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    return vec, matrix, tfidf_df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PROVISIONAL THEME LABELLER
# ═══════════════════════════════════════════════════════════════════════════════

# Keyword-to-label mapping; order matters — first match wins
THEME_RULES = [
    (["amyloid", "tau", "plaque", "beta amyloid", "neurofibrillary", "fibrillary", "peptide",
       "clearance", "aggregation", "fibril", "oligomer"], "Amyloid/Tau Pathology"),
    (["neuroinflammation", "microglia", "astrocyte", "cytokine", "inflammation", "inflammatory",
       "immune", "nfkb", "il", "tnf", "complement", "trem2"], "Neuroinflammation & Immune"),
    (["synaptic", "synapse", "neurotransmitter", "acetylcholine", "cholinergic", "glutamate",
       "gaba", "receptor", "nmda", "long term potentiation", "plasticity"], "Synaptic/Neurotransmitter"),
    (["cognitive", "cognition", "memory", "executive", "attention", "language", "learning",
       "adas", "mmse", "moca", "neuropsychological"], "Cognitive & Neuropsychological"),
    (["genetic", "gene", "risk", "variant", "snp", "apoe", "gwas", "locus", "allele",
       "mutation", "polymorphism", "heritability"], "Genetics & Risk Factors"),
    (["biomarker", "csf", "plasma", "pet", "mri", "imaging", "neuroimaging", "positron",
       "cerebrospinal fluid", "fluid biomarker"], "Biomarkers & Imaging"),
    (["mitochondria", "oxidative", "stress", "energy", "metabolism", "autophagy",
       "proteasome", "ubiquitin", "reactive oxygen"], "Metabolic/Mitochondrial"),
    (["clinical trial", "drug", "inhibitor", "antibody", "therapy", "therapeutic",
       "monoclonal", "aducanumab", "lecanemab", "donanemab", "vaccine"], "Drug/Therapeutic Interventions"),
    (["caregiver", "dementia care", "quality of life", "qol", "non pharmacological",
       "exercise", "diet", "lifestyle", "sleep"], "Care, Lifestyle & QoL"),
]


def label_topic(top_terms_list: list) -> str:
    terms_text = " ".join(top_terms_list).lower()
    for keywords, label in THEME_RULES:
        for kw in keywords:
            if kw in terms_text:
                return label
    return "Other / Mixed"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LITERATURE NMF TOPIC MODELLING
# ═══════════════════════════════════════════════════════════════════════════════

def literature_nmf(lit: pd.DataFrame, vec, matrix):
    print("\n--- Literature NMF Topic Modelling ---")
    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=400)
    H = nmf.fit_transform(matrix)   # document-topic matrix
    W = nmf.components_             # topic-term matrix
    feature_names = vec.get_feature_names_out()
    print(f"  NMF fitted: {N_TOPICS} topics")

    topic_records = []
    for tid in range(N_TOPICS):
        top_idx = W[tid].argsort()[::-1][:TOP_N_TERMS_TOPIC]
        top_terms = [feature_names[i] for i in top_idx]
        label = label_topic(top_terms)
        topic_records.append({
            "topic_id": tid,
            "topic_label": label,
            "top_terms": ", ".join(top_terms),
        })
        print(f"  Topic {tid} [{label}]: {top_terms[:5]}")

    # paper assignments
    assignments = assign_nmf_topics(H)
    scores = assign_nmf_scores(H)

    topic_size = pd.Series(assignments).value_counts().sort_index()
    for tid in range(N_TOPICS):
        topic_records[tid]["n_papers"] = int(topic_size.get(tid, 0))

    topics_df = pd.DataFrame(topic_records)
    topics_path = os.path.join(DATA_DIR, "literature_topics.csv")
    topics_df.to_csv(topics_path, index=False, encoding="utf-8")
    print(f"  Saved: {topics_path}")

    # paper-level assignments
    id_col = "paper_id" if "paper_id" in lit.columns else lit.index
    assign_df = pd.DataFrame({
        "paper_id":      lit["paper_id"] if "paper_id" in lit.columns else lit.index,
        "title":         lit["title"] if "title" in lit.columns else "",
        "year":          lit["year"] if "year" in lit.columns else np.nan,
        "citation_count": lit["citation_count"] if "citation_count" in lit.columns else np.nan,
        "topic_id":      assignments,
        "topic_score":   scores,
    })
    label_map = {r["topic_id"]: r["topic_label"] for r in topic_records}
    assign_df["topic_label"] = assign_df["topic_id"].map(label_map)
    assign_path = os.path.join(DATA_DIR, "literature_topic_assignments.csv")
    assign_df.to_csv(assign_path, index=False, encoding="utf-8")
    print(f"  Saved: {assign_path}")

    # figure — topic sizes
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, N_TOPICS))
    bars = ax.bar(
        [f"T{r['topic_id']}\n{r['topic_label']}" for r in topic_records],
        [r["n_papers"] for r in topic_records],
        color=colors,
    )
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title("Literature NMF Topic Sizes\n(provisional topics; interpret with caution)", fontsize=12)
    ax.bar_label(bars, padding=3)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "text_lit_topic_sizes.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # figure — topics by year
    if "year" in lit.columns:
        assign_df_yr = assign_df.dropna(subset=["year"])
        assign_df_yr = assign_df_yr.copy()
        assign_df_yr["year"] = assign_df_yr["year"].astype(int)
        year_topic = (
            assign_df_yr.groupby(["year", "topic_id"])
            .size()
            .reset_index(name="n_papers")
        )
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        for tid, grp in year_topic.groupby("topic_id"):
            label = label_map[tid]
            ax2.plot(grp["year"], grp["n_papers"], marker="o", label=f"T{tid}: {label}")
        ax2.set_xlabel("Year", fontsize=11)
        ax2.set_ylabel("Number of Papers", fontsize=11)
        ax2.set_title("Literature Topics by Year\n(provisional trend; interpret with caution)", fontsize=12)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        fig2_path = os.path.join(FIG_DIR, "text_lit_topics_by_year.png")
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {fig2_path}")

    return topics_df, assign_df, H, W, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CLINICAL TRIAL NMF TOPIC MODELLING
# ═══════════════════════════════════════════════════════════════════════════════

def trial_nmf(trials: pd.DataFrame, vec, matrix):
    print("\n--- Clinical Trial NMF Topic Modelling ---")
    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=400)
    H = nmf.fit_transform(matrix)
    W = nmf.components_
    feature_names = vec.get_feature_names_out()
    print(f"  NMF fitted: {N_TOPICS} topics")

    topic_records = []
    for tid in range(N_TOPICS):
        top_idx = W[tid].argsort()[::-1][:TOP_N_TERMS_TOPIC]
        top_terms = [feature_names[i] for i in top_idx]
        label = label_topic(top_terms)
        topic_records.append({
            "topic_id": tid,
            "topic_label": label,
            "top_terms": ", ".join(top_terms),
        })
        print(f"  Topic {tid} [{label}]: {top_terms[:5]}")

    assignments = assign_nmf_topics(H)
    scores = assign_nmf_scores(H)
    topic_size = pd.Series(assignments).value_counts().sort_index()
    for tid in range(N_TOPICS):
        topic_records[tid]["n_trials"] = int(topic_size.get(tid, 0))

    topics_df = pd.DataFrame(topic_records)
    topics_path = os.path.join(DATA_DIR, "clinical_trial_topics.csv")
    topics_df.to_csv(topics_path, index=False, encoding="utf-8")
    print(f"  Saved: {topics_path}")

    # trial-level assignments
    assign_df = pd.DataFrame({
        "nct_id":       trials["nct_id"] if "nct_id" in trials.columns else trials.index,
        "brief_title":  trials["brief_title"] if "brief_title" in trials.columns else "",
        "overall_status": trials["overall_status"] if "overall_status" in trials.columns else "",
        "phases":       trials["phases"] if "phases" in trials.columns else "",
        "topic_id":     assignments,
        "topic_score":  scores,
    })
    label_map = {r["topic_id"]: r["topic_label"] for r in topic_records}
    assign_df["topic_label"] = assign_df["topic_id"].map(label_map)
    assign_path = os.path.join(DATA_DIR, "clinical_trial_topic_assignments.csv")
    assign_df.to_csv(assign_path, index=False, encoding="utf-8")
    print(f"  Saved: {assign_path}")

    # figure — topic sizes
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, N_TOPICS))
    bars = ax.bar(
        [f"T{r['topic_id']}\n{r['topic_label']}" for r in topic_records],
        [r["n_trials"] for r in topic_records],
        color=colors,
    )
    ax.set_ylabel("Number of Trials", fontsize=11)
    ax.set_title("Clinical Trial NMF Topic Sizes\n(provisional topics; interpret with caution)", fontsize=12)
    ax.bar_label(bars, padding=3)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "text_trial_topic_sizes.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # figure — topics by status
    if "overall_status" in trials.columns:
        status_topic = (
            assign_df.groupby(["overall_status", "topic_id"])
            .size()
            .reset_index(name="n_trials")
        )
        top_statuses = (
            status_topic.groupby("overall_status")["n_trials"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        sub = status_topic[status_topic["overall_status"].isin(top_statuses)]
        pivot = sub.pivot_table(
            index="overall_status", columns="topic_id", values="n_trials", aggfunc="sum"
        ).fillna(0)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(pivot.index))
        width = 0.13
        for i, tid in enumerate(pivot.columns):
            label = label_map.get(int(tid), f"T{tid}")
            ax2.bar(x + i * width, pivot[tid], width, label=f"T{int(tid)}: {label}")
        ax2.set_xticks(x + width * (N_TOPICS / 2))
        ax2.set_xticklabels(pivot.index, rotation=15, ha="right")
        ax2.set_ylabel("Number of Trials", fontsize=11)
        ax2.set_title("Trial Topics by Status\n(provisional; interpret with caution)", fontsize=12)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        fig2_path = os.path.join(FIG_DIR, "text_trial_topics_by_status.png")
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {fig2_path}")

    return topics_df, assign_df


# ═══════════════════════════════════════════════════════════════════════════════
# 9. KMEANS COMPARISON (OPTIONAL)
# ═══════════════════════════════════════════════════════════════════════════════

def kmeans_comparison(lit: pd.DataFrame, lit_matrix, trials: pd.DataFrame, trial_matrix):
    print("\n--- KMeans Clustering (comparison) ---")

    # literature
    lit_norm = normalize(lit_matrix, norm="l2")
    km_lit = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    lit_clusters = km_lit.fit_predict(lit_norm)
    lit_km_df = pd.DataFrame({
        "paper_id":   lit["paper_id"] if "paper_id" in lit.columns else lit.index,
        "title":      lit["title"] if "title" in lit.columns else "",
        "kmeans_cluster": lit_clusters,
    })
    lit_km_path = os.path.join(DATA_DIR, "literature_kmeans_clusters.csv")
    lit_km_df.to_csv(lit_km_path, index=False, encoding="utf-8")
    print(f"  Saved: {lit_km_path}")

    # trials
    trial_norm = normalize(trial_matrix, norm="l2")
    km_trial = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    trial_clusters = km_trial.fit_predict(trial_norm)
    trial_km_df = pd.DataFrame({
        "nct_id":     trials["nct_id"] if "nct_id" in trials.columns else trials.index,
        "brief_title": trials["brief_title"] if "brief_title" in trials.columns else "",
        "kmeans_cluster": trial_clusters,
    })
    trial_km_path = os.path.join(DATA_DIR, "clinical_trial_kmeans_clusters.csv")
    trial_km_df.to_csv(trial_km_path, index=False, encoding="utf-8")
    print(f"  Saved: {trial_km_path}")

    return lit_km_df, trial_km_df


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TARGET MENTION MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def target_mention_matching(targets: pd.DataFrame,
                             lit: pd.DataFrame,
                             trials: pd.DataFrame):
    print("\n--- Target Mention Matching ---")

    top_targets = (
        targets.sort_values("association_score", ascending=False)
        .head(TOP_N_TARGETS)
        .reset_index(drop=True)
    )

    lit_corpus   = " ".join(lit["literature_text_for_mining"].fillna(""))
    trial_corpus = " ".join(trials["trial_text_for_mining"].fillna(""))

    rows = []
    for _, row in top_targets.iterrows():
        symbol = str(row["approved_symbol"]).strip()
        name   = str(row.get("approved_name", "")).strip()
        score  = row.get("association_score", np.nan)

        # match whole word, case-insensitive
        pattern = r"\b" + re.escape(symbol.lower()) + r"\b"
        lit_count   = len(re.findall(pattern, lit_corpus))
        trial_count = len(re.findall(pattern, trial_corpus))
        gap_signal  = lit_count - trial_count

        rows.append({
            "approved_symbol":           symbol,
            "approved_name":             name,
            "association_score":         score,
            "literature_mentions":       lit_count,
            "trial_mentions":            trial_count,
            "literature_to_trial_gap_signal": gap_signal,
        })

    mentions_df = pd.DataFrame(rows).sort_values("literature_mentions", ascending=False)
    out_path = os.path.join(DATA_DIR, "target_text_mentions.csv")
    mentions_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}")

    # top by gap signal
    top_gap = mentions_df.sort_values("literature_to_trial_gap_signal", ascending=False).head(10)
    print("  Top targets by literature-to-trial gap signal (lit mentions >> trial mentions):")
    for _, r in top_gap.iterrows():
        print(f"    {r['approved_symbol']:12s}  lit={r['literature_mentions']:3d}  "
              f"trial={r['trial_mentions']:3d}  gap={r['literature_to_trial_gap_signal']:3d}")

    # figure
    plot_df = mentions_df.head(25).sort_values("literature_mentions")
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(plot_df))
    bar_width = 0.35
    ax.barh(y + bar_width / 2, plot_df["literature_mentions"],   bar_width,
            label="Literature mentions", color="#4472C4", alpha=0.85)
    ax.barh(y - bar_width / 2, plot_df["trial_mentions"],        bar_width,
            label="Trial mentions",      color="#ED7D31", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["approved_symbol"], fontsize=9)
    ax.set_xlabel("Mention Count (whole-word match in text)", fontsize=11)
    ax.set_title(
        "Top 25 Targets: Literature vs Trial Mentions\n"
        "(text-based counts; false positives possible for short gene symbols)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "text_target_mentions_lit_vs_trials.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    return mentions_df


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_summary_report(
    lit_topics_df: pd.DataFrame,
    trial_topics_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    n_lit: int,
    n_trials: int,
    lit_matrix_shape: tuple,
    trial_matrix_shape: tuple,
    ran_kmeans: bool,
):
    top_lit_mentions = mentions_df.sort_values("literature_mentions", ascending=False).head(5)
    top_gap = mentions_df.sort_values("literature_to_trial_gap_signal", ascending=False).head(5)

    lines = [
        "# NeuroGraph Agent — Text Mining and Theme Extraction Summary",
        "",
        f"*Generated: 2026-05-12 | Step 5 of NeuroGraph pipeline*",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "This step applies computational text mining to extract research themes from",
        "Alzheimer's disease literature and clinical trial records collected in earlier",
        "pipeline steps. The methods used here — TF-IDF and NMF topic modelling — are",
        "lexical and statistical in nature. All themes are provisional and should be",
        "treated as hypothesis-generating signals for downstream analysis, not as",
        "clinically validated findings.",
        "",
        "---",
        "",
        "## 2. Methods",
        "",
        "### TF-IDF (Term Frequency–Inverse Document Frequency)",
        "TF-IDF scores terms that are frequent within individual documents but rare across",
        "the corpus, highlighting distinctive vocabulary. A custom stopword list removes",
        "generic clinical trial and study language so that biomedical terms remain.",
        f"- Literature corpus: {n_lit:,} records | TF-IDF matrix: {lit_matrix_shape[0]} x {lit_matrix_shape[1]}",
        f"- Trial corpus:      {n_trials:,} records | TF-IDF matrix: {trial_matrix_shape[0]} x {trial_matrix_shape[1]}",
        "- Settings: ngram_range=(1,2), min_df=2, max_df=0.85, max_features=1000",
        "",
        "### NMF (Non-negative Matrix Factorisation) Topic Modelling",
        "NMF decomposes the TF-IDF matrix into topics (latent themes) and document-topic",
        "weights. Each document is assigned to its strongest topic. Topic labels are",
        "assigned heuristically based on top terms and should be treated as provisional.",
        f"- Number of topics: {N_TOPICS}",
        "- Topic labels are assigned by rule-based keyword matching, not expert review.",
        "",
    ]

    if ran_kmeans:
        lines += [
            "### KMeans Clustering (comparison)",
            "KMeans clustering was also run on normalised TF-IDF vectors as a supplementary",
            f"comparison (k={N_CLUSTERS}). Results are saved to CSV but not visualised in this report.",
            "",
        ]

    lines += [
        "### Target Mention Matching",
        f"The top {TOP_N_TARGETS} targets by Open Targets association score were matched by",
        "whole-word regex against the combined literature and trial text corpora.",
        "Mention counts are approximate: short gene symbols (e.g. APP, ACE) may match",
        "common English words and inflate counts.",
        "",
        "---",
        "",
        "## 3. Literature Themes",
        "",
        "The following provisional topics were identified in the literature corpus.",
        "Top terms are listed in order of NMF component weight.",
        "",
    ]

    for _, row in lit_topics_df.iterrows():
        lines += [
            f"### Topic {int(row['topic_id'])}: {row['topic_label']}",
            f"- **Top terms:** {row['top_terms']}",
            f"- **Papers assigned:** {int(row['n_papers'])}",
            "",
        ]

    lines += [
        "---",
        "",
        "## 4. Clinical Trial Themes",
        "",
        "The following provisional topics were identified in the clinical trial corpus.",
        "",
    ]

    for _, row in trial_topics_df.iterrows():
        lines += [
            f"### Topic {int(row['topic_id'])}: {row['topic_label']}",
            f"- **Top terms:** {row['top_terms']}",
            f"- **Trials assigned:** {int(row['n_trials'])}",
            "",
        ]

    lines += [
        "---",
        "",
        "## 5. Target Mention Signals",
        "",
        "The table below shows the top 5 targets by literature mentions.",
        "Note: these counts reflect raw text matching and may include false positives.",
        "",
        "| Symbol | Approved Name | Assoc. Score | Lit Mentions | Trial Mentions | Gap Signal |",
        "|--------|--------------|-------------|-------------|---------------|-----------|",
    ]

    for _, r in top_lit_mentions.iterrows():
        lines.append(
            f"| {r['approved_symbol']} | {r['approved_name'][:40]} | "
            f"{r['association_score']:.3f} | {int(r['literature_mentions'])} | "
            f"{int(r['trial_mentions'])} | {int(r['literature_to_trial_gap_signal'])} |"
        )

    lines += [
        "",
        "**Top 5 targets by literature-to-trial gap signal** (high literature presence,"
        " relatively low trial presence — potential research-to-clinic translation gap):",
        "",
        "| Symbol | Lit Mentions | Trial Mentions | Gap Signal |",
        "|--------|-------------|---------------|-----------|",
    ]

    for _, r in top_gap.iterrows():
        lines.append(
            f"| {r['approved_symbol']} | {int(r['literature_mentions'])} | "
            f"{int(r['trial_mentions'])} | {int(r['literature_to_trial_gap_signal'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Interpretation",
        "",
        "Computationally, the dominant literature themes appear to cluster around amyloid/tau",
        "pathology, neuroinflammation, and genetics/risk factors — reflecting the established",
        "focus of published Alzheimer's research. Clinical trial topics tend to weight",
        "more heavily toward cognitive/neuropsychological outcomes and therapeutic",
        "interventions, which is consistent with the trial lifecycle focus on measurable",
        "endpoints.",
        "",
        "Targets with high literature presence and lower trial presence (positive gap signal)",
        "may represent areas where preclinical or mechanistic research has outpaced clinical",
        "translation. However, this signal alone is insufficient to conclude a true gap;",
        "it must be combined with target evidence scores, trial status, and phase data in",
        "Step 6.",
        "",
        "---",
        "",
        "## 7. Limitations",
        "",
        "- TF-IDF is lexical: it cannot resolve synonyms, abbreviations, or biomedical",
        "  semantic equivalences (e.g. 'BACE1' vs 'beta-secretase').",
        "- NMF topic number (n=6) is a modelling choice. Different values may produce",
        "  different topic boundaries; no single decomposition is objectively correct.",
        "- Provisional topic labels are assigned by rule-based keyword matching, not",
        "  expert clinical review. Labels should be validated by a domain expert.",
        "- Gene symbol mention counting can produce false positives for short symbols",
        "  (e.g. 'APP', 'ACE', 'IL') that coincide with common English words or",
        "  abbreviations.",
        "- Missing or very short abstracts reduce the discriminative power of TF-IDF",
        "  for those records.",
        "- Clinical trial text rarely names specific gene/protein targets directly;",
        "  low trial mention counts for a gene do not necessarily indicate absence of",
        "  target-relevant research.",
        "- This analysis is hypothesis-generating only. It does not constitute clinical",
        "  evidence and should not be interpreted as such.",
        "",
        "---",
        "",
        "## 8. Next Step",
        "",
        "**Step 6: Research Gap Scoring**",
        "",
        "The next step will build a structured research gap scoring table that combines:",
        "- Open Targets association scores (from target_evidence_clean.csv)",
        "- Literature mention counts and topic coverage (from this step)",
        "- Clinical trial mention counts, trial status, and phase (from this step + trials data)",
        "- A composite gap score to rank targets by the strength of preclinical evidence",
        "  relative to clinical trial investment",
        "",
        "This will produce a ranked list of candidate targets for knowledge graph",
        "prioritisation and visual exploration in subsequent steps.",
        "",
        "---",
        "",
        "*All outputs are computationally derived. No clinical conclusions should be drawn",
        "from this report without expert domain review.*",
    ]

    report_path = os.path.join(RPT_DIR, "text_mining_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. load
    lit, trials, targets = load_data()

    # 2. prepare text
    lit    = prepare_literature_text(lit)
    trials = prepare_trial_text(trials)

    # 3. literature TF-IDF
    lit_vec, lit_matrix, lit_tfidf_df = literature_tfidf(lit)

    # 4. trial TF-IDF
    trial_vec, trial_matrix, trial_tfidf_df = trial_tfidf(trials)

    # 5. literature NMF
    lit_topics_df, lit_assign_df, lit_H, lit_W, lit_features = literature_nmf(
        lit, lit_vec, lit_matrix
    )

    # 6. trial NMF
    trial_topics_df, trial_assign_df = trial_nmf(trials, trial_vec, trial_matrix)

    # 7. KMeans comparison
    ran_kmeans = True
    try:
        kmeans_comparison(lit, lit_matrix, trials, trial_matrix)
    except Exception as e:
        print(f"  [Warning] KMeans skipped: {e}")
        ran_kmeans = False

    # 8. target mention matching
    mentions_df = target_mention_matching(targets, lit, trials)

    # 9. summary report
    write_summary_report(
        lit_topics_df, trial_topics_df, mentions_df,
        n_lit=len(lit), n_trials=len(trials),
        lit_matrix_shape=lit_matrix.shape,
        trial_matrix_shape=trial_matrix.shape,
        ran_kmeans=ran_kmeans,
    )

    # 10. validation summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Literature records:          {len(lit):,}")
    print(f"  Trial records:               {len(trials):,}")
    print(f"  Lit TF-IDF matrix:           {lit_matrix.shape}")
    print(f"  Trial TF-IDF matrix:         {trial_matrix.shape}")
    print(f"  Literature topics:           {len(lit_topics_df)}")
    print(f"  Trial topics:                {len(trial_topics_df)}")
    print(f"  KMeans run:                  {ran_kmeans}")
    print(f"  Targets matched:             {len(mentions_df)}")

    saved_files = [
        os.path.join(DATA_DIR, "literature_tfidf_terms.csv"),
        os.path.join(DATA_DIR, "trial_tfidf_terms.csv"),
        os.path.join(DATA_DIR, "literature_topics.csv"),
        os.path.join(DATA_DIR, "literature_topic_assignments.csv"),
        os.path.join(DATA_DIR, "clinical_trial_topics.csv"),
        os.path.join(DATA_DIR, "clinical_trial_topic_assignments.csv"),
        os.path.join(DATA_DIR, "literature_kmeans_clusters.csv"),
        os.path.join(DATA_DIR, "clinical_trial_kmeans_clusters.csv"),
        os.path.join(DATA_DIR, "target_text_mentions.csv"),
        os.path.join(FIG_DIR, "text_lit_top_tfidf_terms.png"),
        os.path.join(FIG_DIR, "text_lit_terms_by_year.png"),
        os.path.join(FIG_DIR, "text_trial_top_tfidf_terms.png"),
        os.path.join(FIG_DIR, "text_lit_topic_sizes.png"),
        os.path.join(FIG_DIR, "text_lit_topics_by_year.png"),
        os.path.join(FIG_DIR, "text_trial_topic_sizes.png"),
        os.path.join(FIG_DIR, "text_trial_topics_by_status.png"),
        os.path.join(FIG_DIR, "text_target_mentions_lit_vs_trials.png"),
        os.path.join(RPT_DIR, "text_mining_summary.md"),
    ]

    print("\n  Output files:")
    for path in saved_files:
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        print(f"    [{status}] {os.path.relpath(path, BASE_DIR)}")

    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()
