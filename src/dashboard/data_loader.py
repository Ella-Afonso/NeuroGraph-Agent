"""
data_loader.py — Cached data loading for NeuroGraph dashboard.

All public functions return a DataFrame or None if the source file is missing.
Use @st.cache_data so data is loaded once per session.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT    = Path(__file__).resolve().parents[2]
DATA    = ROOT / "Data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
GRAPHS  = ROOT / "outputs" / "graphs"
FIGURES = ROOT / "outputs" / "figures"


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── Core gap / target files ────────────────────────────────────────────────────

@st.cache_data
def load_gap_scores() -> pd.DataFrame | None:
    df = _read_csv(DATA / "research_gap_scores.csv")
    if df is not None and "symbol_ambiguity_flag" in df.columns:
        df["symbol_ambiguity_flag"] = df["symbol_ambiguity_flag"].fillna("unknown")
    return df


@st.cache_data
def load_strong_candidates() -> pd.DataFrame | None:
    return _read_csv(DATA / "strong_text_supported_gap_candidates.csv")


@st.cache_data
def load_top_candidates() -> pd.DataFrame | None:
    return _read_csv(DATA / "top_research_gap_candidates.csv")


@st.cache_data
def load_target_mentions() -> pd.DataFrame | None:
    return _read_csv(DATA / "target_text_mentions_full.csv")


# ── Knowledge graph ────────────────────────────────────────────────────────────

@st.cache_data
def load_kg_nodes() -> pd.DataFrame | None:
    return _read_csv(DATA / "knowledge_graph_nodes.csv")


@st.cache_data
def load_kg_edges() -> pd.DataFrame | None:
    return _read_csv(DATA / "knowledge_graph_edges.csv")


# ── Theme / topic files ────────────────────────────────────────────────────────

@st.cache_data
def load_lit_topics() -> pd.DataFrame | None:
    return _read_csv(DATA / "literature_topics.csv")


@st.cache_data
def load_trial_topics() -> pd.DataFrame | None:
    return _read_csv(DATA / "clinical_trial_topics.csv")


@st.cache_data
def load_lit_assignments() -> pd.DataFrame | None:
    return _read_csv(DATA / "literature_topic_assignments.csv")


@st.cache_data
def load_trial_assignments() -> pd.DataFrame | None:
    return _read_csv(DATA / "clinical_trial_topic_assignments.csv")


# ── Raw corpora (large; cached to avoid repeated I/O) ─────────────────────────

@st.cache_data
def load_literature_clean() -> pd.DataFrame | None:
    return _read_csv(DATA / "literature_clean.csv")


@st.cache_data
def load_trials_clean() -> pd.DataFrame | None:
    return _read_csv(DATA / "clinical_trials_clean.csv")


# ── Reports ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_report(filename: str) -> str | None:
    path = REPORTS / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


# ── Path helpers (not cached — just return Path objects) ──────────────────────

def figures_dir() -> Path:
    return FIGURES


def graphs_dir() -> Path:
    return GRAPHS


def project_root() -> Path:
    return ROOT
