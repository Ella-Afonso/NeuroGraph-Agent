"""
components.py — Reusable Streamlit UI components for NeuroGraph dashboard.
"""
import pandas as pd
import streamlit as st

_NO_MATCH_MSG = (
    "No direct symbol matches found in this corpus sample. "
    "This does not mean the target is absent from the wider research landscape."
)

# ── Display-value mappings ────────────────────────────────────────────────────
# Maps raw internal values → user-friendly display labels.
# Do NOT change these keys — they match the values stored in CSVs.

TIER_DISPLAY = {
    "Strong text-supported gap signal":             "Potential research gap candidate",
    "Established / trial-covered target":           "Established or trial-covered target",
    "Checked-zero text signal":                     "No direct text match in this corpus",
    "Ambiguous symbol; manual validation required": "Needs manual validation",
    "Low-priority / weak gap signal":               "Lower-priority signal",
}

BIOTYPE_DISPLAY = {
    "protein_coding":        "Protein-coding gene",
    "lncRNA":                "Long non-coding RNA",
    "miRNA":                 "MicroRNA",
    "pseudogene":            "Pseudogene",
    "processed_pseudogene":  "Processed pseudogene",
    "rRNA":                  "Ribosomal RNA",
    "snRNA":                 "Small nuclear RNA",
    "snoRNA":                "Small nucleolar RNA",
}

COVERAGE_DISPLAY = {
    "checked_with_mentions": "Direct text match found",
    "checked_zero_mentions": "No direct text match found",
    "not_checked":           "Not checked",
}

# Column rename map used for display tables across pages
DISPLAY_RENAME = {
    "approved_symbol":                "Symbol",
    "approved_name":                  "Target Name",
    "biotype":                        "Target Type",
    "association_score":              "Association Score",
    "gap_score":                      "Gap Score",
    "conservative_gap_score":         "Conservative Gap Score",
    "gap_rank":                       "Rank",
    "literature_mentions":            "Literature Mentions",
    "trial_mentions":                 "Trial Mentions",
    "literature_documents_mentioned": "Literature Documents",
    "trial_documents_mentioned":      "Trial Documents",
    "gap_interpretation_tier":        "Interpretation",
    "symbol_ambiguity_flag":          "Symbol Ambiguity",
    "text_coverage_status":           "Text Matching Status",
    "evidence_strength_label":        "Evidence Strength",
    "literature_signal_label":        "Literature Signal",
    "trial_signal_label":             "Trial Signal",
    "dominant_literature_theme":      "Top Literature Theme",
    "dominant_trial_theme":           "Top Trial Theme",
    "n_matched_papers":               "Matched Papers",
    "n_matched_trials":               "Matched Trials",
}

# Tier → CSS colour (keyed on raw values, used for inline HTML)
TIER_COLOURS_CSS = {
    "Strong text-supported gap signal":             "#d62728",
    "Established / trial-covered target":           "#1f77b4",
    "Checked-zero text signal":                     "#8bafd1",
    "Ambiguous symbol; manual validation required": "#ff7f0e",
    "Low-priority / weak gap signal":               "#9a9a22",
}


# ── Mapping helpers ───────────────────────────────────────────────────────────

def map_tier(raw: str) -> str:
    """Return user-friendly display label for a raw gap_interpretation_tier value."""
    return TIER_DISPLAY.get(str(raw), str(raw))


def map_biotype(raw: str) -> str:
    """Return user-friendly display label for a raw biotype value."""
    return BIOTYPE_DISPLAY.get(str(raw), str(raw))


def map_coverage(raw: str) -> str:
    """Return user-friendly display label for a raw text_coverage_status value."""
    return COVERAGE_DISPLAY.get(str(raw), str(raw))


def apply_display_maps(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all value mappings and rename columns for display. Returns a copy."""
    df = df.copy()
    if "gap_interpretation_tier" in df.columns:
        df["gap_interpretation_tier"] = df["gap_interpretation_tier"].apply(map_tier)
    if "biotype" in df.columns:
        df["biotype"] = df["biotype"].apply(map_biotype)
    if "text_coverage_status" in df.columns:
        df["text_coverage_status"] = df["text_coverage_status"].apply(map_coverage)
    return df.rename(columns={k: v for k, v in DISPLAY_RENAME.items() if k in df.columns})


# ── Shared UI blocks ──────────────────────────────────────────────────────────

def disclaimer_banner():
    st.info(
        "**NeuroGraph Agent** is a research intelligence prototype. "
        "Outputs are computationally derived and hypothesis-generating. "
        "**This is not clinical decision support.** "
        "All findings require expert validation before informing any decision."
    )


def glossary_expander() -> None:
    """Render a collapsible glossary of dashboard terms."""
    with st.expander("Glossary: What do these terms mean?"):
        st.markdown("""
**Potential research gap candidate**
: A target with literature signal and comparatively low direct trial-text signal
in this prototype. This is not proof of a true research gap — it is a hypothesis-generating signal.

**Established or trial-covered target**
: A target with clear trial-text presence in this corpus. These targets serve as useful
comparators rather than primary gap candidates.

**No direct text match in this corpus**
: The target symbol was not detected in the retrieved paper or trial text.
This does not mean the target is absent from wider research — it only means
this prototype did not detect the symbol in the current corpus sample.

**Protein-coding gene**
: A gene that contains instructions for making a protein. Most therapeutic targets are protein-coding genes.

**Open Targets association score**
: A score summarising multi-source evidence linking a target to a disease.
It is not proof that the target is therapeutically valid or a viable drug target.

**Literature mentions**
: How many times the target gene symbol was detected via whole-word matching
in the retrieved paper corpus (~478 papers).

**Trial mentions**
: How many times the target gene symbol was detected in retrieved clinical trial text
(~1,000 registrations). Low trial mentions do not mean no relevant trials exist —
trial records often describe interventions by drug name, not by molecular target symbol.

**Symbol ambiguity**
: A warning that the gene symbol is short or common and may appear in non-biological
text contexts. Mention counts for ambiguous symbols should be validated manually.
        """)


def signal_labels_expander() -> None:
    """Render a collapsible explanation of signal/label columns."""
    with st.expander("How to read these labels"):
        st.markdown("""
**Evidence Strength**
: Based on the Open Targets association score. It reflects multi-source evidence
linking the target to Alzheimer's disease — not proof of therapeutic validity.
Higher scores indicate more research attention from curated databases, not clinical readiness.

**Literature Signal**
: Based on how often the target gene symbol was detected in the retrieved paper corpus
via whole-word matching. Strong signal means more occurrences in this corpus sample.

**Trial Signal**
: Based on how often the target gene symbol was detected in retrieved clinical trial text.
Low signal does **not** mean no relevant trials exist — trial registrations typically
describe interventions by drug or mechanism name rather than molecular target symbol.

**Matched Papers / Matched Trials**
: The count of paper or trial records directly linked to this target inside the
knowledge graph. This is a sparse subset of the full corpus — a value of 0 does not
mean no evidence exists in the broader literature.
        """)


# ── Evidence card components ──────────────────────────────────────────────────

def target_evidence_card(row: pd.Series, is_strong: bool):
    """Three-column evidence card for a single target."""
    col1, col2, col3 = st.columns(3)

    tier     = str(row.get("gap_interpretation_tier", "—"))
    biotype  = str(row.get("biotype", "—"))
    tier_lbl = map_tier(tier)
    bio_lbl  = map_biotype(biotype)

    with col1:
        st.markdown(f"### {row['approved_symbol']}")
        st.markdown(f"**Target Name:** {row.get('approved_name', '—')}")
        st.markdown(f"**Target Type:** {bio_lbl}")
        st.markdown(f"**Target ID:** `{row.get('target_id', '—')}`")
        st.markdown(f"**Rank:** {int(row.get('gap_rank', 0))}")

    with col2:
        st.metric("Association Score", f"{float(row.get('association_score', 0)):.4f}",
                  help="Open Targets disease-target association score. Not proof of therapeutic validity.")
        st.metric("Gap Score", f"{float(row.get('gap_score', 0)):.4f}",
                  help="Composite gap score: higher = stronger evidence vs lower trial text coverage.")
        st.metric("Conservative Gap Score", f"{float(row.get('conservative_gap_score', 0)):.4f}",
                  help="Gap score with symbol ambiguity penalty applied.")

    with col3:
        st.metric("Literature Mentions", int(row.get("literature_mentions", 0)),
                  help="Whole-word symbol matches in the literature corpus sample (~478 papers).")
        st.metric("Trial Mentions", int(row.get("trial_mentions", 0)),
                  help="Whole-word symbol matches in the trial corpus sample (~1,000 registrations).")
        colour = TIER_COLOURS_CSS.get(tier, "#555")
        st.markdown(
            f"**Interpretation:** <span style='color:{colour};font-weight:bold'>{tier_lbl}</span>",
            unsafe_allow_html=True,
        )
        cov_raw = str(row.get("text_coverage_status", "—"))
        st.markdown(f"**Text Matching Status:** {map_coverage(cov_raw)}")
        ambig = str(row.get("symbol_ambiguity_flag", "—"))
        st.markdown(f"**Symbol Ambiguity:** `{ambig}`")

    if is_strong:
        st.success(
            "This target is a **potential research gap candidate**: "
            "it has literature signal but comparatively low trial-text coverage in this corpus. "
            "This is hypothesis-generating only — it does not confirm a true research gap."
        )

    note = row.get("interpretation_note", "")
    if pd.notna(note) and str(note).strip() not in ("", "nan"):
        with st.expander("Interpretation note"):
            st.write(str(note))


def papers_table(df: pd.DataFrame | None):
    """Render matching papers table, or a cautious no-match message."""
    if df is None or len(df) == 0:
        st.caption(_NO_MATCH_MSG)
        return
    display_cols = [c for c in
                    ["title", "year", "citation_count", "venue", "topic_label"]
                    if c in df.columns]
    rename = {"title": "Title", "year": "Year", "citation_count": "Citations",
              "venue": "Venue", "topic_label": "Topic"}
    disp = df[display_cols].rename(columns=rename).reset_index(drop=True)
    st.dataframe(disp, use_container_width=True, hide_index=True)


def trials_table(df: pd.DataFrame | None):
    """Render matching trials table, or a cautious no-match message."""
    if df is None or len(df) == 0:
        st.caption(_NO_MATCH_MSG)
        return
    display_cols = [c for c in
                    ["nct_id", "brief_title", "overall_status", "phases", "topic_label"]
                    if c in df.columns]
    rename = {"nct_id": "NCT ID", "brief_title": "Brief Title", "overall_status": "Status",
              "phases": "Phase", "topic_label": "Topic"}
    disp = df[display_cols].rename(columns=rename).reset_index(drop=True)
    st.dataframe(disp, use_container_width=True, hide_index=True)


def section_header(text: str, caption: str = ""):
    st.subheader(text)
    if caption:
        st.caption(caption)
