"""
app.py — NeuroGraph Agent: Streamlit Research Intelligence Dashboard

Alzheimer's disease target evidence mapping and research gap analysis.
Hypothesis-generating prototype. Not clinical decision support.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.data_loader import (
    figures_dir,
    graphs_dir,
    load_gap_scores,
    load_kg_edges,
    load_kg_nodes,
    load_lit_assignments,
    load_lit_topics,
    load_literature_clean,
    load_report,
    load_strong_candidates,
    load_trial_assignments,
    load_trial_topics,
    load_trials_clean,
    project_root,
)
from src.dashboard.charts import (
    fig_kg_edge_types,
    fig_kg_node_types,
    fig_lit_vs_trial_scatter,
    fig_topic_sizes_bar,
    fig_top_gap_bar,
)
from src.dashboard.components import (
    apply_display_maps,
    disclaimer_banner,
    glossary_expander,
    map_biotype,
    map_coverage,
    map_tier,
    papers_table,
    section_header,
    signal_labels_expander,
    target_evidence_card,
    TIER_DISPLAY,
    trials_table,
)
from src.dashboard.evidence_cards import (
    load_evidence_cards,
    render_card_summary,
)

FIGURES = figures_dir()
GRAPHS  = graphs_dir()
_ROOT   = project_root()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroGraph Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: navigation ────────────────────────────────────────────────────────
PAGES = [
    "Overview",
    "Gap Candidates",
    "Target Explorer",
    "Evidence Cards",
    "Themes",
    "Knowledge Graph",
    "Reports & Downloads",
    "Methodology & Limitations",
]

with st.sidebar:
    st.markdown("## NeuroGraph Agent")
    st.markdown("*Alzheimer's Research Intelligence*")
    st.divider()
    page = st.radio("Navigate to", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption(
        "Research intelligence prototype. "
        "Outputs are computationally derived and hypothesis-generating. "
        "Not clinical decision support."
    )


# ── Helper ─────────────────────────────────────────────────────────────────────

def _show_figure(path: Path, caption: str = "", use_container_width: bool = True):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=use_container_width)
    else:
        st.info(f"Figure not available: {path.name}")


def _download_btn(label: str, rel_path: str, filename: str, mime: str):
    p = _ROOT / rel_path
    if p.exists():
        st.download_button(label=label, data=p.read_bytes(),
                           file_name=filename, mime=mime)
    else:
        st.caption(f"{filename} — file not found")


# ══════════════════════════════════════════════════════════════════════════════
# Page 1: Overview
# ══════════════════════════════════════════════════════════════════════════════

def page_overview():
    st.title("NeuroGraph Agent")
    st.markdown(
        "A computational research intelligence prototype mapping Alzheimer's disease targets, "
        "literature, clinical trials, and research themes to identify evidence coverage gaps. "
        "All outputs are hypothesis-generating and require expert validation."
    )
    disclaimer_banner()

    gap    = load_gap_scores()
    strong = load_strong_candidates()
    nodes  = load_kg_nodes()
    edges  = load_kg_edges()
    lit    = load_literature_clean()
    trials = load_trials_clean()

    # ── Key metrics ────────────────────────────────────────────────────────────
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Targets Scored",
              len(gap)    if gap    is not None else "—",
              help="Total gene targets scored in this pipeline run.")
    c2.metric("Gap Candidates",
              len(strong) if strong is not None else "—",
              help="Targets with literature signal but comparatively low trial-text mentions.")
    c3.metric("Papers Analysed",
              len(lit)    if lit    is not None else "—",
              help="Literature papers retrieved from Semantic Scholar.")
    c4.metric("Trials Analysed",
              len(trials) if trials is not None else "—",
              help="Clinical trial registrations retrieved from ClinicalTrials.gov.")
    c5.metric("Graph Nodes",
              len(nodes)  if nodes  is not None else "—",
              help="Total nodes in the NeuroGraph knowledge graph.")
    c6.metric("Graph Edges",
              len(edges)  if edges  is not None else "—",
              help="Total edges (associations) in the knowledge graph.")

    st.divider()

    # ── Pipeline ───────────────────────────────────────────────────────────────
    st.subheader("Analysis Pipeline")
    st.markdown(
        "`Data Collection` → `Cleaning & EDA` → `Text Mining (TF-IDF / NMF)` "
        "→ `Target Mention Matching` → `Gap Scoring` → `Knowledge Graph` → **Dashboard**"
    )
    st.caption(
        "Sources: Open Targets (disease-target evidence), "
        "Semantic Scholar (literature corpus), ClinicalTrials.gov (trial registrations)."
    )

    st.divider()

    # ── Top candidates + score distribution ───────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        section_header(
            "Top Gap Candidates",
            "Targets with literature signal but comparatively low trial-text mentions "
            "in this corpus — ranked by gap score.",
        )
        st.caption(
            "**Potential research gap candidates** are targets that appear in the literature "
            "corpus but have comparatively low direct trial-text mentions in this prototype. "
            "This does not prove that no trials exist for these targets."
        )
        if strong is not None and len(strong) > 0:
            ov_cols = [c for c in
                       ["approved_symbol", "approved_name", "gap_score",
                        "literature_mentions", "trial_mentions", "gap_interpretation_tier"]
                       if c in strong.columns]
            ov_disp = apply_display_maps(strong[ov_cols].head(5))
            st.dataframe(ov_disp.reset_index(drop=True),
                         use_container_width=True, hide_index=True)
        else:
            st.warning("strong_text_supported_gap_candidates.csv not found.")

    with col_r:
        section_header("Gap Score Distribution")
        _show_figure(FIGURES / "gap_02_score_distribution.png")

    st.divider()

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        section_header("Interpretation Breakdown")
        _show_figure(FIGURES / "gap_06_interpretation_tier_breakdown.png")
    with col_r2:
        section_header("Knowledge Graph Node Types")
        _show_figure(FIGURES / "kg_01_node_type_counts.png")

    st.divider()
    glossary_expander()


# ══════════════════════════════════════════════════════════════════════════════
# Page 2: Gap Candidates
# ══════════════════════════════════════════════════════════════════════════════

def page_gap_candidates():
    st.title("Gap Candidates")
    st.markdown(
        "Research targets scored for evidence-to-trial coverage gaps. "
        "Higher gap score indicates stronger evidence/literature signal with comparatively "
        "lower representation in trial text within this corpus."
    )

    gap = load_gap_scores()
    if gap is None:
        st.error("research_gap_scores.csv not found.")
        return

    # ── Interpretation tier filter uses display labels; raw values used for filtering
    all_raw_tiers   = sorted(gap["gap_interpretation_tier"].dropna().unique().tolist())
    tier_disp_map   = {raw: map_tier(raw) for raw in all_raw_tiers}
    tier_rev_map    = {v: k for k, v in tier_disp_map.items()}
    all_disp_tiers  = [tier_disp_map[raw] for raw in all_raw_tiers]

    # ── Filters ────────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_disp_tiers = st.multiselect(
                "Interpretation",
                all_disp_tiers, default=all_disp_tiers,
                help="Filter by how each target is classified in this prototype.",
            )
            sel_tiers = [tier_rev_map[d] for d in sel_disp_tiers]
        with f2:
            all_ambig = sorted(gap["symbol_ambiguity_flag"].dropna().unique().tolist())
            sel_ambig = st.multiselect(
                "Symbol Ambiguity",
                all_ambig, default=all_ambig,
                help="'low' = symbol is reasonably specific; 'moderate' = may need manual validation.",
            )
        with f3:
            min_gs    = st.slider("Min Gap Score",
                                  0.0, float(gap["gap_score"].max()), 0.0, step=0.01)
            min_lit   = st.slider("Min Literature Mentions",
                                  0, int(gap["literature_mentions"].max()), 0)
            max_trial = st.slider("Max Trial Mentions",
                                  0, int(gap["trial_mentions"].max()),
                                  int(gap["trial_mentions"].max()))

    filtered = gap[
        gap["gap_interpretation_tier"].isin(sel_tiers)
        & gap["symbol_ambiguity_flag"].isin(sel_ambig)
        & (gap["gap_score"] >= min_gs)
        & (gap["literature_mentions"] >= min_lit)
        & (gap["trial_mentions"] <= max_trial)
    ].reset_index(drop=True)

    st.markdown(f"**{len(filtered)} targets** match filters (of {len(gap)} total).")

    with st.expander("What does gap score mean?"):
        st.markdown(
            "Gap score is a composite of five signals: evidence strength, literature signal, "
            "trial signal (inverted), underexploration, and literature-to-trial coverage gap. "
            "**A high gap score does not prove therapeutic relevance or the absence of trials** — "
            "trial registrations often name drugs or mechanisms rather than molecular targets. "
            "Use gap score as a hypothesis-generating signal for further expert review."
        )

    st.info(
        "**Interpretation** explains how to read the target's signal in this prototype. "
        "It is not a clinical conclusion. A target labelled **Potential research gap candidate** "
        "has literature signal but comparatively low direct trial-text mentions. "
        "Targets labelled **Established or trial-covered** have stronger trial-text presence "
        "and are better used as comparator references."
    )

    # ── Charts ─────────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        section_header("Top 20 by Gap Score")
        if len(filtered) > 0:
            fig = fig_top_gap_bar(filtered, n=20)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No targets match the current filters.")
    with col_r:
        section_header("Literature vs Trial Mentions")
        if len(filtered) > 0:
            fig2 = fig_lit_vs_trial_scatter(filtered)
            st.pyplot(fig2)
            plt.close(fig2)

    st.divider()

    # ── Full table ─────────────────────────────────────────────────────────────
    section_header(
        "Filtered Target Table",
        "Click any column header to sort. All values are computationally derived.",
    )

    st.caption(
        "**Target Type** describes the biological category of the target. "
        "'Protein-coding gene' means the gene contains instructions for making a protein — "
        "most therapeutic targets are protein-coding genes.  \n"
        "**Text Matching Status** shows whether the target symbol was detected in the "
        "retrieved corpus. 'No direct text match found' does not mean the target has no research."
    )

    table_raw_cols = [
        "gap_rank", "approved_symbol", "approved_name", "biotype",
        "association_score", "gap_score", "conservative_gap_score",
        "literature_mentions", "trial_mentions",
        "gap_interpretation_tier", "symbol_ambiguity_flag", "text_coverage_status",
    ]
    disp_cols = [c for c in table_raw_cols if c in filtered.columns]
    tbl_disp  = apply_display_maps(filtered[disp_cols])

    st.dataframe(
        tbl_disp,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Target Type": st.column_config.TextColumn(
                "Target Type",
                help="Biological category of the target. 'Protein-coding gene' means the gene makes a protein.",
            ),
            "Interpretation": st.column_config.TextColumn(
                "Interpretation",
                help="How to read this target's signal. Not a clinical conclusion.",
            ),
            "Text Matching Status": st.column_config.TextColumn(
                "Text Matching Status",
                help="Whether the gene symbol was detected in the retrieved corpus. "
                     "'No direct text match found' does not mean no research exists.",
            ),
            "Symbol Ambiguity": st.column_config.TextColumn(
                "Symbol Ambiguity",
                help="'low' = symbol reasonably specific; 'moderate' = may appear in non-biological contexts.",
            ),
            "Association Score": st.column_config.NumberColumn(
                "Association Score",
                help="Open Targets disease-target association score. Not proof of therapeutic validity.",
                format="%.4f",
            ),
            "Gap Score": st.column_config.NumberColumn(
                "Gap Score",
                help="Composite score: higher = stronger evidence vs lower trial-text coverage.",
                format="%.4f",
            ),
        },
    )

    st.divider()
    signal_labels_expander()
    glossary_expander()


# ══════════════════════════════════════════════════════════════════════════════
# Page 3: Target Explorer
# ══════════════════════════════════════════════════════════════════════════════

def page_target_explorer():
    st.title("Target Explorer")
    st.markdown(
        "Search for a specific target to view its evidence card, "
        "literature mentions, and clinical trial appearances in this corpus."
    )

    gap = load_gap_scores()
    if gap is None:
        st.error("research_gap_scores.csv not found.")
        return

    strong        = load_strong_candidates()
    strong_syms   = set(strong["approved_symbol"].tolist()) if strong is not None else set()
    kg_edges      = load_kg_edges()
    lit_assign    = load_lit_assignments()
    trial_assign  = load_trial_assignments()
    lit_clean     = load_literature_clean()

    symbols = sorted(gap["approved_symbol"].dropna().unique().tolist())
    default = symbols.index("TREM2") if "TREM2" in symbols else 0

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Select Target**")
        selected = st.selectbox("Target Symbol", symbols, index=default,
                                label_visibility="collapsed")

    target_row = gap[gap["approved_symbol"] == selected].iloc[0]
    is_strong  = selected in strong_syms

    # ── Tabs: Evidence Card | Matching Records | Research Brief ───────────────
    tab_card, tab_papers, tab_trials, tab_brief = st.tabs(
        ["Evidence Card", "Matching Papers", "Matching Trials", "Research Brief"]
    )

    with tab_card:
        section_header(f"Evidence Card: {selected}")
        target_evidence_card(target_row, is_strong)

        # Download generated markdown card if available
        ev_cards = load_evidence_cards()
        if ev_cards is not None:
            card_row = ev_cards[ev_cards["approved_symbol"] == selected]
            if len(card_row) > 0:
                render_card_summary(card_row.iloc[0])

    with tab_papers:
        section_header(
            "Matching Papers (corpus sample)",
            "Papers in which the target symbol was found via whole-word matching.",
        )
        if kg_edges is not None and lit_assign is not None:
            p_edges = kg_edges[
                (kg_edges["edge_type"] == "target_mentioned_in_paper")
                & (kg_edges["target_symbol"] == selected)
            ]
            if len(p_edges) > 0:
                pids = p_edges["paper_id"].astype(str).tolist()
                matched = lit_assign[lit_assign["paper_id"].astype(str).isin(pids)].copy()
                if lit_clean is not None and "venue" in lit_clean.columns:
                    venue_map = (
                        lit_clean[["paper_id", "venue"]]
                        .astype({"paper_id": str})
                        .drop_duplicates("paper_id")
                    )
                    matched = matched.merge(venue_map, on="paper_id", how="left")
                papers_table(matched)
            else:
                papers_table(None)
        else:
            papers_table(None)

    with tab_trials:
        section_header(
            "Matching Trials (corpus sample)",
            "Clinical trial registrations in which the target symbol was found via whole-word matching.",
        )
        if kg_edges is not None and trial_assign is not None:
            t_edges = kg_edges[
                (kg_edges["edge_type"] == "target_mentioned_in_trial")
                & (kg_edges["target_symbol"] == selected)
            ]
            if len(t_edges) > 0:
                tids = t_edges["nct_id"].astype(str).tolist()
                matched_t = trial_assign[trial_assign["nct_id"].astype(str).isin(tids)].copy()
                trials_table(matched_t)
            else:
                trials_table(None)
        else:
            trials_table(None)

    with tab_brief:
        section_header(
            "Research Brief",
            "Full generated evidence brief for this target (Markdown format).",
        )
        ev_cards = load_evidence_cards()
        if ev_cards is not None:
            card_row = ev_cards[ev_cards["approved_symbol"] == selected]
            if len(card_row) > 0:
                card_rel = str(card_row.iloc[0].get("evidence_card_path", ""))
                if card_rel and card_rel != "nan":
                    card_path = _ROOT / card_rel
                    if card_path.exists():
                        st.markdown(card_path.read_text(encoding="utf-8"))
                    else:
                        st.info("Evidence card file not found. Run generate_evidence_cards.py.")
                else:
                    st.info("No evidence card path recorded.")
            else:
                st.info("No evidence card found for this target.")
        else:
            st.info(
                "Evidence cards not yet generated. "
                "Run: `python -m src.processing.generate_evidence_cards`"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Page 4: Evidence Cards
# ══════════════════════════════════════════════════════════════════════════════

def page_evidence_cards():
    st.title("Evidence Cards")
    st.markdown(
        "Structured, cautious research briefs generated from pipeline outputs for all 499 scored targets. "
        "All content is computationally derived — hypothesis-generating only. "
        "Not clinical decision support."
    )
    disclaimer_banner()

    ev_cards = load_evidence_cards()

    if ev_cards is None:
        st.error(
            "Evidence card table not found (`Data/processed/target_evidence_cards.csv`). "
            "Run: `python -m src.processing.generate_evidence_cards`"
        )
        return

    st.caption(f"{len(ev_cards)} evidence cards available.")

    signal_labels_expander()

    # ── Filters — use display labels for tier, raw values for filtering ────────
    all_raw_ec_tiers  = sorted(ev_cards["gap_interpretation_tier"].dropna().unique().tolist())
    ec_tier_disp_map  = {raw: map_tier(raw) for raw in all_raw_ec_tiers}
    ec_tier_rev_map   = {v: k for k, v in ec_tier_disp_map.items()}
    all_ec_disp_tiers = [ec_tier_disp_map[raw] for raw in all_raw_ec_tiers]

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Evidence Card Filters**")
        sel_ec_disp_tier = st.selectbox(
            "Interpretation",
            ["All interpretations"] + all_ec_disp_tiers,
            key="ec_tier",
        )
        ev_options = ["All evidence levels"] + sorted(
            ev_cards["evidence_strength_label"].dropna().unique().tolist()
        )
        sel_ev = st.selectbox("Evidence Strength", ev_options, key="ec_ev")

    filtered = ev_cards.copy()
    if sel_ec_disp_tier != "All interpretations":
        raw_tier_filter = ec_tier_rev_map.get(sel_ec_disp_tier, sel_ec_disp_tier)
        filtered = filtered[filtered["gap_interpretation_tier"] == raw_tier_filter]
    if sel_ev != "All evidence levels":
        filtered = filtered[filtered["evidence_strength_label"] == sel_ev]

    filtered = filtered.sort_values("gap_rank", na_position="last").reset_index(drop=True)
    st.caption(f"Showing {len(filtered)} targets after filters.")

    # ── Target selector ───────────────────────────────────────────────────────
    if len(filtered) == 0:
        st.warning("No targets match the current filters.")
        return

    labels  = filtered["approved_symbol"].tolist()
    default = labels.index("TREM2") if "TREM2" in labels else 0
    sel_sym = st.selectbox("Select Target", labels, index=default, key="ec_sel")
    sel_row = filtered[filtered["approved_symbol"] == sel_sym].iloc[0]

    st.divider()

    # ── Card summary ──────────────────────────────────────────────────────────
    render_card_summary(sel_row)

    # ── Filtered overview table ───────────────────────────────────────────────
    st.divider()
    section_header(
        "All Filtered Targets",
        "Scroll or filter. Click any column header to sort.",
    )

    st.info(
        "**Interpretation** summarises how the target should be read in this prototype. "
        "These labels are generated from evidence strength, literature mentions, trial-text mentions, "
        "and symbol ambiguity. They are hypothesis-generating only.  \n"
        "**Matched Papers** and **Matched Trials** count evidence records linked inside the "
        "knowledge graph — a sparse subset. A value of 0 does **not** mean no literature or "
        "real-world trials exist."
    )

    table_cols = [
        "gap_rank", "approved_symbol", "approved_name",
        "gap_interpretation_tier",
        "evidence_strength_label",
        "literature_signal_label", "trial_signal_label",
        "literature_mentions", "trial_mentions",
        "literature_documents_mentioned", "trial_documents_mentioned",
        "n_matched_papers", "n_matched_trials",
        "dominant_literature_theme", "dominant_trial_theme",
    ]
    show_cols = [c for c in table_cols if c in filtered.columns]
    ec_tbl_disp = apply_display_maps(filtered[show_cols])
    # apply_display_maps renames gap_interpretation_tier → "Interpretation"
    # and uses DISPLAY_RENAME; rename remaining columns not in DISPLAY_RENAME
    extra_rename = {
        "evidence_strength_label": "Evidence Strength",
        "literature_signal_label": "Literature Signal",
        "trial_signal_label":      "Trial Signal",
        "dominant_literature_theme": "Top Literature Theme",
        "dominant_trial_theme":      "Top Trial Theme",
        "Rank": "Rank",
    }
    ec_tbl_disp = ec_tbl_disp.rename(
        columns={k: v for k, v in extra_rename.items() if k in ec_tbl_disp.columns}
    )

    st.dataframe(
        ec_tbl_disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Interpretation": st.column_config.TextColumn(
                "Interpretation",
                help="How to read this target's signal. Not a clinical conclusion.",
            ),
            "Evidence Strength": st.column_config.TextColumn(
                "Evidence Strength",
                help="Derived from Open Targets association score.",
            ),
            "Literature Signal": st.column_config.TextColumn(
                "Literature Signal",
                help="Derived from whole-word symbol mention count in the literature corpus.",
            ),
            "Trial Signal": st.column_config.TextColumn(
                "Trial Signal",
                help="Derived from whole-word symbol mention count in trial text. Low signal does not mean no trials exist.",
            ),
            "Matched Papers": st.column_config.NumberColumn(
                "Matched Papers",
                help="Papers linked in the knowledge graph. Sparse subset — 0 does not mean no literature exists.",
            ),
            "Matched Trials": st.column_config.NumberColumn(
                "Matched Trials",
                help="Trials linked in the knowledge graph. Sparse subset — 0 does not mean no trials exist.",
            ),
        },
    )

    st.divider()
    glossary_expander()

    # ── Signal profile figure ─────────────────────────────────────────────────
    st.divider()
    section_header(
        "Signal Profile: Literature vs Trial Mentions",
        "Strong gap candidates (red labels) vs established comparators (grey labels). Corpus sample only.",
    )
    fig_path = _ROOT / "outputs" / "figures" / "cards_01_signal_profile_examples.png"
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info("Signal profile figure not found. Run generate_evidence_cards.py to create it.")

    # ── Brief pack download ────────────────────────────────────────────────────
    st.divider()
    _download_btn(
        "Download Top Gap Candidate Brief Pack (Markdown)",
        "outputs/reports/top_gap_candidate_briefs.md",
        "top_gap_candidate_briefs.md",
        "text/markdown",
    )
    _download_btn(
        "Download Evidence Card Table (CSV)",
        "Data/processed/target_evidence_cards.csv",
        "target_evidence_cards.csv",
        "text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Page 5: Themes
# ══════════════════════════════════════════════════════════════════════════════

def page_themes():
    st.title("Research Themes")
    st.markdown(
        "NMF topic models applied to TF-IDF representations of literature abstracts "
        "and clinical trial text. Topic labels are approximate and computational — "
        "they should be interpreted by domain experts."
    )

    lit_topics   = load_lit_topics()
    trial_topics = load_trial_topics()
    lit_assign   = load_lit_assignments()
    trial_assign = load_trial_assignments()

    tab1, tab2 = st.tabs(["Literature Themes", "Trial Themes"])

    # ── Literature ─────────────────────────────────────────────────────────────
    with tab1:
        if lit_topics is None:
            st.warning("literature_topics.csv not found.")
        else:
            col_l, col_r = st.columns(2)
            with col_l:
                section_header("Topic Sizes (papers per theme)")
                fig = fig_topic_sizes_bar(lit_topics, "n_papers",
                                          "Literature: Papers per NMF Topic")
                st.pyplot(fig)
                plt.close(fig)
            with col_r:
                section_header("Topic Details")
                st.dataframe(lit_topics, use_container_width=True)

            st.divider()
            section_header("Browse Papers by Literature Theme")
            topic_opts = lit_topics["topic_label"].tolist()
            if topic_opts:
                sel = st.selectbox("Select Literature Theme", topic_opts,
                                   key="lit_theme_sel")
                tid = int(lit_topics.loc[
                    lit_topics["topic_label"] == sel, "topic_id"
                ].iloc[0])
                top_terms = str(lit_topics.loc[
                    lit_topics["topic_id"] == tid, "top_terms"
                ].iloc[0])
                st.caption(f"Top NMF terms: {top_terms}")
                if lit_assign is not None:
                    papers = (
                        lit_assign[lit_assign["topic_id"] == tid]
                        .sort_values("topic_score", ascending=False)
                    )
                    disp = [c for c in
                            ["title", "year", "citation_count", "topic_score"]
                            if c in papers.columns]
                    st.dataframe(papers[disp].reset_index(drop=True),
                                 use_container_width=True, height=360)
            _show_figure(
                FIGURES / "text_lit_topic_sizes.png",
                caption="Literature topic size distribution (Step 5)",
            )

    # ── Trials ─────────────────────────────────────────────────────────────────
    with tab2:
        if trial_topics is None:
            st.warning("clinical_trial_topics.csv not found.")
        else:
            col_l, col_r = st.columns(2)
            with col_l:
                section_header("Topic Sizes (trials per theme)")
                fig = fig_topic_sizes_bar(trial_topics, "n_trials",
                                          "Trials: Trials per NMF Topic")
                st.pyplot(fig)
                plt.close(fig)
            with col_r:
                section_header("Topic Details")
                st.dataframe(trial_topics, use_container_width=True)

            st.divider()
            section_header("Browse Trials by Trial Theme")
            topic_opts = trial_topics["topic_label"].tolist()
            if topic_opts:
                sel = st.selectbox("Select Trial Theme", topic_opts,
                                   key="trial_theme_sel")
                tid = int(trial_topics.loc[
                    trial_topics["topic_label"] == sel, "topic_id"
                ].iloc[0])
                top_terms = str(trial_topics.loc[
                    trial_topics["topic_id"] == tid, "top_terms"
                ].iloc[0])
                st.caption(f"Top NMF terms: {top_terms}")
                if trial_assign is not None:
                    trials = (
                        trial_assign[trial_assign["topic_id"] == tid]
                        .sort_values("topic_score", ascending=False)
                    )
                    disp = [c for c in
                            ["nct_id", "brief_title", "overall_status",
                             "phases", "topic_score"]
                            if c in trials.columns]
                    st.dataframe(trials[disp].reset_index(drop=True),
                                 use_container_width=True, height=360)
            _show_figure(
                FIGURES / "text_trial_topics_by_status.png",
                caption="Trial themes by overall status (Step 5)",
            )


# ══════════════════════════════════════════════════════════════════════════════
# Page 5: Knowledge Graph
# ══════════════════════════════════════════════════════════════════════════════

def page_knowledge_graph():
    st.title("Knowledge Graph")
    st.markdown(
        "A directed NetworkX graph connecting Alzheimer's disease, targets, "
        "literature papers, clinical trials, and NMF research themes. "
        "Edges represent computational associations — not causal or therapeutic relationships."
    )

    nodes = load_kg_nodes()
    edges = load_kg_edges()
    gap   = load_gap_scores()

    # ── Graph metrics ──────────────────────────────────────────────────────────
    if nodes is not None and edges is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Nodes", len(nodes))
        c2.metric("Total Edges", len(edges))
        c3.metric("Node Types",  nodes["node_type"].nunique())
        c4.metric("Edge Types",  edges["edge_type"].nunique())

        col_l, col_r = st.columns(2)
        with col_l:
            section_header("Node Type Distribution")
            fig = fig_kg_node_types(nodes)
            st.pyplot(fig)
            plt.close(fig)
        with col_r:
            section_header("Edge Type Distribution")
            fig = fig_kg_edge_types(edges)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("Knowledge graph node/edge tables not found.")

    st.divider()

    # ── Static figures ─────────────────────────────────────────────────────────
    section_header(
        "Gap Candidate Subgraph",
        "Disease node + top 10 strong gap candidates by gap score. "
        "Node size = gap score.  Node colour = gap_interpretation_tier.  "
        "Outer nodes = papers/trials with direct symbol matches.",
    )
    _show_figure(FIGURES / "kg_03_target_gap_subgraph.png")

    st.divider()
    section_header(
        "Theme Subgraph",
        "Literature (NMF) and Trial (NMF) theme nodes with overlap edges.",
    )
    _show_figure(FIGURES / "kg_04_theme_subgraph.png")

    st.divider()

    # ── Interactive subgraph explorer ──────────────────────────────────────────
    section_header(
        "Target Subgraph Explorer",
        "Select a target to inspect its graph edges and connected nodes.",
    )
    if nodes is None or edges is None:
        st.warning("Knowledge graph node/edge tables not found.")
        return
    if gap is None:
        st.warning("Gap scores not available.")
        return

    # ── Build target list from nodes table (authoritative source) ─────────────
    # node_id format: "target_TREM2"; label: "TREM2" (gene symbol)
    target_nodes_df = (
        nodes[nodes["node_type"] == "Target"]
        .drop_duplicates("node_id")
        .sort_values("gap_score", ascending=False)
        .reset_index(drop=True)
    )

    # Strong-candidate filter checkbox
    only_strong = st.checkbox(
        "Show only strong text-supported gap candidates",
        value=False,
        key="kg_strong_filter",
    )
    if only_strong:
        target_nodes_df = target_nodes_df[
            target_nodes_df["gap_interpretation_tier"]
            == "Strong text-supported gap signal"
        ].reset_index(drop=True)
        if len(target_nodes_df) == 0:
            st.info("No strong gap candidates found in the graph.")
            return

    # Display label = gene symbol (label column); internal value = node_id
    labels      = target_nodes_df["label"].tolist()       # ["TREM2", "APOE", ...]
    nid_list    = target_nodes_df["node_id"].tolist()     # ["target_TREM2", ...]
    label_to_nid = dict(zip(labels, nid_list))

    default_lbl = "TREM2" if "TREM2" in labels else labels[0]
    default_idx = labels.index(default_lbl)

    sel_label   = st.selectbox(
        "Select Target", labels, index=default_idx, key="kg_exp_sel",
    )
    sel_node_id = label_to_nid[sel_label]   # e.g. "target_TREM2"
    sel_sym     = sel_label                  # gene symbol for display

    # ── Selected target summary card ──────────────────────────────────────────
    gap_rows = gap[gap["approved_symbol"] == sel_sym]
    if len(gap_rows) > 0:
        gr   = gap_rows.iloc[0]
        tier = str(gr.get("gap_interpretation_tier", ""))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Association Score",
                  f"{float(gr.get('association_score', 0)):.4f}",
                  help="Open Targets disease-target score (not therapeutic proof).")
        c2.metric("Gap Score",
                  f"{float(gr.get('gap_score', 0)):.4f}",
                  help="Composite gap signal: higher = stronger evidence vs lower trial coverage.")
        c3.metric("Literature Mentions",
                  int(gr.get("literature_mentions", 0)),
                  help="Whole-word symbol matches in literature corpus sample.")
        c4.metric("Trial Mentions",
                  int(gr.get("trial_mentions", 0)),
                  help="Whole-word symbol matches in trial corpus sample.")
        st.caption(
            f"**Name:** {gr.get('approved_name', '—')}  |  "
            f"**Tier:** {tier}  |  "
            f"**Ambiguity flag:** `{gr.get('symbol_ambiguity_flag', '—')}`  |  "
            f"**Graph node ID:** `{sel_node_id}`"
        )
        if tier == "Strong text-supported gap signal":
            st.success(
                "Strong text-supported gap candidate: high literature signal with "
                "low trial text coverage in this corpus. Hypothesis-generating only."
            )

    st.divider()

    # ── Retrieve edges for selected target ─────────────────────────────────────
    t_edges = edges[
        (edges["source"] == sel_node_id) | (edges["target"] == sel_node_id)
    ].copy().reset_index(drop=True)

    st.markdown(
        f"**{len(t_edges)} graph edge(s)** for **{sel_sym}** (`{sel_node_id}`):"
    )

    if len(t_edges) == 0:
        st.info(
            f"No edges found for `{sel_sym}`. "
            "The target node exists in the graph but has no recorded connections "
            "in the current subset."
        )
        return

    # ── Compute connected_node_id for each edge ───────────────────────────────
    # An edge involves sel_node_id as either source or target;
    # the connected node is whichever end is NOT sel_node_id.
    t_edges["connected_node_id"] = t_edges.apply(
        lambda r: r["target"] if r["source"] == sel_node_id else r["source"],
        axis=1,
    )

    # ── Join with nodes table to get human-readable labels and types ──────────
    # Select only the columns we need to avoid NaN-heavy cross-type pollution.
    node_join = (
        nodes[[
            "node_id", "node_type", "label",
            "gap_score", "gap_interpretation_tier",
            "year", "citation_count",
            "overall_status", "phases",
            "topic_label",
        ]]
        .rename(columns={
            "node_id":   "connected_node_id",
            "node_type": "connected_node_type",
            "label":     "connected_node_label",
        })
    )

    merged = t_edges.merge(node_join, on="connected_node_id", how="left")

    # ── Friendly empty-state when only disease edge ───────────────────────────
    non_disease = merged[merged["connected_node_type"] != "Disease"]
    if len(non_disease) == 0:
        st.info(
            f"**{sel_sym}** is connected to Alzheimer's disease through an "
            "Open Targets association edge. No direct paper or trial "
            "symbol-match edges were found for this target in the current "
            "graph subset. This does not mean the target is absent from the "
            "wider research landscape."
        )
        # Still show the disease edge row for completeness

    # ── Explanation ───────────────────────────────────────────────────────────
    st.caption(
        "Different edge types have different fields. "
        "Disease-target evidence edges contain Open Targets association scores. "
        "Paper/trial mention edges contain publication or trial metadata. "
        "**N/A** means that field does not apply to that edge type."
    )

    # ── Split edges by type ────────────────────────────────────────────────────
    dis_rows   = merged[merged["edge_type"] == "disease_target_evidence"]
    paper_rows = merged[merged["edge_type"] == "target_mentioned_in_paper"]
    trial_rows = merged[merged["edge_type"] == "target_mentioned_in_trial"]
    theme_rows = merged[
        ~merged["edge_type"].isin([
            "disease_target_evidence",
            "target_mentioned_in_paper",
            "target_mentioned_in_trial",
        ])
    ]

    def _render_edge_section(
        sub: pd.DataFrame,
        col_specs: list,
    ) -> None:
        """
        Render a typed edge sub-table.
        col_specs: list of (source_col, display_name, is_int) tuples.
        Skips columns missing from sub or entirely NaN.
        Replaces remaining NaN with 'N/A'.
        """
        present = [
            (sc, dn, ii) for sc, dn, ii in col_specs
            if sc in sub.columns and sub[sc].notna().any()
        ]
        if not present:
            st.caption("No displayable columns for this section.")
            return
        result = {}
        for src_col, disp_name, is_int in present:
            result[disp_name] = sub[src_col].apply(
                lambda v, _i=is_int:
                    "N/A" if pd.isna(v) else
                    (str(int(float(v))) if _i else str(v))
            )
        st.dataframe(
            pd.DataFrame(result).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    # ── Disease Evidence Edges ─────────────────────────────────────────────────
    st.markdown("#### Disease Evidence Edges")
    if len(dis_rows) > 0:
        st.caption(
            "Open Targets association between Alzheimer's disease and this target. "
            "The association score reflects multi-source research evidence — it is not therapeutic proof."
        )
        _render_edge_section(dis_rows, [
            ("connected_node_label", "Connected Node",  False),
            ("connected_node_type",  "Type",            False),
            ("edge_type",            "Edge Type",        False),
            ("association_score",    "Assoc. Score",    False),
            ("evidence_source",      "Evidence Source", False),
        ])
    else:
        st.caption("No disease-target evidence edges found for this target.")

    # ── Paper Evidence Edges ───────────────────────────────────────────────────
    st.markdown("#### Paper Evidence Edges")
    if len(paper_rows) > 0:
        st.caption(
            "Literature papers in which the target symbol was found via whole-word matching. "
            "Year, Citations, and Topic are sourced from each paper's metadata."
        )
        _render_edge_section(paper_rows, [
            ("connected_node_label", "Connected Node", False),
            ("connected_node_type",  "Type",           False),
            ("edge_type",            "Edge Type",       False),
            ("year",                 "Year",            True),
            ("citation_count",       "Citations",       True),
            ("topic_label",          "Topic",           False),
            ("connected_node_id",    "Node ID",         False),
        ])
    else:
        st.caption(
            "No paper mention edges found for this target in the current graph subset. "
            "The target may still have literature mentions — check the Evidence Cards page."
        )

    # ── Trial Evidence Edges ───────────────────────────────────────────────────
    st.markdown("#### Trial Evidence Edges")
    if len(trial_rows) > 0:
        st.caption(
            "Clinical trial registrations in which the target symbol was found via whole-word matching. "
            "Trial Status, Phase, and Topic are sourced from each trial's metadata."
        )
        _render_edge_section(trial_rows, [
            ("connected_node_label", "Connected Node", False),
            ("connected_node_type",  "Type",           False),
            ("edge_type",            "Edge Type",       False),
            ("overall_status",       "Trial Status",    False),
            ("phases",               "Phase",           False),
            ("topic_label",          "Topic",           False),
            ("nct_id",               "NCT ID",          False),
        ])
    else:
        st.caption(
            "No trial mention edges found for this target in the current graph subset. "
            "Trial registrations often describe interventions by drug name rather than "
            "molecular target symbol — the raw trial mention count (Evidence Cards page) "
            "may still be non-zero."
        )

    # ── Theme Edges (shown only when present) ─────────────────────────────────
    if len(theme_rows) > 0:
        st.markdown("#### Theme Edges")
        st.caption("Edges connecting to research theme nodes.")
        _render_edge_section(theme_rows, [
            ("connected_node_label", "Connected Node", False),
            ("connected_node_type",  "Type",           False),
            ("edge_type",            "Edge Type",       False),
            ("topic_label",          "Topic",           False),
            ("connected_node_id",    "Node ID",         False),
        ])

    # ── Raw combined table (collapsed by default) ─────────────────────────────
    with st.expander("Show raw combined edge table"):
        st.caption(
            "All edges combined. **N/A** means a field does not apply to that edge type — "
            "this is expected because different edge types carry different attributes."
        )
        raw_col_specs = [
            ("connected_node_label", "Connected Node",  False),
            ("connected_node_type",  "Type",            False),
            ("edge_type",            "Edge Type",        False),
            ("association_score",    "Assoc. Score",    False),
            ("evidence_source",      "Evidence Source", False),
            ("year",                 "Year",            True),
            ("citation_count",       "Citations",       True),
            ("overall_status",       "Trial Status",    False),
            ("phases",               "Phase",           False),
            ("topic_label",          "Topic",           False),
            ("nct_id",               "NCT ID",          False),
            ("connected_node_id",    "Node ID",         False),
        ]
        raw_present = [
            (sc, dn, ii) for sc, dn, ii in raw_col_specs
            if sc in merged.columns and merged[sc].notna().any()
        ]
        raw_result = {}
        for src_col, disp_name, is_int in raw_present:
            raw_result[disp_name] = merged[src_col].apply(
                lambda v, _i=is_int:
                    "N/A" if pd.isna(v) else
                    (str(int(float(v))) if _i else str(v))
            )
        st.dataframe(
            pd.DataFrame(raw_result).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    # ── Connection type summary ───────────────────────────────────────────────
    type_counts = merged["connected_node_type"].value_counts()
    parts = [f"{cnt} {nt.lower()}(s)" for nt, cnt in type_counts.items()]
    st.caption(f"Connected to: {', '.join(parts)}.")


# ══════════════════════════════════════════════════════════════════════════════
# Page 6: Reports & Downloads
# ══════════════════════════════════════════════════════════════════════════════

def page_reports_downloads():
    st.title("Reports & Downloads")
    st.markdown(
        "Download processed data tables, knowledge graph files, and analysis reports "
        "generated by the NeuroGraph pipeline."
    )

    st.subheader("Data Tables")
    c1, c2, c3 = st.columns(3)
    with c1:
        _download_btn("research_gap_scores.csv",
                      "Data/processed/research_gap_scores.csv",
                      "research_gap_scores.csv", "text/csv")
        _download_btn("strong_gap_candidates.csv",
                      "Data/processed/strong_text_supported_gap_candidates.csv",
                      "strong_text_supported_gap_candidates.csv", "text/csv")
        _download_btn("top_gap_candidates.csv",
                      "Data/processed/top_research_gap_candidates.csv",
                      "top_research_gap_candidates.csv", "text/csv")
    with c2:
        _download_btn("knowledge_graph_nodes.csv",
                      "Data/processed/knowledge_graph_nodes.csv",
                      "knowledge_graph_nodes.csv", "text/csv")
        _download_btn("knowledge_graph_edges.csv",
                      "Data/processed/knowledge_graph_edges.csv",
                      "knowledge_graph_edges.csv", "text/csv")
        _download_btn("target_evidence_cards.csv",
                      "Data/processed/target_evidence_cards.csv",
                      "target_evidence_cards.csv", "text/csv")
    with c3:
        _download_btn("neurograph_knowledge_graph.json",
                      "outputs/graphs/neurograph_knowledge_graph.json",
                      "neurograph_knowledge_graph.json", "application/json")
        _download_btn("neurograph_knowledge_graph.graphml",
                      "outputs/graphs/neurograph_knowledge_graph.graphml",
                      "neurograph_knowledge_graph.graphml", "application/xml")

    st.divider()
    st.subheader("Analysis Reports")
    report_files = [
        ("eda_summary.md",                      "EDA Summary"),
        ("text_mining_summary.md",              "Text Mining Summary"),
        ("gap_scoring_summary.md",              "Gap Scoring Summary"),
        ("knowledge_graph_summary.md",          "Knowledge Graph Summary"),
        ("evidence_card_generation_summary.md", "Evidence Card Generation Summary"),
        ("top_gap_candidate_briefs.md",         "Top Gap Candidate Briefs"),
    ]
    c1, c2 = st.columns(2)
    for i, (fname, label) in enumerate(report_files):
        col = c1 if i % 2 == 0 else c2
        with col:
            _download_btn(
                f"Download: {label}",
                f"outputs/reports/{fname}",
                fname, "text/markdown",
            )

    st.divider()
    st.subheader("Report Previews")
    for fname, label in report_files:
        content = load_report(fname)
        with st.expander(label):
            if content:
                st.markdown(content)
            else:
                st.warning(f"{fname} not found.")


# ══════════════════════════════════════════════════════════════════════════════
# Page 7: Methodology & Limitations
# ══════════════════════════════════════════════════════════════════════════════

def page_methodology():
    st.title("Methodology & Limitations")

    st.subheader("Data Sources")
    st.markdown("""
| Source | Content | Access |
|--------|---------|--------|
| **Open Targets** | Disease-target association scores (Alzheimer's DOID:10652) | Public API |
| **Semantic Scholar** | Literature abstracts, citation counts, year, venue | Public API |
| **ClinicalTrials.gov** | Trial registrations, status, phases, protocol text | Public API |
    """)

    st.divider()
    st.subheader("Analytical Methods")

    with st.expander("Text Mining — TF-IDF + NMF"):
        st.markdown("""
- **TF-IDF** (Term Frequency–Inverse Document Frequency) applied to title + abstract for
  literature and to full protocol text for clinical trials.
- **NMF** (Non-negative Matrix Factorization) with k = 6 topics per corpus used for theme extraction.
- Topic labels are assigned manually based on top NMF terms and are approximate.
- K-Means clustering was also run as a cross-check but is not the primary theme source.
- These are statistical patterns, not curated knowledge-base categories.
        """)

    with st.expander("Target Mention Matching"):
        st.markdown("""
- Gene symbols from Open Targets are matched against document text using
  whole-word regex tokenisation (same tokeniser across Steps 5.5 and 7).
- Matching is case-insensitive and requires the symbol to appear as a standalone token.
- **Ambiguity risk:** short or common symbols (e.g. APP, ACE, CLU, GBA) may appear in
  non-biological contexts and produce false positives. These are flagged with
  `symbol_ambiguity_flag`.
- Mention counts reflect this corpus only. Zero mentions ≠ zero coverage in the broader literature.
        """)

    with st.expander("Research Gap Scoring"):
        st.markdown("""
Gap score is a weighted composite of five normalised signals:

| Component | Description |
|-----------|-------------|
| Evidence strength | Normalised Open Targets association score |
| Literature signal | Normalised literature mention count |
| Trial signal | Normalised trial mention count (inverted — fewer = higher gap) |
| Underexploration | Literature-to-evidence ratio |
| Lit-to-trial gap | Normalised literature-minus-trial gap |

- **`gap_interpretation_tier`** is the primary classification attribute.
- **`conservative_gap_score`** applies an ambiguity penalty for flagged symbols.
- A high gap score is a hypothesis-generating signal, not a therapeutic recommendation.
        """)

    with st.expander("Knowledge Graph Construction"):
        st.markdown("""
- **Nodes:** Disease (1), Target (499), Paper (150), Trial (150), LiteratureTheme (6), TrialTheme (6)
- **Edges:**
  - `disease_target_evidence` — disease → target via Open Targets score
  - `target_mentioned_in_paper` — target → paper via whole-word symbol match
  - `target_mentioned_in_trial` — target → trial via whole-word symbol match
  - `paper_has_literature_theme` — paper → NMF topic
  - `trial_has_theme` — trial → NMF topic
  - `theme_overlap` — LiteratureTheme ↔ TrialTheme shared keywords
- GraphML, GEXF, and JSON exports are available for downstream analysis (Gephi, D3.js, etc.).
        """)

    st.divider()
    st.subheader("Limitations")
    st.error(
        "NeuroGraph Agent is a research prototype for portfolio demonstration. "
        "It is NOT clinical decision support and must NOT inform clinical practice or drug development."
    )

    with st.expander("Full Limitations"):
        st.markdown("""
1. **Incomplete corpus.** Literature covers ~478 papers; trials cover ~1,000 registrations.
   Results do not represent the full published evidence landscape.

2. **Trial registration bias.** Clinical trial registrations describe interventions by
   drug name or mechanism class, not necessarily molecular target symbol. Low trial
   mentions for a target do not confirm it is untargeted in clinical research.

3. **Gene symbol ambiguity.** Short gene symbols can appear in unrelated text contexts.
   Ambiguity-flagged targets require manual expert validation.

4. **Open Targets scores are not therapeutic verdicts.** Association scores summarise
   multi-source research evidence. High scores reflect research attention, not drug
   development priority or clinical viability.

5. **NMF topics are approximate.** Topic labels are manually assigned to statistical
   patterns. They may not align with established clinical or ontological categories.

6. **No causal inference.** All graph edges represent text co-occurrence or database
   associations. No mechanistic, causal, or regulatory relationships are implied.

7. **Expert validation required.** All findings must be reviewed by qualified biomedical
   researchers or clinicians before being considered for any research or development decision.

8. **Not a systematic review.** This is a computational screening tool, not an
   exhaustive or clinically validated literature review.
        """)

    st.divider()
    st.subheader("Recommended Next Steps")
    st.markdown("""
- Extend the literature corpus via systematic search (PubMed, Cochrane)
- Add gene-disease mechanistic pathway data (Reactome, STRING)
- Integrate biobank-level variant data for genetic validation
- Apply Named Entity Recognition (NER) for more precise target-paper linking
- Use ontology-grounded topic labels (MeSH, HP, SNOMED)
- Add a user feedback loop for expert annotation of gap candidates
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════════════════

_ROUTER = {
    "Overview":                  page_overview,
    "Gap Candidates":            page_gap_candidates,
    "Target Explorer":           page_target_explorer,
    "Evidence Cards":            page_evidence_cards,
    "Themes":                    page_themes,
    "Knowledge Graph":           page_knowledge_graph,
    "Reports & Downloads":       page_reports_downloads,
    "Methodology & Limitations": page_methodology,
}

_ROUTER[page]()
