"""
evidence_cards.py — Streamlit helpers for the Evidence Cards dashboard page.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.components import map_tier, TIER_COLOURS_CSS

ROOT = Path(__file__).resolve().parents[2]

_NO_CARD_MSG = (
    "Evidence card not yet generated for this target. "
    "Run: python -m src.processing.generate_evidence_cards"
)

SIGNAL_COLOUR = {
    # evidence strength
    "High evidence signal":          "#2ca02c",
    "Moderate evidence signal":      "#1f77b4",
    "Lower evidence signal":         "#9467bd",
    # literature
    "Strong literature signal":      "#2ca02c",
    "Moderate literature signal":    "#1f77b4",
    "Weak literature signal":        "#ff7f0e",
    "No detected literature signal": "#aaaaaa",
    # trial
    "High trial text signal":        "#2ca02c",
    "Moderate trial text signal":    "#1f77b4",
    "Low trial text signal":         "#ff7f0e",
    "No detected trial text signal": "#aaaaaa",
}

# Use the canonical colour map from components (keyed on raw tier values)
TIER_COLOUR = TIER_COLOURS_CSS


@st.cache_data
def load_evidence_cards() -> pd.DataFrame | None:
    path = ROOT / "Data" / "processed" / "target_evidence_cards.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df["gap_rank"] = pd.to_numeric(df["gap_rank"], errors="coerce")
    return df


def _coloured(label: str, colour_map: dict[str, str]) -> str:
    colour = colour_map.get(label, "#555")
    return f"<span style='color:{colour};font-weight:bold'>{label}</span>"


def render_card_summary(row: pd.Series) -> None:
    """Render an evidence card summary inside a Streamlit page."""
    sym  = str(row.get("approved_symbol", "—"))
    name = str(row.get("approved_name",   "—"))
    tier = str(row.get("gap_interpretation_tier", "—"))

    lit_ment   = int(row.get("literature_mentions", 0) or 0)
    tri_ment   = int(row.get("trial_mentions",      0) or 0)
    n_papers   = int(row.get("n_matched_papers",    0) or 0)
    n_trials   = int(row.get("n_matched_trials",    0) or 0)
    assoc      = row.get("association_score", 0.0)

    tier_lbl    = map_tier(tier)
    tier_colour = TIER_COLOUR.get(tier, "#555")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"### {sym}")
    st.caption(name)
    st.markdown(
        f"**Interpretation:** <span style='color:{tier_colour};font-weight:bold'>{tier_lbl}</span>",
        unsafe_allow_html=True,
    )

    # ── Signal labels ──────────────────────────────────────────────────────────
    ev_lbl  = str(row.get("evidence_strength_label",  "—"))
    lit_lbl = str(row.get("literature_signal_label",  "—"))
    tri_lbl = str(row.get("trial_signal_label",        "—"))

    st.markdown(
        f"**Evidence:** {_coloured(ev_lbl, SIGNAL_COLOUR)} &nbsp;|&nbsp; "
        f"**Literature:** {_coloured(lit_lbl, SIGNAL_COLOUR)} &nbsp;|&nbsp; "
        f"**Trial text:** {_coloured(tri_lbl, SIGNAL_COLOUR)}",
        unsafe_allow_html=True,
    )

    # ── Metrics: two rows for clarity ─────────────────────────────────────────
    st.caption("**Raw corpus mention counts** (whole-word symbol matching across all retrieved documents):")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gap Rank",       int(row["gap_rank"]) if pd.notna(row.get("gap_rank")) else "—")
    c2.metric("Gap Score",      f"{float(row['gap_score']):.4f}"
              if pd.notna(row.get("gap_score")) else "—",
              help="Composite gap score; higher = stronger evidence vs lower trial text coverage.")
    c3.metric("Literature Mentions",
              lit_ment,
              help="Raw whole-word symbol matches across all retrieved literature documents.")
    c4.metric("Trial Mentions",
              tri_ment,
              help="Raw whole-word symbol matches across all retrieved trial registrations.")

    st.caption(
        "**Graph-linked records** (subset of corpus evidence linked inside the knowledge graph):"
    )
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Assoc. Score",    f"{float(assoc):.4f}" if pd.notna(assoc) else "—",
              help="Open Targets disease-target association score. Not therapeutic proof.")
    c6.metric("",                "")
    c7.metric("Matched Papers",  n_papers,
              help="Papers directly linked to this target via graph edges (knowledge graph subset).")
    c8.metric("Matched Trials",  n_trials,
              help="Trials directly linked to this target via graph edges (knowledge graph subset).")

    # ── Mismatch note ─────────────────────────────────────────────────────────
    # Signal > 0 but no matched records: explain the graph sparsity
    sig_but_no_paper = lit_ment > 0 and n_papers == 0
    sig_but_no_trial = tri_ment > 0 and n_trials == 0
    only_ot_evidence = pd.notna(assoc) and float(assoc) > 0 and lit_ment == 0 and tri_ment == 0

    if sig_but_no_paper or sig_but_no_trial:
        notes = []
        if sig_but_no_paper:
            notes.append(
                f"{sym} has {lit_ment} literature mention(s) in the corpus, but no paper "
                "records are linked in the current knowledge graph. The graph includes only a "
                "selected subset of evidence edges."
            )
        if sig_but_no_trial:
            notes.append(
                f"{sym} has {tri_ment} trial mention(s) in the corpus, but no trial records "
                "are linked in the current knowledge graph. Trial registrations often describe "
                "interventions by drug name rather than target symbol, so the mention count "
                "may include indirect references."
            )
        st.warning(
            " ".join(notes) + " "
            "These counts reflect the retrieved corpus sample only, not the full published evidence landscape."
        )
    elif only_ot_evidence:
        st.info(
            f"This target has Open Targets association evidence (score: {float(assoc):.4f}), "
            "but no direct paper or trial evidence records were linked in the current graph subset. "
            "This may reflect alias-only coverage, drug-name-based trial registration, "
            "or absence from this corpus sample."
        )

    # ── Theme context ──────────────────────────────────────────────────────────
    dom_lit   = str(row.get("dominant_literature_theme", "—"))
    dom_trial = str(row.get("dominant_trial_theme",       "—"))
    if dom_lit not in ("—", "nan") or dom_trial not in ("—", "nan"):
        st.caption(
            f"Dominant literature theme: **{dom_lit}** &nbsp;|&nbsp; "
            f"Dominant trial theme: **{dom_trial}**"
        )

    st.divider()

    # ── Why interesting ────────────────────────────────────────────────────────
    why = str(row.get("why_interesting", ""))
    if why and why != "nan":
        with st.expander("Why this target may be interesting"):
            st.write(why)

    # ── Limitations ───────────────────────────────────────────────────────────
    lims = str(row.get("key_limitations", ""))
    if lims and lims != "nan":
        with st.expander("Limitations"):
            for lim in lims.split(" | "):
                st.write(f"- {lim}")

    # ── Next steps ─────────────────────────────────────────────────────────────
    steps = str(row.get("suggested_next_steps", ""))
    if steps and steps != "nan":
        with st.expander("Suggested next validation steps"):
            for i, step in enumerate(steps.split(" | "), 1):
                st.write(f"{i}. {step}")

    # ── Download button ────────────────────────────────────────────────────────
    card_rel = str(row.get("evidence_card_path", ""))
    if card_rel and card_rel != "nan":
        card_path = ROOT / card_rel
        if card_path.exists():
            st.download_button(
                label=f"Download {sym} Evidence Card (Markdown)",
                data=card_path.read_bytes(),
                file_name=card_path.name,
                mime="text/markdown",
            )
        else:
            st.caption(_NO_CARD_MSG)
    else:
        st.caption(_NO_CARD_MSG)
