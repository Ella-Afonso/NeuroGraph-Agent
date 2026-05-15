"""
charts.py — Matplotlib figure builders for NeuroGraph dashboard.

Each function returns a matplotlib Figure.  Callers are responsible for:
    st.pyplot(fig)
    plt.close(fig)
No seaborn, plotly, or pyvis is used.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

TIER_COLOURS = {
    "Strong text-supported gap signal":             "#d62728",
    "Established / trial-covered target":           "#1f77b4",
    "Checked-zero text signal":                     "#aec7e8",
    "Ambiguous symbol; manual validation required": "#ff7f0e",
    "Low-priority / weak gap signal":               "#bcbd22",
}
_FALLBACK_COLOUR = "#7f7f7f"
_DISCLAIMER = "Hypothesis-generating only — not clinical decision support."


def _tc(tier: str) -> str:
    return TIER_COLOURS.get(tier, _FALLBACK_COLOUR)


# ── Gap candidate charts ───────────────────────────────────────────────────────

def fig_top_gap_bar(df, n: int = 20, title: str = "Top Gap Candidates by Gap Score"):
    """Horizontal bar chart of top-N targets by gap_score, coloured by tier."""
    sub = (
        df.sort_values("gap_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    colours = [_tc(t) for t in sub["gap_interpretation_tier"]]
    height = max(4, len(sub) * 0.42)
    fig, ax = plt.subplots(figsize=(9, height))
    bars = ax.barh(
        sub["approved_symbol"][::-1],
        sub["gap_score"][::-1],
        color=colours[::-1],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar_label(
        bars,
        labels=[f"{v:.3f}" for v in sub["gap_score"][::-1]],
        padding=3, fontsize=8,
    )
    ax.set_xlabel("Gap Score")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, (sub["gap_score"].max() or 1.0) * 1.18)

    seen_tiers = sub["gap_interpretation_tier"].unique()
    patches = [mpatches.Patch(color=_tc(t), label=t) for t in seen_tiers]
    ax.legend(handles=patches, fontsize=7, loc="lower right")
    ax.text(0.01, -0.08, _DISCLAIMER,
            transform=ax.transAxes, fontsize=6, color="grey", style="italic")
    plt.tight_layout()
    return fig


def fig_lit_vs_trial_scatter(df, title: str = "Literature vs Trial Mentions"):
    """Scatter plot coloured by gap_interpretation_tier."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for tier, grp in df.groupby("gap_interpretation_tier", sort=False):
        ax.scatter(
            grp["trial_mentions"],
            grp["literature_mentions"],
            c=_tc(tier),
            label=tier,
            alpha=0.72,
            s=45,
            edgecolors="white",
            linewidths=0.5,
        )
    top5 = df.nlargest(5, "gap_score") if len(df) >= 5 else df
    for _, row in top5.iterrows():
        ax.annotate(
            row["approved_symbol"],
            (row["trial_mentions"], row["literature_mentions"]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )
    ax.set_xlabel("Trial Mentions (corpus sample)")
    ax.set_ylabel("Literature Mentions (corpus sample)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.85)
    ax.text(0.01, 0.01, _DISCLAIMER,
            transform=ax.transAxes, fontsize=6, color="grey", style="italic")
    plt.tight_layout()
    return fig


# ── Theme / topic charts ───────────────────────────────────────────────────────

def fig_topic_sizes_bar(topics_df, count_col: str, title: str):
    """Horizontal bar chart of document counts per NMF topic."""
    height = max(3, len(topics_df) * 0.55)
    fig, ax = plt.subplots(figsize=(8, height))
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(topics_df)))
    bars = ax.barh(
        topics_df["topic_label"][::-1],
        topics_df[count_col][::-1],
        color=colours[::-1],
        edgecolor="white",
    )
    ax.bar_label(
        bars,
        labels=list(topics_df[count_col][::-1]),
        padding=3, fontsize=9,
    )
    ax.set_xlabel("Count")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, topics_df[count_col].max() * 1.15)
    plt.tight_layout()
    return fig


# ── Knowledge graph summary charts ────────────────────────────────────────────

def fig_kg_node_types(nodes_df):
    """Bar chart of node counts by node_type."""
    type_counts = nodes_df["node_type"].value_counts()
    ordered = ["Disease", "Target", "Paper", "Trial", "LiteratureTheme", "TrialTheme"]
    types  = [t for t in ordered if t in type_counts.index]
    counts = [type_counts[t] for t in types]
    colour_map = {
        "Disease":         "#9467bd",
        "Target":          "#1f77b4",
        "Paper":           "#2ca02c",
        "Trial":           "#ff7f0e",
        "LiteratureTheme": "#d62728",
        "TrialTheme":      "#8c564b",
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        types, counts,
        color=[colour_map.get(t, _FALLBACK_COLOUR) for t in types],
        edgecolor="white", linewidth=0.8,
    )
    ax.bar_label(bars, labels=counts, padding=3, fontsize=10)
    ax.set_title("Knowledge Graph: Node Types", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.18)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def fig_kg_edge_types(edges_df):
    """Horizontal bar chart of edge counts by edge_type."""
    type_counts = edges_df["edge_type"].value_counts()
    height = max(3, len(type_counts) * 0.55)
    fig, ax = plt.subplots(figsize=(8, height))
    bars = ax.barh(
        type_counts.index[::-1],
        type_counts.values[::-1],
        color="#5c9bd6",
        edgecolor="white",
    )
    ax.bar_label(
        bars,
        labels=list(type_counts.values[::-1]),
        padding=3, fontsize=10,
    )
    ax.set_title("Knowledge Graph: Edge Types", fontsize=11, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_xlim(0, type_counts.max() * 1.2)
    plt.tight_layout()
    return fig
