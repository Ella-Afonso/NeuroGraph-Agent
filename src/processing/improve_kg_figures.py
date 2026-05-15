"""
improve_kg_figures.py
NeuroGraph Agent — Improved Knowledge Graph Figures

Regenerates ONLY:
  outputs/figures/kg_03_target_gap_subgraph.png         (overwritten)
  outputs/figures/kg_03_target_gap_subgraph_improved.png (new)
  outputs/figures/kg_04_theme_subgraph.png              (overwritten)
  outputs/figures/kg_04_theme_subgraph_improved.png      (new)

Loads the existing saved graph (GraphML) — does NOT re-run the pipeline
or modify any data files, graph exports, edge/node tables, or reports.
"""

import io
import sys
import textwrap
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parents[2]
DATA    = ROOT / "Data" / "processed"
GRAPHS  = ROOT / "outputs" / "graphs"
FIGURES = ROOT / "outputs" / "figures"

DISEASE_NODE_ID = "alzheimers_disease"

TIER_COLOURS = {
    "Strong text-supported gap signal":             "#d62728",
    "Established / trial-covered target":           "#1f77b4",
    "Checked-zero text signal":                     "#aec7e8",
    "Ambiguous symbol; manual validation required": "#ff7f0e",
    "Low-priority / weak gap signal":               "#bcbd22",
}


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Target gap subgraph
# ══════════════════════════════════════════════════════════════════════════════

def plot_target_gap_subgraph(G, gap):
    """
    Hub-spoke: Disease + top 10 strong gap candidates + connected evidence nodes.

    Improvements over original:
    - Top 10 candidates only (was 15).
    - Target node size scales linearly with gap_score.
    - Disease-target edges are darker (#3a3a7a) and thicker (1.6 pt).
    - Target-evidence edges are lighter (#888888) and thinner (0.9 pt).
    - Evidence nodes placed near their parent target, not on a random outer ring.
    - Evidence nodes labelled "Paper evidence" / "Trial evidence" (not full titles).
    - Subtitle: "Edges represent computational evidence links, not clinical proof."
    - Saved as both kg_03_target_gap_subgraph.png and ..._improved.png.
    """
    TOP_N = 10

    strong = (
        gap[gap["gap_interpretation_tier"] == "Strong text-supported gap signal"]
        .sort_values("gap_score", ascending=False)
        .head(TOP_N)
    )
    sym_to_gs = dict(zip(strong["approved_symbol"], strong["gap_score"]))

    target_nodes = [
        f"target_{s}" for s in strong["approved_symbol"]
        if f"target_{s}" in G
    ]
    sub_nodes = {DISEASE_NODE_ID} | set(target_nodes)

    # Collect evidence nodes, grouped by their parent target
    target_papers = {tn: [] for tn in target_nodes}
    target_trials = {tn: [] for tn in target_nodes}
    for tn in target_nodes:
        for _, dst, d in G.out_edges(tn, data=True):
            et = d.get("edge_type", "")
            if et == "target_mentioned_in_paper":
                target_papers[tn].append(dst)
            elif et == "target_mentioned_in_trial":
                target_trials[tn].append(dst)

    all_evidence = set()
    for v in target_papers.values():
        all_evidence.update(v)
    for v in target_trials.values():
        all_evidence.update(v)
    sub_nodes |= all_evidence
    SG = G.subgraph(sub_nodes).copy()

    # ── Positions ──────────────────────────────────────────────────────────────
    INNER_R = 1.85   # radius of target ring around disease node
    OUTER_R = 1.20   # distance of each evidence node from its parent target

    pos = {DISEASE_NODE_ID: (0.0, 0.0)}
    n_t = max(len(target_nodes), 1)
    target_angles = {}

    for i, tn in enumerate(target_nodes):
        # Start at top (π/2) and go clockwise
        angle = np.pi / 2.0 - 2.0 * np.pi * i / n_t
        pos[tn] = (INNER_R * np.cos(angle), INNER_R * np.sin(angle))
        target_angles[tn] = angle

    # Place evidence nodes in a small fan around their parent target,
    # pointing outward from the disease node
    for tn in target_nodes:
        children = target_papers[tn] + target_trials[tn]
        n_c = len(children)
        if n_c == 0:
            continue
        base_a = target_angles[tn]
        for j, cn in enumerate(children):
            spread = 0.0 if n_c == 1 else (j - (n_c - 1) / 2.0) * 0.38
            a = base_a + spread
            pos[cn] = (
                pos[tn][0] + OUTER_R * np.cos(a),
                pos[tn][1] + OUTER_R * np.sin(a),
            )

    # ── Node colours and sizes ─────────────────────────────────────────────────
    node_list = list(SG.nodes())
    node_colours, node_sizes = [], []
    for n in node_list:
        nd  = SG.nodes[n]
        nt  = str(nd.get("node_type", ""))
        if nt == "Disease":
            node_colours.append("#5e2d91")
            node_sizes.append(1800)
        elif nt == "Target":
            tier = str(nd.get("gap_interpretation_tier", ""))
            node_colours.append(TIER_COLOURS.get(tier, "#aec7e8"))
            sym = str(nd.get("label", ""))
            gs  = float(sym_to_gs.get(sym, 0.15))
            node_sizes.append(int(350 + gs * 2000))
        elif nt == "Paper":
            node_colours.append("#2ca02c")
            node_sizes.append(220)
        elif nt == "Trial":
            node_colours.append("#ff7f0e")
            node_sizes.append(220)
        else:
            node_colours.append("#aaaaaa")
            node_sizes.append(120)

    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    # ── Edges: separated by type for different visual weight ──────────────────
    disease_edges = [
        (u, v) for u, v, d in SG.edges(data=True)
        if d.get("edge_type") == "disease_target_evidence"
    ]
    evidence_edges = [
        (u, v) for u, v, d in SG.edges(data=True)
        if d.get("edge_type") in ("target_mentioned_in_paper",
                                   "target_mentioned_in_trial")
    ]

    # Disease-target: dark, prominent arrows
    nx.draw_networkx_edges(
        SG, pos, edgelist=disease_edges, ax=ax,
        edge_color="#3a3a7a", alpha=0.62, arrows=True,
        arrowsize=14, width=1.6,
        connectionstyle="arc3,rad=0.06",
    )
    # Target-evidence: lighter, thinner
    if evidence_edges:
        nx.draw_networkx_edges(
            SG, pos, edgelist=evidence_edges, ax=ax,
            edge_color="#888888", alpha=0.50, arrows=True,
            arrowsize=10, width=0.9,
        )

    nx.draw_networkx_nodes(
        SG, pos, nodelist=node_list, ax=ax,
        node_color=node_colours, node_size=node_sizes,
        alpha=0.92, linewidths=0.6, edgecolors="white",
    )

    # ── Labels ─────────────────────────────────────────────────────────────────
    # Disease node: white text on dark purple
    nx.draw_networkx_labels(
        SG, pos,
        labels={DISEASE_NODE_ID: "Alzheimer's\nDisease"},
        ax=ax, font_size=8.5, font_weight="bold", font_color="white",
    )
    # Target symbols
    target_labels = {
        tn: str(SG.nodes[tn].get("label", tn.replace("target_", "")))
        for tn in target_nodes if tn in SG
    }
    nx.draw_networkx_labels(
        SG, pos, labels=target_labels,
        ax=ax, font_size=8, font_weight="bold", font_color="#111111",
    )
    # Evidence nodes: generic short label
    ev_labels = {}
    for n, d in SG.nodes(data=True):
        if d.get("node_type") == "Paper":
            ev_labels[n] = "Paper\nevidence"
        elif d.get("node_type") == "Trial":
            ev_labels[n] = "Trial\nevidence"
    if ev_labels:
        nx.draw_networkx_labels(
            SG, pos, labels=ev_labels,
            ax=ax, font_size=5.5, font_color="#444444",
        )

    # ── Legends ────────────────────────────────────────────────────────────────
    colour_patches = [
        mpatches.Patch(color="#5e2d91", label="Disease"),
        mpatches.Patch(
            color=TIER_COLOURS["Strong text-supported gap signal"],
            label="Strong gap signal (target)"),
        mpatches.Patch(color="#2ca02c", label="Paper with symbol match"),
        mpatches.Patch(color="#ff7f0e", label="Trial with symbol match"),
        mpatches.Patch(color="#3a3a7a", label="Disease-target evidence edge"),
        mpatches.Patch(color="#888888", label="Target-document mention edge"),
    ]
    leg1 = ax.legend(
        handles=colour_patches, loc="upper right",
        fontsize=7.5, title="Node / edge types", title_fontsize=8,
        framealpha=0.93,
    )
    ax.add_artist(leg1)

    # Size reference legend (scatter points at three gap-score levels)
    size_h = [
        ax.scatter(
            [], [],
            s=int(350 + gs * 2000),
            c=TIER_COLOURS["Strong text-supported gap signal"],
            alpha=0.92, edgecolors="white", linewidths=0.5,
        )
        for gs in [0.20, 0.50, 0.75]
    ]
    ax.legend(
        handles=size_h,
        labels=["Gap score ~0.20", "Gap score ~0.50", "Gap score ~0.75"],
        loc="upper left", fontsize=7,
        title="Target node size = gap score", title_fontsize=7.5,
        framealpha=0.93,
    )

    # ── Title and disclaimer ───────────────────────────────────────────────────
    ax.set_title(
        "Target Research Gap Subgraph\n"
        "Disease + Top 10 Strong Gap Candidates (by gap score)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.axis("off")
    fig.text(
        0.5, 0.005,
        "Edges represent computational evidence links, not clinical proof.  "
        "Hypothesis-generating only — not clinical decision support.",
        ha="center", fontsize=7.5, color="#555555", style="italic",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    for fname in ("kg_03_target_gap_subgraph.png",
                  "kg_03_target_gap_subgraph_improved.png"):
        plt.savefig(FIGURES / fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Theme subgraph
# ══════════════════════════════════════════════════════════════════════════════

def plot_theme_subgraph(G):
    """
    Bipartite layout: LiteratureTheme (left) vs TrialTheme (right).

    Improvements over original:
    - Duplicate labels disambiguated with (T{id}), e.g. "Neuroinflammation & Immune (T1)".
    - Nodes sorted on each side to minimise edge crossings.
    - Edge width proportional to average document count of the connected themes.
    - Column headers: "Literature Themes" / "Trial Themes" (one per side, no repeats).
    - Node size proportional to document count.
    - Subtitle: "Theme overlap based on shared top terms."
    - Saved as both kg_04_theme_subgraph.png and ..._improved.png.
    """
    theme_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") in ("LiteratureTheme", "TrialTheme")
    ]
    SG = G.subgraph(theme_nodes).copy()

    lit_nodes   = [n for n, d in SG.nodes(data=True)
                   if d.get("node_type") == "LiteratureTheme"]
    trial_nodes = [n for n, d in SG.nodes(data=True)
                   if d.get("node_type") == "TrialTheme"]

    # ── Disambiguate duplicate labels ──────────────────────────────────────────
    def _disambiguate(nodes_list):
        raw    = {n: str(SG.nodes[n].get("label", n)) for n in nodes_list}
        counts = {}
        for lbl in raw.values():
            counts[lbl] = counts.get(lbl, 0) + 1
        out = {}
        for n, lbl in raw.items():
            if counts[lbl] > 1:
                try:
                    tid = f"T{int(float(SG.nodes[n].get('topic_id', '?')))}"
                except (TypeError, ValueError):
                    tid = "T?"
                out[n] = f"{lbl} ({tid})"
            else:
                out[n] = lbl
        return out

    lit_labels   = _disambiguate(lit_nodes)
    trial_labels = _disambiguate(trial_nodes)
    all_labels   = {**lit_labels, **trial_labels}

    # ── Overlap edges and document-count-based widths ─────────────────────────
    overlap_edges = [
        (u, v) for u, v, d in SG.edges(data=True)
        if d.get("edge_type") == "theme_overlap"
    ]

    def _doc_count(n):
        nd = SG.nodes[n]
        for key in ("n_papers", "n_trials"):
            val = nd.get(key)
            if val is not None:
                try:
                    f = float(val)
                    if f > 0:
                        return f
                except (TypeError, ValueError):
                    pass
        return 0.0

    edge_weights = {
        (u, v): (_doc_count(u) + _doc_count(v)) / 2.0
        for u, v in overlap_edges
    }
    max_w = max(edge_weights.values(), default=1.0)
    min_w = min(edge_weights.values(), default=max_w)
    w_rng = max(max_w - min_w, 1.0)

    def _edge_width(u, v):
        w = edge_weights.get((u, v), min_w)
        return 1.5 + ((w - min_w) / w_rng) * 2.5   # range [1.5, 4.0]

    # ── Sort nodes to minimise edge crossings ─────────────────────────────────
    # Sort trial nodes by topic_id (stable, predictable vertical order)
    trial_sorted = sorted(
        trial_nodes,
        key=lambda n: float(SG.nodes[n].get("topic_id", 99) or 99),
    )
    trial_idx = {n: i for i, n in enumerate(trial_sorted)}

    # Build lit→trial adjacency from overlap edges
    lit_adj = {n: [] for n in lit_nodes}
    for u, v in overlap_edges:
        if u in lit_adj:
            lit_adj[u].append(v)

    def _lit_sort_key(n):
        connected = lit_adj.get(n, [])
        if connected:
            return float(np.mean([trial_idx.get(c, 99) for c in connected]))
        return 99.0     # disconnected lit nodes go to the bottom

    lit_sorted = sorted(lit_nodes, key=_lit_sort_key)

    # ── Layout positions ───────────────────────────────────────────────────────
    X_LIT   = -2.2
    X_TRIAL =  2.2
    V_STEP  =  1.15    # vertical gap between nodes

    pos = {}
    n_l = len(lit_sorted)
    n_r = len(trial_sorted)

    for i, n in enumerate(lit_sorted):
        pos[n] = (X_LIT,   ((n_l - 1) / 2.0 - i) * V_STEP)
    for i, n in enumerate(trial_sorted):
        pos[n] = (X_TRIAL, ((n_r - 1) / 2.0 - i) * V_STEP)

    # ── Node sizes: proportional to document count ─────────────────────────────
    lit_max   = max((_doc_count(n) for n in lit_nodes),   default=1.0)
    trial_max = max((_doc_count(n) for n in trial_nodes), default=1.0)

    def _node_size(n):
        nt    = str(SG.nodes[n].get("node_type", ""))
        denom = lit_max if nt == "LiteratureTheme" else trial_max
        return int(700 + (_doc_count(n) / max(denom, 1.0)) * 800)

    # ── Draw ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    # Edges: one draw call per edge so each can have its own width
    for u, v in overlap_edges:
        nx.draw_networkx_edges(
            SG, pos, edgelist=[(u, v)], ax=ax,
            edge_color="#606060", alpha=0.72,
            arrows=True, arrowsize=14,
            width=_edge_width(u, v),
            connectionstyle="arc3,rad=0.08",
        )

    nx.draw_networkx_nodes(
        SG, pos, nodelist=lit_sorted,
        node_color=["#c94040"] * len(lit_sorted),
        node_size=[_node_size(n) for n in lit_sorted],
        alpha=0.90, linewidths=0.8, edgecolors="white", ax=ax,
    )
    nx.draw_networkx_nodes(
        SG, pos, nodelist=trial_sorted,
        node_color=["#7a4030"] * len(trial_sorted),
        node_size=[_node_size(n) for n in trial_sorted],
        alpha=0.90, linewidths=0.8, edgecolors="white", ax=ax,
    )

    # Labels: wrap long names to keep them readable
    wrapped = {
        n: "\n".join(textwrap.wrap(lbl, width=20))
        for n, lbl in all_labels.items()
    }
    nx.draw_networkx_labels(
        SG, pos, labels=wrapped,
        ax=ax, font_size=7.5, font_color="#111111", font_weight="bold",
    )

    # ── Column headers (one per side, above the top node) ─────────────────────
    y_top = max(p[1] for p in pos.values())
    hdr_y = y_top + V_STEP * 0.88

    ax.text(X_LIT,   hdr_y, "Literature Themes",
            ha="center", va="bottom", fontsize=11,
            fontweight="bold", color="#c94040")
    ax.text(X_TRIAL, hdr_y, "Trial Themes",
            ha="center", va="bottom", fontsize=11,
            fontweight="bold", color="#7a4030")

    # Thin vertical divider
    y_bot = min(p[1] for p in pos.values())
    ax.axvline(x=0, ymin=0.05, ymax=0.93, color="#cccccc",
               linewidth=0.8, linestyle="--", alpha=0.6)

    # ── Legend ─────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color="#c94040", label="Literature Theme (NMF)"),
        mpatches.Patch(color="#7a4030", label="Trial Theme (NMF)"),
        mpatches.Patch(color="#606060", label="Theme overlap (shared top terms); "
                       "edge width = document count"),
    ]
    ax.legend(handles=patches, loc="lower center", fontsize=8.5, ncol=3,
              framealpha=0.92, bbox_to_anchor=(0.5, -0.05))

    ax.text(
        0.01, 0.02,
        "Node size proportional\nto document count",
        transform=ax.transAxes, fontsize=6.5, color="#666666", style="italic",
    )

    # ── Title and disclaimer ───────────────────────────────────────────────────
    ax.set_title(
        "Theme Subgraph: NMF Topics — Literature vs Clinical Trials",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.axis("off")
    fig.text(
        0.5, 0.01,
        "Theme overlap based on shared top terms.  "
        "NMF topic labels are approximate — require expert interpretation.",
        ha="center", fontsize=7.5, color="#555555", style="italic",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])

    for fname in ("kg_04_theme_subgraph.png",
                  "kg_04_theme_subgraph_improved.png"):
        plt.savefig(FIGURES / fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("NeuroGraph — Improved Knowledge Graph Figure Generation")
    print("=" * 60)

    graphml_path = GRAPHS / "neurograph_knowledge_graph.graphml"
    if not graphml_path.exists():
        sys.exit(f"ERROR: {graphml_path} not found. Run build_knowledge_graph.py first.")

    print("\nLoading saved graph from GraphML ...")
    G   = nx.read_graphml(str(graphml_path))
    gap = pd.read_csv(DATA / "research_gap_scores.csv")
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Gap scores: {len(gap)} targets")

    print("\nGenerating improved figures ...")
    plot_target_gap_subgraph(G, gap)
    plot_theme_subgraph(G)

    print("\n" + "=" * 60)
    print("Done. Outputs in outputs/figures/")
    print("  kg_03_target_gap_subgraph.png          (overwritten)")
    print("  kg_03_target_gap_subgraph_improved.png (new)")
    print("  kg_04_theme_subgraph.png               (overwritten)")
    print("  kg_04_theme_subgraph_improved.png      (new)")
    print("=" * 60)


if __name__ == "__main__":
    main()
