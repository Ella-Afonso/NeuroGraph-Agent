"""
build_knowledge_graph.py
NeuroGraph Agent – Step 7: Knowledge Graph Construction

Builds a NetworkX DiGraph connecting Disease → Target → Paper/Trial → Theme.
Uses gap_interpretation_tier as the primary node-colouring attribute.

IMPORTANT: Hypothesis-generating only. Not clinical decision support.
"""

import io
import sys
import os
import re
import json
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# UTF-8 fix for Windows cp1252 console
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "processed"
GRAPHS = ROOT / "outputs" / "graphs"
FIGURES = ROOT / "outputs" / "figures"
REPORTS = ROOT / "outputs" / "reports"

GRAPHS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
PAPER_LIMIT = 150
TRIAL_LIMIT = 150
TRIAL_PRIORITY_STATUSES = ["COMPLETED", "RECRUITING"]
DISEASE_NODE_ID = "alzheimers_disease"
DISEASE_LABEL = "Alzheimer's Disease"

# Tier colours for visualisation
TIER_COLOURS = {
    "Strong text-supported gap signal":              "#d62728",
    "Established / trial-covered target":            "#1f77b4",
    "Checked-zero text signal":                      "#aec7e8",
    "Ambiguous symbol; manual validation required":  "#ff7f0e",
    "Low-priority / weak gap signal":                "#bcbd22",
}

# ── Unicode sanitisation ───────────────────────────────────────────────────────
_UNICODE_MAP = {
    "β": "beta",
    "α": "alpha",
    "γ": "gamma",
    "δ": "delta",
    "κ": "kappa",
    "τ": "tau",
    "ω": "omega",
    "μ": "mu",
}


def sanitise(text):
    """Replace Greek/non-ASCII characters for GraphML/GEXF compatibility."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    for char, replacement in _UNICODE_MAP.items():
        text = text.replace(char, replacement)
    return text.encode("ascii", errors="replace").decode("ascii")


# ── Tokeniser (same regex as Step 5.5) ────────────────────────────────────────
_TOKENISE = re.compile(r'[A-Za-z0-9]+')


def tokenise(text):
    if not isinstance(text, str) or not text.strip():
        return Counter()
    return Counter(_TOKENISE.findall(text.lower()))


def symbol_in_doc(symbol, doc_counter):
    """Whole-word check: symbol must appear as a standalone token."""
    return doc_counter.get(symbol.lower(), 0) > 0


# ── Load data ──────────────────────────────────────────────────────────────────
def load_data():
    print("Loading input data ...")

    gap = pd.read_csv(DATA / "research_gap_scores.csv")
    lit_assign = pd.read_csv(DATA / "literature_topic_assignments.csv")
    trial_assign = pd.read_csv(DATA / "clinical_trial_topic_assignments.csv")
    lit_topics = pd.read_csv(DATA / "literature_topics.csv")
    trial_topics = pd.read_csv(DATA / "clinical_trial_topics.csv")
    lit_clean = pd.read_csv(DATA / "literature_clean.csv")
    trial_clean = pd.read_csv(DATA / "clinical_trials_clean.csv")

    print(f"  research_gap_scores:          {len(gap)} rows")
    print(f"  literature_topic_assignments: {len(lit_assign)} rows")
    print(f"  clinical_trial_assignments:   {len(trial_assign)} rows")
    print(f"  literature_clean:             {len(lit_clean)} rows")
    print(f"  clinical_trials_clean:        {len(trial_clean)} rows")

    return gap, lit_assign, trial_assign, lit_topics, trial_topics, lit_clean, trial_clean


# ── Select papers and trials ───────────────────────────────────────────────────
def select_papers(lit_assign, lit_clean):
    """Top PAPER_LIMIT papers by citation_count."""
    if "citation_count" not in lit_assign.columns:
        lit_assign = lit_assign.merge(
            lit_clean[["paper_id", "citation_count"]], on="paper_id", how="left"
        )
    top = (
        lit_assign.sort_values("citation_count", ascending=False)
        .drop_duplicates(subset="paper_id")
        .head(PAPER_LIMIT)
        .reset_index(drop=True)
    )
    print(f"  Selected {len(top)} papers (top by citation_count)")
    return top


def select_trials(trial_assign, trial_clean):
    """Top TRIAL_LIMIT trials, prioritising COMPLETED/RECRUITING."""
    if "overall_status" not in trial_assign.columns:
        trial_assign = trial_assign.merge(
            trial_clean[["nct_id", "overall_status"]], on="nct_id", how="left"
        )

    def _priority(status):
        if status == "COMPLETED":  return 2
        if status == "RECRUITING": return 1
        return 0

    ta = trial_assign.copy()
    ta["_pri"] = ta["overall_status"].apply(_priority)
    top = (
        ta.sort_values(["_pri", "topic_score"], ascending=[False, False])
        .drop_duplicates(subset="nct_id")
        .head(TRIAL_LIMIT)
        .reset_index(drop=True)
    )
    print(f"  Selected {len(top)} trials (COMPLETED/RECRUITING prioritised)")
    return top


# ── Document counters for mention re-scanning ─────────────────────────────────
def build_doc_counters(lit_clean, trial_clean, selected_papers, selected_trials):
    """Pre-build token Counter for each selected paper and trial."""
    paper_ids = set(selected_papers["paper_id"])
    trial_ids = set(selected_trials["nct_id"])

    paper_counters = {}
    for _, row in lit_clean[lit_clean["paper_id"].isin(paper_ids)].iterrows():
        paper_counters[row["paper_id"]] = tokenise(row.get("paper_text", ""))

    trial_counters = {}
    for _, row in trial_clean[trial_clean["nct_id"].isin(trial_ids)].iterrows():
        trial_counters[row["nct_id"]] = tokenise(row.get("trial_text", ""))

    print(
        f"  Built {len(paper_counters)} paper counters, "
        f"{len(trial_counters)} trial counters"
    )
    return paper_counters, trial_counters


# ── Build graph ────────────────────────────────────────────────────────────────
def build_graph(
    gap, lit_assign, trial_assign, lit_topics, trial_topics,
    lit_clean, trial_clean,
    selected_papers, selected_trials,
    paper_counters, trial_counters,
):
    G = nx.DiGraph()
    print("\nBuilding knowledge graph ...")

    # ── 1. Disease node ────────────────────────────────────────────────────────
    G.add_node(
        DISEASE_NODE_ID,
        node_type="Disease",
        label=DISEASE_LABEL,
        description="Alzheimer's disease (DOID:10652)",
    )
    print(f"  Added 1 Disease node")

    # ── 2. Target nodes ────────────────────────────────────────────────────────
    for _, row in gap.iterrows():
        nid = f"target_{row['approved_symbol']}"
        tier = sanitise(str(row.get("gap_interpretation_tier", "")))
        G.add_node(
            nid,
            node_type="Target",
            label=sanitise(str(row["approved_symbol"])),
            approved_name=sanitise(str(row.get("approved_name", ""))),
            target_id=str(row.get("target_id", "")),
            biotype=str(row.get("biotype", "")),
            association_score=float(row.get("association_score", 0.0)),
            gap_score=float(row.get("gap_score", 0.0)),
            conservative_gap_score=float(row.get("conservative_gap_score", 0.0)),
            gap_rank=int(row["gap_rank"]) if not pd.isna(row.get("gap_rank")) else 0,
            gap_category=str(row.get("gap_category", "")),
            gap_interpretation_tier=tier,
            text_coverage_status=str(row.get("text_coverage_status", "")),
            literature_mentions=int(row.get("literature_mentions", 0)),
            trial_mentions=int(row.get("trial_mentions", 0)),
            symbol_ambiguity_flag=str(row.get("symbol_ambiguity_flag", "")),
            mention_match_quality=str(row.get("mention_match_quality", "")),
        )
    print(f"  Added {len(gap)} Target nodes")

    # ── 3. Disease → Target edges ──────────────────────────────────────────────
    for _, row in gap.iterrows():
        nid = f"target_{row['approved_symbol']}"
        G.add_edge(
            DISEASE_NODE_ID, nid,
            edge_type="disease_target_evidence",
            association_score=float(row.get("association_score", 0.0)),
            evidence_source="Open Targets",
        )
    print(f"  Added {len(gap)} disease_target_evidence edges")

    # ── 4. Paper nodes ─────────────────────────────────────────────────────────
    paper_node_ids = {}
    for _, row in selected_papers.iterrows():
        pid = str(row["paper_id"])
        nid = f"paper_{pid}"
        paper_node_ids[pid] = nid
        citation = int(row["citation_count"]) if not pd.isna(row.get("citation_count")) else 0
        year = int(row["year"]) if not pd.isna(row.get("year")) else 0
        topic_id = int(row["topic_id"]) if not pd.isna(row.get("topic_id")) else -1
        G.add_node(
            nid,
            node_type="Paper",
            label=sanitise(str(row.get("title", pid)))[:80],
            paper_id=pid,
            year=year,
            citation_count=citation,
            topic_id=topic_id,
            topic_label=sanitise(str(row.get("topic_label", ""))),
        )
    print(f"  Added {len(paper_node_ids)} Paper nodes")

    # ── 5. Trial nodes ─────────────────────────────────────────────────────────
    trial_node_ids = {}
    for _, row in selected_trials.iterrows():
        tid = str(row["nct_id"])
        nid = f"trial_{tid}"
        trial_node_ids[tid] = nid
        topic_id = int(row["topic_id"]) if not pd.isna(row.get("topic_id")) else -1
        G.add_node(
            nid,
            node_type="Trial",
            label=sanitise(str(row.get("brief_title", tid)))[:80],
            nct_id=tid,
            overall_status=str(row.get("overall_status", "")),
            phases=str(row.get("phases", "")),
            topic_id=topic_id,
            topic_label=sanitise(str(row.get("topic_label", ""))),
        )
    print(f"  Added {len(trial_node_ids)} Trial nodes")

    # ── 6. LiteratureTheme nodes ───────────────────────────────────────────────
    for _, row in lit_topics.iterrows():
        tid = int(row.get("topic_id", row.name))
        nid = f"lit_theme_{tid}"
        G.add_node(
            nid,
            node_type="LiteratureTheme",
            label=sanitise(str(row.get("topic_label", f"LitTheme_{tid}"))),
            topic_id=tid,
            top_terms=sanitise(str(row.get("top_terms", ""))),
            n_papers=int(row.get("n_papers", 0)) if not pd.isna(row.get("n_papers")) else 0,
            source="literature_NMF",
        )
    print(f"  Added {len(lit_topics)} LiteratureTheme nodes")

    # ── 7. TrialTheme nodes ────────────────────────────────────────────────────
    for _, row in trial_topics.iterrows():
        tid = int(row.get("topic_id", row.name))
        nid = f"trial_theme_{tid}"
        G.add_node(
            nid,
            node_type="TrialTheme",
            label=sanitise(str(row.get("topic_label", f"TrialTheme_{tid}"))),
            topic_id=tid,
            top_terms=sanitise(str(row.get("top_terms", ""))),
            n_trials=int(row.get("n_trials", 0)) if not pd.isna(row.get("n_trials")) else 0,
            source="trial_NMF",
        )
    print(f"  Added {len(trial_topics)} TrialTheme nodes")

    # ── 8. Target → Paper edges ────────────────────────────────────────────────
    # Only for targets with literature_mentions > 0
    lit_targets = gap[gap["literature_mentions"] > 0].copy()
    target_paper_edges = 0
    for _, trow in lit_targets.iterrows():
        symbol = trow["approved_symbol"]
        target_nid = f"target_{symbol}"
        for pid, counter in paper_counters.items():
            if symbol_in_doc(symbol, counter) and pid in paper_node_ids:
                G.add_edge(
                    target_nid, paper_node_ids[pid],
                    edge_type="target_mentioned_in_paper",
                    target_symbol=str(symbol),
                    paper_id=str(pid),
                )
                target_paper_edges += 1
    print(f"  Added {target_paper_edges} target_mentioned_in_paper edges")

    # ── 9. Target → Trial edges ────────────────────────────────────────────────
    # Only for targets with trial_mentions > 0
    trial_targets = gap[gap["trial_mentions"] > 0].copy()
    target_trial_edges = 0
    for _, trow in trial_targets.iterrows():
        symbol = trow["approved_symbol"]
        target_nid = f"target_{symbol}"
        for tid, counter in trial_counters.items():
            if symbol_in_doc(symbol, counter) and tid in trial_node_ids:
                G.add_edge(
                    target_nid, trial_node_ids[tid],
                    edge_type="target_mentioned_in_trial",
                    target_symbol=str(symbol),
                    nct_id=str(tid),
                )
                target_trial_edges += 1
    print(f"  Added {target_trial_edges} target_mentioned_in_trial edges")

    # ── 10. Paper → LiteratureTheme edges ─────────────────────────────────────
    paper_theme_edges = 0
    for _, row in selected_papers.iterrows():
        pid = str(row["paper_id"])
        topic_id = row.get("topic_id")
        if pd.isna(topic_id):
            continue
        theme_nid = f"lit_theme_{int(topic_id)}"
        if pid in paper_node_ids and theme_nid in G:
            G.add_edge(
                paper_node_ids[pid], theme_nid,
                edge_type="paper_has_literature_theme",
                topic_id=int(topic_id),
                topic_score=float(row.get("topic_score", 0.0)),
            )
            paper_theme_edges += 1
    print(f"  Added {paper_theme_edges} paper_has_literature_theme edges")

    # ── 11. Trial → TrialTheme edges ──────────────────────────────────────────
    trial_theme_edges = 0
    for _, row in selected_trials.iterrows():
        tid = str(row["nct_id"])
        topic_id = row.get("topic_id")
        if pd.isna(topic_id):
            continue
        theme_nid = f"trial_theme_{int(topic_id)}"
        if tid in trial_node_ids and theme_nid in G:
            G.add_edge(
                trial_node_ids[tid], theme_nid,
                edge_type="trial_has_theme",
                topic_id=int(topic_id),
                topic_score=float(row.get("topic_score", 0.0)),
            )
            trial_theme_edges += 1
    print(f"  Added {trial_theme_edges} trial_has_theme edges")

    # ── 12. Optional theme_overlap edges ──────────────────────────────────────
    # Link LiteratureTheme and TrialTheme that share the same first keyword token
    lit_theme_nodes = {
        nid: data["label"]
        for nid, data in G.nodes(data=True)
        if data.get("node_type") == "LiteratureTheme"
    }
    trial_theme_nodes = {
        nid: data["label"]
        for nid, data in G.nodes(data=True)
        if data.get("node_type") == "TrialTheme"
    }

    theme_overlap_edges = 0
    for lit_nid, lit_label in lit_theme_nodes.items():
        for trial_nid, trial_label in trial_theme_nodes.items():
            lit_first = lit_label.split("/")[0].split()[0].lower()
            trial_first = trial_label.split("/")[0].split()[0].lower()
            if lit_first == trial_first and len(lit_first) > 3:
                G.add_edge(
                    lit_nid, trial_nid,
                    edge_type="theme_overlap",
                    lit_label=lit_label,
                    trial_label=trial_label,
                )
                theme_overlap_edges += 1
    print(f"  Added {theme_overlap_edges} theme_overlap edges")

    return G


# ── Export ─────────────────────────────────────────────────────────────────────
def export_graph(G):
    print("\nExporting graph ...")

    nx.write_graphml(G, str(GRAPHS / "neurograph_knowledge_graph.graphml"))
    print("  Saved: neurograph_knowledge_graph.graphml")

    nx.write_gexf(G, str(GRAPHS / "neurograph_knowledge_graph.gexf"))
    print("  Saved: neurograph_knowledge_graph.gexf")

    data = nx.node_link_data(G)
    with open(GRAPHS / "neurograph_knowledge_graph.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("  Saved: neurograph_knowledge_graph.json")

    node_rows = [{"node_id": nid, **attrs} for nid, attrs in G.nodes(data=True)]
    pd.DataFrame(node_rows).to_csv(DATA / "knowledge_graph_nodes.csv", index=False, encoding="utf-8")
    print(f"  Saved: Data/processed/knowledge_graph_nodes.csv ({len(node_rows)} nodes)")

    edge_rows = [{"source": s, "target": t, **attrs} for s, t, attrs in G.edges(data=True)]
    pd.DataFrame(edge_rows).to_csv(DATA / "knowledge_graph_edges.csv", index=False, encoding="utf-8")
    print(f"  Saved: Data/processed/knowledge_graph_edges.csv ({len(edge_rows)} edges)")


# ── Figures ────────────────────────────────────────────────────────────────────
def plot_node_type_counts(G):
    """Figure 1: Node type distribution."""
    type_counts = Counter(d["node_type"] for _, d in G.nodes(data=True))
    ordered_types = ["Disease", "Target", "Paper", "Trial", "LiteratureTheme", "TrialTheme"]
    types = [t for t in ordered_types if t in type_counts]
    counts = [type_counts[t] for t in types]

    colour_map = {
        "Disease":          "#9467bd",
        "Target":           "#1f77b4",
        "Paper":            "#2ca02c",
        "Trial":            "#ff7f0e",
        "LiteratureTheme":  "#d62728",
        "TrialTheme":       "#8c564b",
    }
    bar_colours = [colour_map.get(t, "#7f7f7f") for t in types]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(types, counts, color=bar_colours, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, labels=counts, padding=3, fontsize=10)
    ax.set_title("NeuroGraph: Node Type Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Node Type")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "kg_01_node_type_counts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: kg_01_node_type_counts.png")


def plot_edge_type_counts(G):
    """Figure 2: Edge type distribution."""
    type_counts = Counter(d["edge_type"] for _, _, d in G.edges(data=True))
    types = list(type_counts.keys())
    counts = [type_counts[t] for t in types]

    # Sort descending
    pairs = sorted(zip(types, counts), key=lambda x: -x[1])
    types, counts = zip(*pairs) if pairs else ([], [])

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(list(types), list(counts), color="#5c9bd6", edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, labels=list(counts), padding=3, fontsize=10)
    ax.set_title("NeuroGraph: Edge Type Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_xlim(0, max(counts) * 1.2)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES / "kg_02_edge_type_counts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: kg_02_edge_type_counts.png")


def plot_target_gap_subgraph(G, gap):
    """Figure 3: Hub-spoke — Disease + top 10 strong gap candidates + evidence nodes.

    Improvements:
    - Top 10 candidates (was 15).
    - Target node size scales with gap_score.
    - Disease-target edges darker/thicker; target-evidence edges lighter/thinner.
    - Evidence nodes placed near their parent target (not a random outer ring).
    - Evidence nodes labelled 'Paper evidence' / 'Trial evidence'.
    - Subtitle: 'Edges represent computational evidence links, not clinical proof.'
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
    INNER_R = 1.85
    OUTER_R = 1.20

    pos = {DISEASE_NODE_ID: (0.0, 0.0)}
    n_t = max(len(target_nodes), 1)
    target_angles = {}

    for i, tn in enumerate(target_nodes):
        angle = np.pi / 2.0 - 2.0 * np.pi * i / n_t
        pos[tn] = (INNER_R * np.cos(angle), INNER_R * np.sin(angle))
        target_angles[tn] = angle

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

    # ── Node visual attributes ─────────────────────────────────────────────────
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

    # ── Edges ──────────────────────────────────────────────────────────────────
    disease_edges = [
        (u, v) for u, v, d in SG.edges(data=True)
        if d.get("edge_type") == "disease_target_evidence"
    ]
    evidence_edges = [
        (u, v) for u, v, d in SG.edges(data=True)
        if d.get("edge_type") in ("target_mentioned_in_paper",
                                   "target_mentioned_in_trial")
    ]
    nx.draw_networkx_edges(
        SG, pos, edgelist=disease_edges, ax=ax,
        edge_color="#3a3a7a", alpha=0.62, arrows=True,
        arrowsize=14, width=1.6,
        connectionstyle="arc3,rad=0.06",
    )
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
    nx.draw_networkx_labels(
        SG, pos,
        labels={DISEASE_NODE_ID: "Alzheimer's\nDisease"},
        ax=ax, font_size=8.5, font_weight="bold", font_color="white",
    )
    target_labels = {
        tn: str(SG.nodes[tn].get("label", tn.replace("target_", "")))
        for tn in target_nodes if tn in SG
    }
    nx.draw_networkx_labels(
        SG, pos, labels=target_labels,
        ax=ax, font_size=8, font_weight="bold", font_color="#111111",
    )
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
    plt.close()
    print("  Saved: kg_03_target_gap_subgraph.png + _improved.png")


def plot_theme_subgraph(G):
    """Figure 4: LiteratureTheme + TrialTheme nodes and theme_overlap edges.

    Improvements:
    - Duplicate labels disambiguated with (T{id}).
    - Nodes sorted on each side to minimise edge crossings.
    - Edge width proportional to average document count of connected themes.
    - Column headers 'Literature Themes' / 'Trial Themes' (one per side).
    - Node size proportional to document count.
    - Subtitle: 'Theme overlap based on shared top terms.'
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
        return 1.5 + ((w - min_w) / w_rng) * 2.5

    # ── Sort nodes to minimise crossings ──────────────────────────────────────
    trial_sorted = sorted(
        trial_nodes,
        key=lambda n: float(SG.nodes[n].get("topic_id", 99) or 99),
    )
    trial_idx = {n: i for i, n in enumerate(trial_sorted)}

    lit_adj = {n: [] for n in lit_nodes}
    for u, v in overlap_edges:
        if u in lit_adj:
            lit_adj[u].append(v)

    def _lit_sort_key(n):
        conn = lit_adj.get(n, [])
        return float(np.mean([trial_idx.get(c, 99) for c in conn])) if conn else 99.0

    lit_sorted = sorted(lit_nodes, key=_lit_sort_key)

    # ── Layout ─────────────────────────────────────────────────────────────────
    X_LIT, X_TRIAL, V_STEP = -2.2, 2.2, 1.15
    pos = {}
    n_l, n_r = len(lit_sorted), len(trial_sorted)
    for i, n in enumerate(lit_sorted):
        pos[n] = (X_LIT,   ((n_l - 1) / 2.0 - i) * V_STEP)
    for i, n in enumerate(trial_sorted):
        pos[n] = (X_TRIAL, ((n_r - 1) / 2.0 - i) * V_STEP)

    # ── Node sizes ─────────────────────────────────────────────────────────────
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

    wrapped = {
        n: "\n".join(textwrap.wrap(lbl, width=20))
        for n, lbl in all_labels.items()
    }
    nx.draw_networkx_labels(
        SG, pos, labels=wrapped,
        ax=ax, font_size=7.5, font_color="#111111", font_weight="bold",
    )

    y_top = max(p[1] for p in pos.values())
    hdr_y = y_top + V_STEP * 0.88
    ax.text(X_LIT,   hdr_y, "Literature Themes",
            ha="center", va="bottom", fontsize=11,
            fontweight="bold", color="#c94040")
    ax.text(X_TRIAL, hdr_y, "Trial Themes",
            ha="center", va="bottom", fontsize=11,
            fontweight="bold", color="#7a4030")
    ax.axvline(x=0, ymin=0.05, ymax=0.93, color="#cccccc",
               linewidth=0.8, linestyle="--", alpha=0.6)

    patches = [
        mpatches.Patch(color="#c94040", label="Literature Theme (NMF)"),
        mpatches.Patch(color="#7a4030", label="Trial Theme (NMF)"),
        mpatches.Patch(color="#606060", label="Theme overlap; edge width = document count"),
    ]
    ax.legend(handles=patches, loc="lower center", fontsize=8.5, ncol=3,
              framealpha=0.92, bbox_to_anchor=(0.5, -0.05))
    ax.text(0.01, 0.02, "Node size proportional\nto document count",
            transform=ax.transAxes, fontsize=6.5, color="#666666", style="italic")

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
    plt.close()
    print("  Saved: kg_04_theme_subgraph.png + _improved.png")


# ── Summary report ─────────────────────────────────────────────────────────────
def write_report(G, gap):
    print("\nWriting knowledge_graph_summary.md ...")

    node_type_counts = Counter(d["node_type"] for _, d in G.nodes(data=True))
    edge_type_counts = Counter(d["edge_type"] for _, _, d in G.edges(data=True))
    tier_counts = Counter(
        d["gap_interpretation_tier"]
        for _, d in G.nodes(data=True)
        if d.get("node_type") == "Target"
    )

    strong_targets = sorted(
        [
            (d["label"], d["gap_score"], d["literature_mentions"], d["trial_mentions"])
            for _, d in G.nodes(data=True)
            if d.get("node_type") == "Target"
            and d.get("gap_interpretation_tier") == "Strong text-supported gap signal"
        ],
        key=lambda x: -x[1],
    )

    lines = [
        "# NeuroGraph Knowledge Graph — Summary Report",
        "",
        "> **Hypothesis-generating only. Not clinical decision support.**",
        "> All findings represent associations in the literature and trial corpora analysed.",
        "> Results should not inform clinical practice or treatment decisions.",
        "",
        "---",
        "",
        "## 1. Overview",
        "",
        f"- **Total nodes:** {G.number_of_nodes()}",
        f"- **Total edges:** {G.number_of_edges()}",
        f"- **Graph type:** Directed (DiGraph)",
        f"- **NetworkX version:** {nx.__version__}",
        "",
        "## 2. Node Type Counts",
        "",
        "| Node Type | Count |",
        "|-----------|-------|",
    ]
    for nt in ["Disease", "Target", "Paper", "Trial", "LiteratureTheme", "TrialTheme"]:
        lines.append(f"| {nt} | {node_type_counts.get(nt, 0)} |")

    lines += [
        "",
        "## 3. Edge Type Counts",
        "",
        "| Edge Type | Count |",
        "|-----------|-------|",
    ]
    for et, cnt in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {et} | {cnt} |")

    lines += [
        "",
        "## 4. Target Gap Interpretation Tiers",
        "",
        "| Tier | Count |",
        "|------|-------|",
    ]
    for tier, cnt in sorted(tier_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {tier} | {cnt} |")

    lines += [
        "",
        "## 5. Top Strong Text-Supported Gap Targets",
        "",
        "These targets have literature mentions but low or absent trial mentions,",
        "suggesting they may appear underrepresented in clinical research relative to their",
        "scientific evidence base. This is hypothesis-generating only.",
        "",
        "| Symbol | Gap Score | Lit Mentions | Trial Mentions |",
        "|--------|-----------|--------------|----------------|",
    ]
    for sym, gs, lit, trial in strong_targets:
        lines.append(f"| {sym} | {gs:.4f} | {lit} | {trial} |")

    lines += [
        "",
        "## 6. Graph File Outputs",
        "",
        "- `outputs/graphs/neurograph_knowledge_graph.graphml` — GraphML (Gephi/Cytoscape)",
        "- `outputs/graphs/neurograph_knowledge_graph.gexf`    — GEXF (Gephi native)",
        "- `outputs/graphs/neurograph_knowledge_graph.json`    — JSON node-link (D3.js)",
        "- `Data/processed/knowledge_graph_nodes.csv`          — Node attribute table",
        "- `Data/processed/knowledge_graph_edges.csv`          — Edge attribute table",
        "",
        "## 7. Figure Outputs",
        "",
        "- `outputs/figures/kg_01_node_type_counts.png`  — Node type distribution",
        "- `outputs/figures/kg_02_edge_type_counts.png`  — Edge type distribution",
        "- `outputs/figures/kg_03_target_gap_subgraph.png` — Hub-spoke: Disease + top 15 strong gap candidates",
        "- `outputs/figures/kg_04_theme_subgraph.png`    — Literature vs Trial NMF theme subgraph",
        "",
        "## 8. Design Notes",
        "",
        "- **Primary node-colouring attribute:** `gap_interpretation_tier` (not `gap_category`)",
        "- **Paper selection:** Top 150 papers by citation count from 478 literature papers",
        "- **Trial selection:** Top 150 trials prioritising COMPLETED and RECRUITING status",
        "- **Target-to-document edges:** Created only when whole-word symbol match is",
        "  confirmed by re-scanning document text (same tokeniser as Step 5.5)",
        "- **Unicode safety:** Greek characters (beta, alpha, tau, etc.) are sanitised",
        "  to ASCII in all graph attributes for GraphML/GEXF export compatibility",
        "- **theme_overlap edges:** Optional links between LiteratureTheme and TrialTheme",
        "  nodes that share the same first keyword in their NMF topic label",
        "",
        "## 9. Limitations",
        "",
        "- The corpus covers 478 literature papers and 1,000 clinical trials; results",
        "  may not generalise to the full published literature",
        "- 471/499 targets (94.4%) have zero text mentions in this corpus;",
        "  absence of mention does not imply absence of scientific interest",
        "- Theme labels are NMF-derived and may not align with standard medical ontologies",
        "- All edges reflect text co-occurrence or database associations, not causal",
        "  or clinical relationships",
        "",
        "---",
        "*Generated by NeuroGraph Agent — Step 7: Knowledge Graph Construction*",
        "*Hypothesis-generating only. Not clinical decision support.*",
    ]

    with open(REPORTS / "knowledge_graph_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  Saved: knowledge_graph_summary.md")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("NeuroGraph Agent — Step 7: Knowledge Graph Construction")
    print("=" * 60)

    (gap, lit_assign, trial_assign,
     lit_topics, trial_topics,
     lit_clean, trial_clean) = load_data()

    print("\nSelecting nodes ...")
    selected_papers = select_papers(lit_assign, lit_clean)
    selected_trials = select_trials(trial_assign, trial_clean)

    print("\nBuilding document counters ...")
    paper_counters, trial_counters = build_doc_counters(
        lit_clean, trial_clean, selected_papers, selected_trials
    )

    G = build_graph(
        gap, lit_assign, trial_assign, lit_topics, trial_topics,
        lit_clean, trial_clean,
        selected_papers, selected_trials,
        paper_counters, trial_counters,
    )

    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    export_graph(G)

    print("\nGenerating figures ...")
    plot_node_type_counts(G)
    plot_edge_type_counts(G)
    plot_target_gap_subgraph(G, gap)
    plot_theme_subgraph(G)

    write_report(G, gap)

    print("\n" + "=" * 60)
    print("Step 7 complete.")
    print("  Graphs:  outputs/graphs/")
    print("  Figures: outputs/figures/ (kg_01 to kg_04)")
    print("  Report:  outputs/reports/knowledge_graph_summary.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
