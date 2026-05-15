<<<<<<< HEAD
# NeuroGraph Agent
### Agentic Biomedical Research Intelligence for Alzheimer's Disease

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![NetworkX](https://img.shields.io/badge/NetworkX-Knowledge_Graph-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-NMF%20%7C%20TF--IDF-orange)
![pandas](https://img.shields.io/badge/pandas-Data_Processing-150458)
![Status](https://img.shields.io/badge/Status-Prototype-orange)
![Domain](https://img.shields.io/badge/Domain-Biomedical_AI-purple)
![Type](https://img.shields.io/badge/Type-Research_Intelligence-blueviolet)
[![CI](https://github.com/Ella-Afonso/NeuroGraph-Agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Daniella-Afonso/NeuroGraph-Agent/actions/workflows/ci.yml)

> **NeuroGraph Agent** is a research intelligence prototype that combines public biomedical data, literature mining, clinical trial analysis, target-disease evidence integration, research gap scoring, knowledge graph construction, and evidence cards to explore hypothesis-generating Alzheimer's disease research signals.

> ⚠️ **Disclaimer:** This is a research intelligence prototype. All outputs are computationally derived and hypothesis-generating only. This is **not clinical decision support** and must not inform clinical practice, drug development, or patient care. All findings require expert biomedical validation.

---

## Live Demo

**[Launch the Streamlit Dashboard →](PASTE_STREAMLIT_LINK_HERE)**

Explore the dashboard to:
- Browse and filter 499 scored Alzheimer's disease targets
- Identify potential research gap candidates such as TREM2 and CD33
- Compare literature and clinical trial text signals side by side
- Inspect individual target evidence cards with full provenance
- Explore the knowledge graph and theme landscape
- Download processed data, reports, and evidence briefs

---

## Screenshots

### Dashboard Overview
![Dashboard Overview](assets/screenshots/dashboard_overview.png)

### Gap Candidates
![Gap Candidates](assets/screenshots/gap_candidates.png)

### Target Explorer — Evidence Card
![Target Explorer](assets/screenshots/target_explorer.png)

### Evidence Cards Page
![Evidence Cards](assets/screenshots/evidence_cards.png)

### Knowledge Graph Explorer
![Knowledge Graph](assets/screenshots/knowledge_graph.png)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Key Features](#3-key-features)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Data Sources](#5-data-sources)
6. [Methodology](#6-methodology)
7. [Results Summary](#7-results-summary)
8. [Dashboard Pages](#8-dashboard-pages)
9. [Output Files](#9-output-files)
10. [Technology Stack](#10-technology-stack)
11. [CI & Testing](#11-ci--testing)
12. [Ethical Disclaimer](#12-ethical-disclaimer)
13. [About](#13-about)

---

## 1. Overview

NeuroGraph Agent is a multi-step research intelligence pipeline built to address a practical challenge in biomedical research: integrating fragmented evidence sources into a structured, navigable layer of insight.

The system ingests data from three public biomedical APIs — **Open Targets** for disease-target association evidence, **Semantic Scholar** for literature metadata, and **ClinicalTrials.gov** for trial registrations — and applies a sequence of text mining, statistical scoring, and graph construction methods to produce structured research signals across **499 Alzheimer's disease-associated targets**.

What makes NeuroGraph Agent different from a standard data analysis project is its end-to-end integration:

- Raw API data is cleaned, validated, and enriched through a reproducible 10-step pipeline
- NMF topic models extract thematic structure from both literature and trial corpora independently
- All 499 targets are matched against paper and trial text using whole-word regex tokenisation
- Research gap scores are computed as a weighted composite of five evidence signals
- A NetworkX knowledge graph connects disease, targets, papers, trials, and research themes
- Each target receives an individual evidence card — a structured, cautious research brief generated purely from pipeline outputs, with no LLM involvement
- A 7-page Streamlit dashboard makes all outputs interactive, filterable, and downloadable

Alzheimer's disease was selected as the first domain because it is one of the most studied and underfunded neurodegenerative conditions globally, with a large and growing target landscape in Open Targets, and a rich body of publicly available literature and trial data.

---

## 2. Problem Statement

Biomedical researchers face a large and fragmented evidence landscape:

- Clinical trial registrations are stored separately from the literature that motivates them
- Target-disease association evidence from curated databases is disconnected from trial activity
- Literature volume is too large to review comprehensively without computational assistance
- Trial registrations often describe interventions by drug name or mechanism, not by molecular target symbol — making it difficult to assess target-level trial coverage directly
- Identifying where evidence is accumulating but translation may be limited requires cross-source signal integration

NeuroGraph Agent is a prototype that structures this evidence into a navigable intelligence layer, combining source integration, text mining, scoring, graph construction, and interactive visualisation into a single end-to-end system.

It does not replace expert review. It is designed to surface structured hypothesis-generating signals that can direct expert attention more efficiently.

---

## 3. Key Features

| Feature | Description |
|---|---|
| **Multi-source API integration** | Open Targets, Semantic Scholar, ClinicalTrials.gov |
| **Literature mining** | TF-IDF vectorisation + NMF topic extraction (6 topics) |
| **Clinical trial analysis** | TF-IDF + NMF topic extraction (6 topics) on trial protocol text |
| **Full target mention matching** | Whole-word regex matching across all 499 targets and all 478 papers + 1,000 trials |
| **Research gap scoring** | Weighted composite of 5 normalised signals per target |
| **Knowledge graph** | NetworkX DiGraph: 812 nodes, 829 edges, 6 node types, 6 edge types |
| **Evidence cards** | 499 individual target research briefs, generated programmatically |
| **Streamlit dashboard** | 7-page interactive dashboard with filters, charts, and download buttons |
| **Downloadable outputs** | CSVs, GraphML, GEXF, JSON, markdown reports |
| **Ethical guardrails** | Cautious language, no clinical claims, explicit disclaimers throughout |

---

## 4. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NeuroGraph Agent Pipeline                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1 │ Project Setup & Configuration                             │
│                                                                     │
│  Step 2 │ API Data Collection                                       │
│         │  ├── Open Targets  →  499 Alzheimer's disease targets     │
│         │  ├── Semantic Scholar  →  478 literature records          │
│         │  └── ClinicalTrials.gov  →  1,000 trial registrations     │
│                                                                     │
│  Step 3 │ Data Cleaning & Validation                                │
│         │  ├── Literature: text fields, deduplication, year filter  │
│         │  ├── Trials: status, phase, protocol text assembly        │
│         │  └── Targets: association scores, symbol validation       │
│                                                                     │
│  Step 4 │ Exploratory Data Analysis                                 │
│         │  └── 42 output figures, EDA summary report                │
│                                                                     │
│  Step 5 │ Text Mining: TF-IDF + NMF Topic Extraction               │
│         │  ├── Literature corpus  →  6 NMF topics                   │
│         │  └── Trial corpus  →  6 NMF topics                        │
│                                                                     │
│  Step 5.5│ Full Target Mention Matching                             │
│         │  └── Whole-word regex across all 499 × 1,478 records      │
│                                                                     │
│  Step 6 │ Research Gap Scoring                                      │
│         │  └── 5-signal composite score, tier classification        │
│                                                                     │
│  Step 7 │ Knowledge Graph Construction                              │
│         │  └── NetworkX DiGraph: 812 nodes, 829 edges               │
│                                                                     │
│  Step 8 │ Streamlit Dashboard                                       │
│         │  └── 7-page interactive dashboard                         │
│                                                                     │
│  Step 9 │ Evidence Card Generation                                  │
│         │  └── 499 individual target research briefs                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Sources

| Source | What Was Collected | Volume |
|---|---|---|
| **Open Targets** | Disease-target association scores, gene symbols, approved names, biotypes, datasource scores for Alzheimer's disease (MONDO:0004975) | 499 targets |
| **Semantic Scholar** | Paper title, abstract, year, citation count, venue, fields of study | 478 papers |
| **ClinicalTrials.gov** | Trial title, brief summary, study status, phase, conditions, interventions, eligibility criteria, outcomes | 1,000 registrations |

All data was retrieved via public APIs. No proprietary data was used. Raw files are stored locally and not redistributed.

---

## 6. Methodology

### 6.1 Text Mining: TF-IDF + NMF

TF-IDF (Term Frequency–Inverse Document Frequency) was applied independently to the literature corpus (title + abstract) and to the trial corpus (protocol text). Non-negative Matrix Factorisation (NMF) with `k = 6` topics was applied to each TF-IDF matrix, yielding statistically derived thematic groupings. Topic labels were assigned manually based on top NMF terms.

**Literature topics identified:** Amyloid/Tau Pathology, Neuroinflammation & Immune, Care, Lifestyle & QoL, Cognitive & Neuropsychological, Drug/Therapeutic Interventions, and mixed overlapping themes.

**Trial topics identified:** Care, Lifestyle & QoL, Amyloid/Tau Pathology, Neuroinflammation & Immune (multiple clusters), and Other/Mixed.

### 6.2 Target Mention Matching

Each of the 499 gene symbols was matched against every paper abstract and every trial protocol text using whole-word regex tokenisation. This produced per-target counts of:
- `literature_mentions` — symbol occurrences across papers
- `trial_mentions` — symbol occurrences across trial text
- `literature_documents_mentioned` — distinct papers containing the symbol
- `trial_documents_mentioned` — distinct trials containing the symbol

Symbols with short or ambiguous names were flagged with a `symbol_ambiguity_flag` (low / moderate) to indicate elevated false-positive risk.

### 6.3 Research Gap Scoring

Each target received a composite `gap_score` (0–1) derived from five normalised signals:

| Signal Component | Description |
|---|---|
| Evidence strength | Normalised Open Targets association score |
| Literature signal | Normalised literature mention count |
| Trial signal | Normalised trial mention count (inverted — fewer = higher gap) |
| Underexploration | Literature-to-evidence coverage ratio |
| Lit-to-trial gap | Normalised difference between literature and trial mention counts |

A `conservative_gap_score` additionally penalises targets with a moderate or high ambiguity flag. Targets are assigned to one of four interpretation tiers:

| Tier | Count | Description |
|---|---|---|
| Potential research gap candidate | 19 | Literature signal present; comparatively low trial-text signal |
| Established or trial-covered target | 8 | Strong trial-text presence; useful as comparator |
| No direct text match in this corpus | 471 | Symbol not detected in retrieved corpus |
| Lower-priority signal | 1 | Weak evidence and mention signals |

### 6.4 Knowledge Graph Construction

A directed NetworkX graph was constructed with:

**Node types (812 total):**
- Disease (1): Alzheimer's disease
- Target (499): scored gene targets
- Paper (150): literature records with direct symbol matches
- Trial (150): trial registrations with direct symbol matches
- LiteratureTheme (6): NMF topic nodes for the literature corpus
- TrialTheme (6): NMF topic nodes for the trial corpus

**Edge types (829 total):**
- `disease_target_evidence` — disease → target via Open Targets association
- `target_mentioned_in_paper` — target → paper via whole-word symbol match
- `target_mentioned_in_trial` — target → trial via whole-word symbol match
- `paper_has_literature_theme` — paper → NMF literature topic
- `trial_has_theme` — trial → NMF trial topic
- `theme_overlap` — LiteratureTheme ↔ TrialTheme via shared top terms

Graph exported in GraphML, GEXF, and JSON formats for downstream analysis in tools such as Gephi or D3.js.

### 6.5 Evidence Card Generation

Each of the 499 targets received a structured, programmatically generated evidence card containing:
- Target identity (symbol, name, biotype, IDs)
- Evidence strength classification (from Open Targets association score)
- Literature signal classification (from mention counts)
- Trial text signal classification (from mention counts)
- Dominant research themes (from matched paper/trial topic assignments)
- A rule-based narrative explaining why the target may be of interest
- Matched paper and trial records with metadata
- Limitations and suggested next validation steps

Evidence cards are generated entirely from pipeline outputs, with no LLM involvement. All language is cautious and explicitly hypothesis-generating.

---

## 7. Results Summary

| Metric | Value |
|---|---|
| Targets scored | 499 |
| Potential research gap candidates | 19 |
| Established / trial-covered comparators | 8 |
| Papers analysed | 478 |
| Clinical trials analysed | 1,000 |
| NMF topics extracted | 6 literature + 6 trial |
| Knowledge graph nodes | 812 |
| Knowledge graph edges | 829 |
| Evidence cards generated | 499 |
| Output figures | 42 |
| Markdown reports | 8 |

### Top Gap Candidates (by gap score)

| Rank | Symbol | Approved Name | Gap Score | Literature Mentions | Trial Mentions |
|---|---|---|---|---|---|
| 1 | **TREM2** | Triggering receptor expressed on myeloid cells 2 | 0.748 | 56 | 0 |
| 6 | **CD33** | CD33 molecule | — | 6 | 0 |
| 8 | **CDK5** | Cyclin dependent kinase 5 | — | 2 | 0 |
| 9 | **ADAM10** | ADAM metallopeptidase domain 10 | — | 1 | 0 |
| 10 | **CSF1R** | Colony stimulating factor 1 receptor | — | 1 | 0 |

> All gap scores are hypothesis-generating signals, not therapeutic recommendations. Literature and trial mention counts reflect this corpus sample only.

### Established Comparator Targets

For context, well-studied targets such as **APOE** (70 literature mentions, 21 trial mentions) and **APP** (25 literature mentions, 76 trial mentions) serve as calibration references — demonstrating what a target with high evidence coverage looks like in this pipeline.

---

## 8. Dashboard Pages

| Page | Content |
|---|---|
| **Overview** | Key metrics, pipeline summary, top gap candidates, score distribution charts |
| **Gap Candidates** | Filterable table of all 499 scored targets; filter by interpretation, symbol ambiguity, gap score, and mention counts |
| **Target Explorer** | 4-tab view per target: Evidence Card, Matching Papers, Matching Trials, Research Brief |
| **Evidence Cards** | Browse all 499 evidence cards; filter by interpretation and evidence strength; signal profile chart |
| **Themes** | NMF topic sizes, top terms, paper/trial browsers for each literature and trial theme |
| **Knowledge Graph** | Graph metrics, static subgraph figures, interactive target edge explorer split by edge type |
| **Reports & Downloads** | Download all processed CSVs, graph files (GraphML / GEXF / JSON), and markdown reports |

---

## 9. Output Files

```
outputs/
├── evidence_cards/          # 499 individual markdown evidence card files
│   ├── TREM2_evidence_card.md
│   ├── APOE_evidence_card.md
│   └── ... (499 total)
├── figures/                 # 42 PNG output figures
│   ├── eda_01_*.png         # EDA figures
│   ├── gap_01_*.png         # Gap scoring figures
│   └── kg_01_*.png          # Knowledge graph figures
├── graphs/                  # Knowledge graph exports
│   ├── neurograph_knowledge_graph.graphml
│   ├── neurograph_knowledge_graph.gexf
│   └── neurograph_knowledge_graph.json
└── reports/                 # Markdown analysis reports
    ├── eda_summary.md
    ├── text_mining_summary.md
    ├── gap_scoring_summary.md
    ├── knowledge_graph_summary.md
    ├── evidence_card_generation_summary.md
    └── top_gap_candidate_briefs.md    # 100KB research brief pack

Data/processed/
├── research_gap_scores.csv             # Full 499-target scored table
├── strong_text_supported_gap_candidates.csv
├── target_text_mentions_full.csv       # Raw mention counts per target
├── target_evidence_cards.csv           # 499-row evidence card table
├── knowledge_graph_nodes.csv           # 812 nodes with attributes
├── knowledge_graph_edges.csv           # 829 edges with attributes
├── literature_topics.csv               # 6 NMF literature topics
├── clinical_trial_topics.csv           # 6 NMF trial topics
├── literature_topic_assignments.csv    # Per-paper topic scores
└── clinical_trial_topic_assignments.csv
```

---

## 10. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas, numpy |
| Text mining | scikit-learn (TF-IDF, NMF) |
| Knowledge graph | NetworkX |
| Visualisation | matplotlib |
| Dashboard | Streamlit |
| API access | requests |
| Graph export | NetworkX (GraphML, GEXF, JSON) |
| Environment | Standard Python virtual environment |

No proprietary libraries, paid APIs, or LLM calls are used anywhere in the pipeline. All processing is local, deterministic, and fully reproducible from the original API data.

---

## 11. CI & Testing

This project includes a lightweight GitHub Actions CI workflow at `.github/workflows/ci.yml` that runs automatically on every push and pull request to `main`.

**What the workflow validates:**

| Step | Check |
|---|---|
| File existence | 9 required files must be committed (processed CSVs, reports, `app.py`) |
| Import check | `pandas`, `numpy`, `scikit-learn`, `networkx`, `streamlit` must import without error |
| Syntax check | `app.py`, all `src/processing/*.py`, all `src/dashboard/*.py` compiled with `py_compile` |
| Project structure tests | 68 pytest tests covering CSV shape, required columns, gap score range, node types, edge types, and forbidden phrase guardrails in evidence cards |

This workflow does not call external biomedical APIs or rerun the full data pipeline. It is designed as a lightweight reproducibility and project-health check.

### How to Run Tests Locally

```bash
pip install -r requirements.txt
pytest tests/test_project_structure.py -v
python -m py_compile app.py
```

Running `pytest -v` will execute all 68 project structure tests. Tests complete in under 2 seconds and require no internet connection or API keys.

---

## 12. Ethical Disclaimer

NeuroGraph Agent is built with explicit ethical guardrails:

- **Not clinical decision support.** This system must not be used to inform clinical practice, patient care, or drug development decisions.
- **No therapeutic claims.** A high gap score does not mean a target is therapeutically valid, safe, or ready for development.
- **No absence claims.** Low trial-text signal does not prove that no relevant trials exist — trial registrations may describe interventions by drug name rather than molecular target symbol.
- **Incomplete corpus.** The literature corpus covers 478 papers; the trial corpus covers 1,000 registrations. These are representative samples, not exhaustive reviews.
- **Symbol ambiguity.** Gene symbol matching is lexical and may produce false positives for short or ambiguous symbols.
- **Expert review required.** All outputs require review by qualified biomedical researchers or clinicians before any finding can inform research decisions.
- **Hypothesis-generating only.** Every output in this system is a structured signal for expert-directed investigation, not a conclusion.

These guardrails are enforced in the evidence card text, dashboard language, and all generated reports.

---

## 13. About

NeuroGraph Agent was designed and built as an end-to-end research intelligence portfolio project demonstrating:

- Multi-source public API integration and data engineering
- NLP and unsupervised learning applied to biomedical text
- Systematic scoring and ranking of targets across multiple evidence dimensions
- Knowledge graph construction and traversal
- Programmatic evidence card generation
- Interactive Streamlit dashboard development
- Responsible AI: cautious language, explicit limitations, and ethical guardrails throughout

The project was built in Python, without proprietary tools, paid APIs, or LLM assistance in evidence generation.

---

*NeuroGraph Agent — Research intelligence prototype. Not clinical decision support.*
=======
# NeuroGraph-Agent
An Agentic Research System for Neurodegenerative Disease Discovery
>>>>>>> 9fe67f9da0d18fb3fa95ce2bb21c2a98bc1120b0
