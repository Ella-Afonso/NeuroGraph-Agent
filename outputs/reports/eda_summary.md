# NeuroGraph Agent — Exploratory Data Analysis Summary

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
| Clinical Trials | 1,000 | 22 | ClinicalTrials.gov API v2 |
| Literature | 478 | 12 | Semantic Scholar API |
| Target Evidence | 500 | 21 | Open Targets GraphQL API |

---

## 3. Clinical Trial Landscape

### Status Distribution

Of 1,000 trials in the dataset:
- **539 (53.9%) are COMPLETED** — indicating a historically active research landscape
- **137 (13.7%) are RECRUITING** — representing the active trial pipeline
- **76 (7.6%) are TERMINATED** — this attrition rate may reflect challenges in Alzheimer's drug development, though termination reasons vary and require further investigation
- **126 trials have UNKNOWN status** — a data quality limitation worth noting in downstream analysis

### Phase Distribution

Phase information is not recorded for 583 of 1,000 trials (58.3%) — these are predominantly observational studies, registries, and expanded-access records. Among the 417 trials with a recorded interventional phase, Phase 2 is the most common (141 trials), followed by Phase 1 (104). The smaller Phase 3 count (83) compared to Phase 2 is consistent with the high attrition rate typically observed between phases in Alzheimer's drug development, though this dataset alone is insufficient to quantify that attrition.

### Sponsor Type

Academic institutions and non-profits represent the largest funder class (621 trials, 62.1%), with industry sponsors accounting for 324 trials (32.4%). This pattern suggests that early exploratory research may be largely publicly funded, while commercial sponsors could be more concentrated at later-stage trials — though this requires further validation against phase-stratified data.

### Geographic Distribution

The United States leads in trial count by a significant margin. The geographic spread indicates global research interest, but may also reflect ClinicalTrials.gov registration biases towards US-based research.

### Enrolment

Median trial enrolment is 84. The distribution is highly right-skewed — a small number of large Phase 3 trials drive the upper tail.

### Data Quality Issues

- `start_date` missing in 368 of 1,000 trials (36.8%)
- `completion_date` missing in 425 trials (42.5%)
- `phases` missing in 583 trials (58.3%) — predominantly observational studies

---

## 4. Literature Landscape

### Publication Year Trends

Papers in this dataset range from 2018 to 2025. Publication volume in this sample peaks at 2020 (97 records). Counts for recent years (2024–2025) are notably lower, which may reflect indexing lag in the Semantic Scholar API — recently published papers take time to be indexed and have had less time to accumulate citations. This should be treated as a retrieval artefact rather than a genuine decline in research activity.

### Citation Patterns

| Statistic | Value |
|---|---|
| Minimum | 1 |
| Maximum | 3,354 |
| Mean | 404.2 |
| Median | 266 |
| 25th Percentile | 174 |
| 75th Percentile | 443 |

The citation distribution is strongly right-skewed. The relatively high median (266) suggests this API sample may be skewed toward highly cited and prominent papers, which could reflect Semantic Scholar's retrieval or ranking behaviour rather than the full distribution of Alzheimer's research output. This dataset does not represent the complete Alzheimer's literature landscape and citation counts should be interpreted with that context in mind.

### Venue Patterns

`Alzheimer's & Dementia` is the most represented venue (61 papers), reflecting the field's dedicated journal infrastructure. High-impact general journals (Nature, Lancet Neurology, Nature Medicine) are also well-represented, indicating that Alzheimer's research reaches broad scientific audiences.

### Abstract Missingness

418 of 478 papers (87.4%) have an abstract. The 60 (12.6%) without abstracts will contribute reduced signal in downstream NLP analysis. These records can still contribute year, venue, and citation data to bibliometric analysis.

### Fields of Study

Medicine dominates the field classification, with a small number of papers classified under Biology, Computer Science, and Chemistry. This suggests the dataset is primarily clinical and translational rather than basic science — relevant to note for TF-IDF theme extraction.

---

## 5. Target Evidence Landscape

### Top Targets / Genes

The highest-scoring targets by Alzheimer's association evidence are:

| Rank | Gene | Full Name | Association Score |
|---|---|---|---|
| 1 | APP | amyloid beta precursor protein | 0.8697 |
| 2 | PSEN1 | presenilin 1 | 0.8663 |
| 3 | PSEN2 | presenilin 2 | 0.8165 |
| 4 | APOE | apolipoprotein E | 0.7749 |
| 5 | GRIN1 | glutamate ionotropic receptor NMDA type subunit 1 | 0.7000 |

APP, PSEN1, and PSEN2 are the canonical amyloid pathway genes, while APOE represents the strongest common genetic risk factor. Their top-ranking positions are consistent with known Alzheimer's research themes and suggest the Open Targets query is returning well-supported targets. However, a high association score indicates that a target is well-represented in available evidence — it does not imply therapeutic success or causal proof.

### Association Score Distribution

| Statistic | Value |
|---|---|
| Minimum | 0.3307 |
| Maximum | 0.8697 |
| Mean | 0.4261 |
| Median | 0.3999 |
| 75th Percentile | 0.4673 |
| 90th Percentile | 0.5278 |

The distribution is right-skewed with the majority of targets clustering near the lower bound. Only a small proportion of targets have strong multi-source evidence. This skew is expected — it indicates a core of targets that are well-supported by available evidence, and a long tail of lower-confidence candidates that may represent underexplored research opportunities requiring further validation.

### Datasource Evidence Patterns

The most prevalent evidence source is **Europe PMC** (84.8% of targets), indicating that most associations are literature-derived. **CRISPR Screen** data is the second most common (69.2%), reflecting growing use of genome-wide screens in neurodegeneration research. **Clinical Precedence** — which indicates an approved or investigational drug — is present for only 13.4% of targets, highlighting the large gap between genetic/literature evidence and clinical translatability.

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
- **Abstract missingness**: 12.6% of papers lack abstracts, reducing coverage for NLP-based theme extraction.
- **ClinicalTrials.gov scope**: Only registered trials appear in this dataset. Unregistered studies, observational research, and trials registered outside ClinicalTrials.gov are absent.
- **Open Targets truncation**: Only the top 500 associations by score were retrieved out of 13,142 available in Open Targets for this disease. Lower-scoring targets that may represent novel hypotheses are not included in this analysis.
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
