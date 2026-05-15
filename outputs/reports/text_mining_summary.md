# NeuroGraph Agent — Text Mining and Theme Extraction Summary

*Generated: 2026-05-12 | Step 5 of NeuroGraph pipeline*

---

## 1. Purpose

This step applies computational text mining to extract research themes from
Alzheimer's disease literature and clinical trial records collected in earlier
pipeline steps. The methods used here — TF-IDF and NMF topic modelling — are
lexical and statistical in nature. All themes are provisional and should be
treated as hypothesis-generating signals for downstream analysis, not as
clinically validated findings.

---

## 2. Methods

### TF-IDF (Term Frequency–Inverse Document Frequency)
TF-IDF scores terms that are frequent within individual documents but rare across
the corpus, highlighting distinctive vocabulary. A custom stopword list removes
generic clinical trial and study language so that biomedical terms remain.
- Literature corpus: 478 records | TF-IDF matrix: 478 x 1000
- Trial corpus:      1,000 records | TF-IDF matrix: 1000 x 1000
- Settings: ngram_range=(1,2), min_df=2, max_df=0.85, max_features=1000

### NMF (Non-negative Matrix Factorisation) Topic Modelling
NMF decomposes the TF-IDF matrix into topics (latent themes) and document-topic
weights. Each document is assigned to its strongest topic. Topic labels are
assigned heuristically based on top terms and should be treated as provisional.
- Number of topics: 6
- Topic labels are assigned by rule-based keyword matching, not expert review.

### KMeans Clustering (comparison)
KMeans clustering was also run on normalised TF-IDF vectors as a supplementary
comparison (k=6). Results are saved to CSV but not visualised in this report.

### Target Mention Matching
The top 50 targets by Open Targets association score were matched by
whole-word regex against the combined literature and trial text corpora.
Mention counts are approximate: short gene symbols (e.g. APP, ACE) may match
common English words and inflate counts.

---

## 3. Literature Themes

The following provisional topics were identified in the literature corpus.
Top terms are listed in order of NMF component weight.

### Topic 0: Amyloid/Tau Pathology
- **Top terms:** amyloid, tau, aβ, role, pathogenesis, therapeutic, dysfunction, brain, mechanisms, neurodegenerative
- **Papers assigned:** 148

### Topic 1: Care, Lifestyle & QoL
- **Top terms:** care, dementia, health, dementias, caregivers, 65, age 65, payments, deaths, united
- **Papers assigned:** 33

### Topic 2: Amyloid/Tau Pathology
- **Top terms:** biomarkers, plasma, tau, csf, pet, blood, biomarker, amyloid, fluid, cerebrospinal
- **Papers assigned:** 114

### Topic 3: Cognitive & Neuropsychological
- **Top terms:** learning, deep, deep learning, classification, diagnosis, machine, model, accuracy, machine learning, mri
- **Papers assigned:** 47

### Topic 4: Neuroinflammation & Immune
- **Top terms:** microglia, risk, human, gene, mouse, genes, genetic, microglial, expression, model
- **Papers assigned:** 97

### Topic 5: Drug/Therapeutic Interventions
- **Top terms:** drug, pipeline, drug development, development, drugs, needed, symptoms, behavioral, therapies, improve
- **Papers assigned:** 39

---

## 4. Clinical Trial Themes

The following provisional topics were identified in the clinical trial corpus.

### Topic 0: Care, Lifestyle & QoL
- **Top terms:** dementia, care, caregivers, health, quality, life, people, program, older, caregiver
- **Trials assigned:** 298

### Topic 1: Neuroinflammation & Immune
- **Top terms:** mild moderate, moderate, mild, safety, moderate purpose, long term, subjects mild, term, long, donepezil
- **Trials assigned:** 104

### Topic 2: Amyloid/Tau Pathology
- **Top terms:** pet, brain, imaging, amyloid, 18f, tau, emission, positron emission, positron, tomography
- **Trials assigned:** 219

### Topic 3: Neuroinflammation & Immune
- **Top terms:** cognitive, impairment, cognitive impairment, mild cognitive, mild, mci, impairment mci, early, memory, impairment mild
- **Trials assigned:** 182

### Topic 4: Neuroinflammation & Immune
- **Top terms:** safety tolerability, tolerability, pharmacokinetics, safety, subjects, healthy, dose, single, tolerability pharmacokinetics, doses
- **Trials assigned:** 86

### Topic 5: Other / Mixed
- **Top terms:** efficacy, safety, efficacy safety, safety efficacy, double blind, blind, double, controlled, blind controlled, evaluate
- **Trials assigned:** 111

---

## 5. Target Mention Signals

The table below shows the top 5 targets by literature mentions.
Note: these counts reflect raw text matching and may include false positives.

| Symbol | Approved Name | Assoc. Score | Lit Mentions | Trial Mentions | Gap Signal |
|--------|--------------|-------------|-------------|---------------|-----------|
| APOE | apolipoprotein E | 0.775 | 69 | 18 | 51 |
| TREM2 | triggering receptor expressed on myeloid | 0.572 | 56 | 0 | 56 |
| APP | amyloid beta precursor protein | 0.870 | 25 | 76 | -51 |
| ACHE | acetylcholinesterase (Yt blood group) | 0.629 | 7 | 2 | 5 |
| CD33 | CD33 molecule | 0.582 | 6 | 0 | 6 |

**Top 5 targets by literature-to-trial gap signal** (high literature presence, relatively low trial presence — potential research-to-clinic translation gap):

| Symbol | Lit Mentions | Trial Mentions | Gap Signal |
|--------|-------------|---------------|-----------|
| TREM2 | 56 | 0 | 56 |
| APOE | 69 | 18 | 51 |
| CD33 | 6 | 0 | 6 |
| ACHE | 7 | 2 | 5 |
| CLU | 2 | 0 | 2 |

---

## 6. Interpretation

Computationally, the dominant literature themes appear to cluster around amyloid/tau
pathology, neuroinflammation, and genetics/risk factors — reflecting the established
focus of published Alzheimer's research. Clinical trial topics tend to weight
more heavily toward cognitive/neuropsychological outcomes and therapeutic
interventions, which is consistent with the trial lifecycle focus on measurable
endpoints.

Targets with high literature presence and lower trial presence (positive gap signal)
may represent areas where preclinical or mechanistic research has outpaced clinical
translation. However, this signal alone is insufficient to conclude a true gap;
it must be combined with target evidence scores, trial status, and phase data in
Step 6.

---

## 7. Limitations

- TF-IDF is lexical: it cannot resolve synonyms, abbreviations, or biomedical
  semantic equivalences (e.g. 'BACE1' vs 'beta-secretase').
- NMF topic number (n=6) is a modelling choice. Different values may produce
  different topic boundaries; no single decomposition is objectively correct.
- Provisional topic labels are assigned by rule-based keyword matching, not
  expert clinical review. Labels should be validated by a domain expert.
- Gene symbol mention counting can produce false positives for short symbols
  (e.g. 'APP', 'ACE', 'IL') that coincide with common English words or
  abbreviations.
- Missing or very short abstracts reduce the discriminative power of TF-IDF
  for those records.
- Clinical trial text rarely names specific gene/protein targets directly;
  low trial mention counts for a gene do not necessarily indicate absence of
  target-relevant research.
- This analysis is hypothesis-generating only. It does not constitute clinical
  evidence and should not be interpreted as such.

---

## 8. Next Step

**Step 6: Research Gap Scoring**

The next step will build a structured research gap scoring table that combines:
- Open Targets association scores (from target_evidence_clean.csv)
- Literature mention counts and topic coverage (from this step)
- Clinical trial mention counts, trial status, and phase (from this step + trials data)
- A composite gap score to rank targets by the strength of preclinical evidence
  relative to clinical trial investment

This will produce a ranked list of candidate targets for knowledge graph
prioritisation and visual exploration in subsequent steps.

---

*All outputs are computationally derived. No clinical conclusions should be drawn
from this report without expert domain review.*