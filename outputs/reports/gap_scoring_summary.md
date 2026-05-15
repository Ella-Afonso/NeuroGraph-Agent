# NeuroGraph Agent — Research Gap Scoring Summary

*Generated: 2026-05-13 | Step 6 (revised) of NeuroGraph pipeline*

> **Important notice:** This report contains computationally derived research
> gap signals. All findings are hypothesis-generating only. No clinical claims
> are made. All candidates require expert validation.

---

## 1. Purpose

This step ranks all 499 Alzheimer's disease targets by a transparent,
multi-component research gap score. Step 5.5 performed full text-mention
matching across all 499 targets (not just the top 50). The scoring now uses
a **literature-aware gap component** so that targets with zero literature
signal cannot receive inflated underexploration scores.

---

## 2. Data Inputs and Coverage

| Source | Records |
|--------|---------|
| target_text_mentions_full.csv | 499 targets (all checked in Step 5.5) |
| Literature corpus | 478 papers |
| Trial corpus | 1,000 trials |

| Mention coverage | Count |
|-----------------|-------|
| Targets with literature mentions (> 0) | 26 |
| Targets with trial mentions (> 0) | 10 |
| Targets with zero mentions in both | 471 |

### Important: Zero-mention targets

> **471 targets had zero mentions in both corpora.** This means the
> approved gene symbol was not found as a standalone alphanumeric token in
> the retrieved documents. This is a **lexical absence**, not evidence of
> absence from science or clinical trials. The target may be referenced via
> protein name, drug name, or pathway, or may simply not appear in this
> specific corpus of 478 papers and 1,000 trials.
> These targets receive a `gap_interpretation_tier` of **'Checked-zero text signal'**,
> not 'evidence-only'. Their gap score is driven by Open Targets evidence alone.

---

## 3. Scoring Formula

### Primary Gap Score

```
gap_score = 0.40 x evidence_strength_norm
          + 0.30 x literature_signal_norm
          + 0.30 x literature_trial_gap_norm
```

| Component | Weight | Derivation |
|-----------|--------|-----------|
| evidence_strength_norm | 0.40 | MinMaxScaler(association_score) |
| literature_signal_norm | 0.30 | MinMaxScaler(log1p(literature_mentions)) |
| literature_trial_gap_norm | 0.30 | literature_signal_norm x (1 - trial_signal_norm) |

### Why literature_trial_gap_norm replaces underexploration_norm

The previous formula used `underexploration_norm = 1 - trial_signal_norm`, which
gave maximum underexploration credit to *any* target with zero trial mentions —
including targets with zero literature signal. This inflated gap scores for targets
that are simply absent from both corpora.

The new component `literature_trial_gap_norm = literature_signal_norm x (1 - trial_signal_norm)`
is **zero when literature_signal is zero**. A target only receives a positive gap
contribution when it has measurable literature presence *and* relatively low trial
text presence. Targets with zero literature signal get gap_score = 0.40 x evidence_strength_norm only.

### Conservative Gap Score

```
conservative_gap_score = 0.50 x evidence_strength_norm
                       + 0.20 x literature_signal_norm
                       + 0.30 x literature_trial_gap_norm
```

### Category Assignment
Categories use percentile-based thresholds (Percentile-based: top 10% = High-priority (>= 0.1634), next 20% = Moderate (>= 0.0983)).

| Category | Count | Threshold |
|----------|-------|-----------|
| High-priority gap signal | 50 | gap_score >= 0.1634 |
| Moderate gap signal | 100 | gap_score >= 0.0983 |
| Low gap signal | 349 | gap_score < 0.0983 |

### Interpretation Tiers

| Tier | Criteria | Count |
|------|----------|-------|
| Strong text-supported gap signal | lit > 0, trial <= 1 (90th pct), high gap_score, low/mod ambiguity | 19 |
| Established / trial-covered target | trial > 1 (90th pct) | 8 |
| Checked-zero text signal | lit = 0 AND trial = 0 | 471 |
| Ambiguous symbol; manual validation required | symbol_ambiguity_flag = high | 0 |
| Low-priority / weak gap signal | all other | — |

---

## 4. Top Gap Candidates

### 4a. All targets — top 15 by gap score

Note: many targets near the top of this list have zero text mentions.
Their ranking reflects evidence strength, not text-supported gap signals.
See Section 4b for the more defensible subset.

| Rank | Symbol | Name | Assoc. | Lit | Trial | Gap Score | Tier |
|------|--------|------|--------|-----|-------|-----------|------|
| 1 | TREM2 | triggering receptor expressed on... | 0.572 | 56 | 0 | 0.7483 | Strong text-supported gap si... |
| 2 | APOE | apolipoprotein E | 0.775 | 70 | 21 | 0.7162 | Established / trial-covered ... |
| 3 | APP | amyloid beta precursor protein | 0.870 | 25 | 76 | 0.6293 | Established / trial-covered ... |
| 4 | PSEN1 | presenilin 1 | 0.866 | 3 | 6 | 0.5489 | Established / trial-covered ... |
| 5 | ACHE | acetylcholinesterase (Yt blood g... | 0.629 | 7 | 2 | 0.4768 | Established / trial-covered ... |
| 6 | CD33 | CD33 molecule | 0.582 | 6 | 0 | 0.4603 | Strong text-supported gap si... |
| 7 | PSEN2 | presenilin 2 | 0.817 | 1 | 1 | 0.4503 | Strong text-supported gap si... |
| 8 | CDK5 | cyclin dependent kinase 5 | 0.680 | 2 | 0 | 0.4139 | Strong text-supported gap si... |
| 9 | ADAM10 | ADAM metallopeptidase domain 10 | 0.682 | 1 | 0 | 0.3580 | Strong text-supported gap si... |
| 10 | CLU | clusterin | 0.556 | 2 | 0 | 0.3216 | Strong text-supported gap si... |
| 11 | MTOR | mechanistic target of rapamycin ... | 0.494 | 3 | 0 | 0.3161 | Strong text-supported gap si... |
| 12 | SCD | stearoyl-CoA desaturase | 0.481 | 6 | 9 | 0.3126 | Established / trial-covered ... |
| 13 | ACE | angiotensin I converting enzyme | 0.634 | 1 | 4 | 0.3047 | Established / trial-covered ... |
| 14 | MAPT | microtubule associated protein t... | 0.377 | 5 | 0 | 0.2868 | Strong text-supported gap si... |
| 15 | GRIN1 | glutamate ionotropic receptor NM... | 0.700 | 0 | 0 | 0.2741 | Checked-zero text signal... |

### 4b. Strong text-supported gap candidates

These candidates have measurable literature presence, low trial text presence,
and high gap scores. They represent the most defensible subset of gap signals
in this framework.

| Rank | Symbol | Approved Name | Assoc. | Lit | Trial | Gap Score | Ambiguity |
|------|--------|--------------|--------|-----|-------|-----------|-----------|
| 1 | TREM2 | triggering receptor expressed on myelo | 0.572 | 56 | 0 | 0.7483 | low |
| 6 | CD33 | CD33 molecule | 0.582 | 6 | 0 | 0.4603 | low |
| 7 | PSEN2 | presenilin 2 | 0.817 | 1 | 1 | 0.4503 | low |
| 8 | CDK5 | cyclin dependent kinase 5 | 0.680 | 2 | 0 | 0.4139 | low |
| 9 | ADAM10 | ADAM metallopeptidase domain 10 | 0.682 | 1 | 0 | 0.3580 | low |
| 10 | CLU | clusterin | 0.556 | 2 | 0 | 0.3216 | moderate |
| 11 | MTOR | mechanistic target of rapamycin kinase | 0.494 | 3 | 0 | 0.3161 | low |
| 14 | MAPT | microtubule associated protein tau | 0.377 | 5 | 0 | 0.2868 | low |
| 17 | PLCG2 | phospholipase C gamma 2 | 0.551 | 1 | 0 | 0.2614 | low |
| 18 | MS4A6A | membrane spanning 4-domains A6A | 0.538 | 1 | 0 | 0.2517 | low |
| 20 | PICALM | phosphatidylinositol binding clathrin  | 0.519 | 1 | 0 | 0.2374 | low |
| 21 | ATP6V1A | ATPase H+ transporting V1 subunit A | 0.518 | 1 | 0 | 0.2368 | low |
| 22 | CSF1R | colony stimulating factor 1 receptor | 0.405 | 3 | 1 | 0.2347 | low |
| 23 | KAT8 | lysine acetyltransferase 8 | 0.506 | 1 | 0 | 0.2275 | low |
| 24 | BACE1 | beta-secretase 1 | 0.423 | 2 | 0 | 0.2230 | low |
| 32 | SHARPIN | SHANK associated RH domain interactor | 0.477 | 1 | 0 | 0.2058 | low |
| 36 | LRP1 | LDL receptor related protein 1 | 0.462 | 1 | 0 | 0.1949 | low |
| 37 | PTK2B | protein tyrosine kinase 2 beta | 0.461 | 1 | 0 | 0.1946 | low |
| 40 | NOX4 | NADPH oxidase 4 | 0.379 | 2 | 0 | 0.1907 | low |

### 4c. Established / trial-covered targets

These targets have high trial text presence and are less likely to represent
underexplored gaps based on this signal. Targets with moderate symbol ambiguity
in this list should be manually validated for false-positive trial mentions.

| Symbol | Approved Name | Lit | Trial | Ambiguity | Note |
|--------|--------------|-----|-------|-----------|------|
| APOE | apolipoprotein E | 70 | 21 | low | reliable |
| APP | amyloid beta precursor protein | 25 | 76 | moderate | review mentions |
| PSEN1 | presenilin 1 | 3 | 6 | low | reliable |
| ACHE | acetylcholinesterase (Yt blood | 7 | 2 | low | reliable |
| SCD | stearoyl-CoA desaturase | 6 | 9 | moderate | review mentions |
| ACE | angiotensin I converting enzym | 1 | 4 | moderate | review mentions |
| PDE4B | phosphodiesterase 4B | 0 | 2 | low | reliable |
| TSPO | translocator protein | 0 | 2 | low | reliable |

---

## 5. Score Distribution

Gap scores across all 499 targets:

- Mean:   0.0798
- Median: 0.0532
- Std:    0.0893
- Min:    0.0000
- Max:    0.7483

The distribution is strongly right-skewed / positively skewed: 471 targets
have zero literature mentions and therefore zero literature_trial_gap_norm,
clustering at lower scores, while a smaller set of 19 text-supported
candidates forms the upper tail.

---

## 6. Interpretation

### 6a. Strong text-supported gap candidates

The following 19 targets have literature text signal, low trial text presence (trial_mentions <= 1), and high gap scores. They *may* represent areas where preclinical research attention has not been matched by clinical trial activity, though this requires manual validation.

- **TREM2** (triggering receptor expressed on myeloid cells 2): lit=56, trial=0, assoc=0.572, gap=0.7483
- **CD33** (CD33 molecule): lit=6, trial=0, assoc=0.582, gap=0.4603
- **PSEN2** (presenilin 2): lit=1, trial=1, assoc=0.817, gap=0.4503
- **CDK5** (cyclin dependent kinase 5): lit=2, trial=0, assoc=0.680, gap=0.4139
- **ADAM10** (ADAM metallopeptidase domain 10): lit=1, trial=0, assoc=0.682, gap=0.3580
- **CLU** (clusterin): lit=2, trial=0, assoc=0.556, gap=0.3216 [symbol ambiguity = moderate; review mention counts]
- **MTOR** (mechanistic target of rapamycin kinase): lit=3, trial=0, assoc=0.494, gap=0.3161
- **MAPT** (microtubule associated protein tau): lit=5, trial=0, assoc=0.377, gap=0.2868
- **PLCG2** (phospholipase C gamma 2): lit=1, trial=0, assoc=0.551, gap=0.2614
- **MS4A6A** (membrane spanning 4-domains A6A): lit=1, trial=0, assoc=0.538, gap=0.2517
- **PICALM** (phosphatidylinositol binding clathrin assembly pro): lit=1, trial=0, assoc=0.519, gap=0.2374
- **ATP6V1A** (ATPase H+ transporting V1 subunit A): lit=1, trial=0, assoc=0.518, gap=0.2368
- **CSF1R** (colony stimulating factor 1 receptor): lit=3, trial=1, assoc=0.405, gap=0.2347
- **KAT8** (lysine acetyltransferase 8): lit=1, trial=0, assoc=0.506, gap=0.2275
- **BACE1** (beta-secretase 1): lit=2, trial=0, assoc=0.423, gap=0.2230
- **SHARPIN** (SHANK associated RH domain interactor): lit=1, trial=0, assoc=0.477, gap=0.2058
- **LRP1** (LDL receptor related protein 1): lit=1, trial=0, assoc=0.462, gap=0.1949
- **PTK2B** (protein tyrosine kinase 2 beta): lit=1, trial=0, assoc=0.461, gap=0.1946
- **NOX4** (NADPH oxidase 4): lit=2, trial=0, assoc=0.379, gap=0.1907

### 6b. Established / trial-covered targets

These targets should not be described as underexplored based on this corpus.
Note that APP, SCD, and ACE have moderate symbol ambiguity — their high trial
mention counts may include false positives (e.g. 'APP' matching 'application',
'SCD' matching 'sickle cell disease', 'ACE' matching 'ACE inhibitor' class).
Manual validation of these counts is recommended before citing them as evidence.

---

## 7. Limitations

- **471 targets had zero text mentions**: This is a lexical absence from this
  specific corpus, not evidence of absent research. The corpus (478 papers,
  1,000 trials) is a sample. These targets receive gap scores driven by
  evidence_strength_norm alone and should not be treated as confirmed gaps.

- **Gene symbol ambiguity**: Short 3-letter all-alpha symbols (APP, ACE, SCD,
  CLU) may match non-gene uses in clinical text. APP could match 'application',
  SCD could match 'sickle cell disease', ACE could match 'ACE inhibitor' as a
  drug class reference. Mention counts for these symbols should be manually
  verified before citing as evidence of gene-specific research activity.

- **Trial text does not name molecular targets**: A clinical trial targeting
  TREM2 via AL002 will not contain 'TREM2' in its registration text. Low trial
  mention counts do not imply absence of relevant trials.

- **Corpus sample**: 478 papers and 1,000 trials are a limited sample of the
  global literature and trial landscape.

- **Open Targets scores**: association_score summarises curation breadth, not
  therapeutic tractability. High scores do not imply valid drug targets.

- **No causal inference**: This framework is purely lexical and statistical.
  It is not clinical decision support.

---

## 8. Recommended Step 7

**Step 7: Knowledge Graph Construction**

```
Disease -> Target -> Paper -> Trial -> Theme
```

Inputs from Steps 5, 5.5, and 6:
- `research_gap_scores.csv` — gap score, tier, and ambiguity as node attributes
- `strong_text_supported_gap_candidates.csv` — prioritised gap candidates
- `literature_topic_assignments.csv` — paper-to-theme edges
- `clinical_trial_topic_assignments.csv` — trial-to-theme edges
- `target_evidence_clean.csv` — target-to-disease edges with evidence weights

**Node colouring:** Use `gap_interpretation_tier` as the primary node-colouring
attribute rather than `gap_category` alone. `gap_interpretation_tier` better
separates strong text-supported gap candidates (red) from established /
trial-covered targets (orange), checked-zero targets (grey), and low-priority
candidates (blue), making the knowledge graph immediately interpretable.

---

*All outputs are computationally derived. No clinical conclusions should be
drawn from this report without expert domain review.*