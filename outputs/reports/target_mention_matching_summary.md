# NeuroGraph Agent — Target Mention Matching Summary

*Generated: 2026-05-13 | Step 5.5 of NeuroGraph pipeline*

> All mention counts are derived from whole-word tokenisation of biomedical text.
> They are lexical signals only. Gene symbols may match non-target tokens,
> and targets may be referenced without their gene symbol. These counts
> are hypothesis-generating signals, not measures of research activity.

---

## 1. Coverage

All 499 unique targets in `target_evidence_clean.csv` were checked against
the full literature and clinical trial text corpora.

| Status | Count |
|--------|-------|
| Targets checked | 499 |
| With literature mentions (>0) | 26 |
| With trial mentions (>0) | 10 |
| Zero mentions in both corpora | 471 |
| not_checked | 0 |

**471 targets had zero mentions in both corpora.** This means the gene
symbol was not found as a standalone alphanumeric token in the retrieved
documents. It does not mean the target is absent from the broader literature
or from ongoing trials — it may be referenced via protein name, drug name,
pathway, or not present in this specific corpus sample.

---

## 2. Methods

Text was sourced from `paper_text` (literature) and `trial_text` (trials).
Each document was tokenised into lowercased alphanumeric sequences using
`[A-Za-z0-9]+`. For each target, the lowercased approved_symbol was
looked up in each document's token Counter. This approach:

- Handles hyphens and slashes as separators (TREM2-targeted → ['trem2', 'targeted'])
- Is case-insensitive via lowercasing
- Does NOT resolve synonyms, protein names, or compound symbols (APP-swe)
- Does NOT use approximate or semantic matching

---

## 3. Symbol Ambiguity

Gene symbols were classified by potential for false-positive matches:

| Ambiguity | Criteria | Count |
|-----------|----------|-------|
| low | has digit, or len >= 4 | 476 |
| moderate | len == 3, all alpha (e.g. APP, ACE) | 23 |
| high | len <= 2 | 0 |

Targets with moderate ambiguity are flagged in `mention_match_quality`
as `moderate_ambiguity_review`. Their mention counts may include matches
from non-gene uses of the same abbreviation in clinical text.

> In a focused biomedical corpus, most 3-letter gene symbols are used in
> their molecular biology context and rarely match unrelated abbreviations.
> However, manual validation is recommended for key findings from this tier.

---

## 4. Top Targets by Literature Mentions

| Symbol | Approved Name | Lit Mentions | Trial Mentions | Gap Signal | Ambiguity |
|--------|--------------|-------------|---------------|-----------|-----------|
| APOE | apolipoprotein E | 70 | 21 | 49 | low |
| TREM2 | triggering receptor expressed on myeloid | 56 | 0 | 56 | low |
| APP | amyloid beta precursor protein | 25 | 76 | -51 | moderate |
| ACHE | acetylcholinesterase (Yt blood group) | 7 | 2 | 5 | low |
| CD33 | CD33 molecule | 6 | 0 | 6 | low |
| SCD | stearoyl-CoA desaturase | 6 | 9 | -3 | moderate |
| MAPT | microtubule associated protein tau | 5 | 0 | 5 | low |
| PSEN1 | presenilin 1 | 3 | 6 | -3 | low |
| MTOR | mechanistic target of rapamycin kinase | 3 | 0 | 3 | low |
| CSF1R | colony stimulating factor 1 receptor | 3 | 1 | 2 | low |

---

## 5. Top Targets by Trial Mentions

| Symbol | Approved Name | Trial Mentions | Lit Mentions | Ambiguity |
|--------|--------------|---------------|-------------|-----------|
| APP | amyloid beta precursor protein | 76 | 25 | moderate |
| APOE | apolipoprotein E | 21 | 70 | low |
| SCD | stearoyl-CoA desaturase | 9 | 6 | moderate |
| PSEN1 | presenilin 1 | 6 | 3 | low |
| ACE | angiotensin I converting enzyme | 4 | 1 | moderate |
| ACHE | acetylcholinesterase (Yt blood group) | 2 | 7 | low |
| PDE4B | phosphodiesterase 4B | 2 | 0 | low |
| TSPO | translocator protein | 2 | 0 | low |
| PSEN2 | presenilin 2 | 1 | 1 | low |
| CSF1R | colony stimulating factor 1 receptor | 1 | 3 | low |

---

## 6. Limitations and Warnings

### Gene symbol ambiguity
Short gene symbols (especially 3-letter all-alpha, e.g. APP, ACE, CLU) may
match common biomedical abbreviations beyond the target gene. For example,
'ACE' appears in 'ACE inhibitor' which does refer to the ACE protein — however
some clinical text may use 'ACE' to refer to the drug class rather than the gene.
Targets with `mention_match_quality = moderate_ambiguity_review` should be
manually confirmed before citing counts as evidence of research activity.

### Trial text does not name molecular targets
Clinical trial registrations describe drugs, interventions, and patient criteria.
A trial targeting TREM2 via a monoclonal antibody (e.g. AL002) will not
contain the word 'TREM2' in its registration text. **Low or zero trial mention
counts are not evidence of underinvestigation** — they reflect the lexical
structure of trial registrations, not the true scope of target-related trials.

### Corpus size
The literature corpus (478 papers) and trial corpus (1,000 trials) are samples.
Absence from this corpus does not imply absence from the broader literature.

### No semantic resolution
The tokenisation approach does not resolve synonyms (e.g. 'presenilin 1' → PSEN1),
protein names, drug names, or pathway references. Targets may be well-studied
without appearing as gene symbols in text.

---

*This report is for research intelligence purposes only. It is not clinical
evidence and should not inform clinical or drug development decisions.*