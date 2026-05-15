"""
NeuroGraph Agent - Shared Configuration
"""

import os

# ── Disease focus ──────────────────────────────────────────────────────────────
DISEASE_NAME = "Alzheimer's disease"
DISEASE_ID_OPEN_TARGETS = "MONDO_0004975"   # Open Targets EFO identifier for Alzheimer's disease

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "Data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "processed")

# ── ClinicalTrials.gov ─────────────────────────────────────────────────────────
CLINICAL_TRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_MAX_RECORDS = 1000       # max results to fetch per run

# ── Semantic Scholar ───────────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_MAX_RECORDS = 500       # max papers to fetch per run
SEMANTIC_SCHOLAR_YEAR_FROM = 2018        # fetch papers from this year onwards

# ── Open Targets ───────────────────────────────────────────────────────────────
OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OPEN_TARGETS_MAX_RECORDS = 500           # max target associations to fetch
