"""
NeuroGraph - Data Collection Runner
-------------------------------------
Runs all three data collection scripts in sequence.
Output files will be saved to Data/raw/

Usage:
    python collect_all_data.py

To run a single source only:
    python -m src.data_collection.clinical_trials
    python -m src.data_collection.semantic_scholar
    python -m src.data_collection.open_targets
"""

print("=" * 55)
print("  NeuroGraph — Data Collection")
print("=" * 55)

print("\n[1/3] ClinicalTrials.gov")
print("-" * 35)
from src.data_collection.clinical_trials import main as run_clinical_trials
run_clinical_trials()

print("\n[2/3] Semantic Scholar")
print("-" * 35)
from src.data_collection.semantic_scholar import main as run_semantic_scholar
run_semantic_scholar()

print("\n[3/3] Open Targets")
print("-" * 35)
from src.data_collection.open_targets import main as run_open_targets
run_open_targets()

print("\n" + "=" * 55)
print("  All data collection complete.")
print("  Check Data/raw/ for output files:")
print("    clinical_trials_raw.json")
print("    semantic_scholar_raw.json")
print("    open_targets_raw.json")
print("=" * 55)
