"""
NeuroGraph - Data Cleaning Runner
-----------------------------------
Runs all three cleaning scripts in sequence.
Reads from : Data/raw/
Writes to  : Data/processed/

Usage:
    python clean_all_data.py

To run a single cleaner only:
    python -m src.processing.clean_clinical_trials
    python -m src.processing.clean_literature
    python -m src.processing.clean_open_targets
"""

print("=" * 55)
print("  NeuroGraph — Data Cleaning")
print("=" * 55)

print("\n[1/3] Clinical Trials")
print("-" * 35)
from src.processing.clean_clinical_trials import main as clean_trials
clean_trials()

print("\n[2/3] Literature (Semantic Scholar)")
print("-" * 35)
from src.processing.clean_literature import main as clean_literature
clean_literature()

print("\n[3/3] Open Targets")
print("-" * 35)
from src.processing.clean_open_targets import main as clean_targets
clean_targets()

print("\n" + "=" * 55)
print("  All cleaning complete.")
print("  Check Data/processed/ for output files:")
print("    clinical_trials_clean.csv")
print("    literature_clean.csv")
print("    target_evidence_clean.csv")
print("=" * 55)
