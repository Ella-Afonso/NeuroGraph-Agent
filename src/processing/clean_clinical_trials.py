"""
Clinical Trials Cleaning Script
---------------------------------
Reads : Data/raw/clinical_trials_raw.json
Writes: Data/processed/clinical_trials_clean.csv

Extraction map (raw path -> column name)
-----------------------------------------
identificationModule.nctId                          -> nct_id
identificationModule.briefTitle                     -> brief_title
identificationModule.officialTitle                  -> official_title
statusModule.overallStatus                          -> overall_status
statusModule.startDateStruct.date                   -> start_date
statusModule.completionDateStruct.date              -> completion_date
designModule.studyType                              -> study_type
designModule.phases[]                               -> phases  ("|"-joined)
designModule.designInfo.primaryPurpose              -> primary_purpose
designModule.enrollmentInfo.count                   -> enrollment_count
sponsorCollaboratorsModule.leadSponsor.name         -> sponsor_name
sponsorCollaboratorsModule.leadSponsor.class        -> sponsor_class
conditionsModule.conditions[]                       -> conditions  ("|"-joined)
armsInterventionsModule.interventions[].name        -> interventions  ("|"-joined)
contactsLocationsModule.locations[].country         -> countries  ("|"-joined unique)
eligibilityModule.minimumAge                        -> minimum_age
eligibilityModule.maximumAge                        -> maximum_age
eligibilityModule.sex                               -> sex
eligibilityModule.eligibilityCriteria               -> eligibility_criteria
outcomesModule.primaryOutcomes[].measure            -> primary_outcomes  ("|"-joined)
descriptionModule.briefSummary                      -> brief_summary
<brief_title> + " " + <brief_summary>               -> trial_text  (NLP input field)
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_PROCESSED_DIR, DATA_RAW_DIR

INPUT_FILE = os.path.join(DATA_RAW_DIR, "clinical_trials_raw.json")
OUTPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "clinical_trials_clean.csv")

# Columns that must not be entirely empty for a row to be useful
REQUIRED_COLUMNS = ["nct_id", "brief_title", "overall_status"]


# ── Extraction helpers ──────────────────────────────────────────────────────────

def _safe(d: dict, *keys, default=None):
    """Safely traverse nested dict keys."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


def _join(lst, key=None, sep="|") -> str:
    """Flatten a list of strings or dicts to a delimited string."""
    if not lst:
        return ""
    if key:
        items = [str(item.get(key, "")) for item in lst if isinstance(item, dict)]
    else:
        items = [str(item) for item in lst]
    return sep.join(i for i in items if i)


def extract_row(study: dict) -> dict:
    """Extract and flatten one study record."""
    ps = study.get("protocolSection", {})

    id_mod   = ps.get("identificationModule", {})
    stat_mod = ps.get("statusModule", {})
    des_mod  = ps.get("designModule", {})
    spon_mod = ps.get("sponsorCollaboratorsModule", {})
    cond_mod = ps.get("conditionsModule", {})
    arms_mod = ps.get("armsInterventionsModule", {})
    elig_mod = ps.get("eligibilityModule", {})
    out_mod  = ps.get("outcomesModule", {})
    desc_mod = ps.get("descriptionModule", {})
    loc_mod  = ps.get("contactsLocationsModule", {})

    # Locations: unique countries
    locations   = loc_mod.get("locations", []) or []
    countries   = list(dict.fromkeys(
        loc.get("country", "") for loc in locations if loc.get("country")
    ))

    # Interventions: name field inside interventions list
    interventions_list = arms_mod.get("interventions", []) or []

    # Primary outcome measures
    primary_outcomes = out_mod.get("primaryOutcomes", []) or []

    brief_title   = id_mod.get("briefTitle", "") or ""
    brief_summary = desc_mod.get("briefSummary", "") or ""

    return {
        "nct_id":               id_mod.get("nctId", ""),
        "brief_title":          brief_title,
        "official_title":       id_mod.get("officialTitle", ""),
        "overall_status":       stat_mod.get("overallStatus", ""),
        "start_date":           _safe(stat_mod, "startDateStruct", "date"),
        "completion_date":      _safe(stat_mod, "completionDateStruct", "date"),
        "study_type":           des_mod.get("studyType", ""),
        "phases":               _join(des_mod.get("phases", [])),
        "primary_purpose":      _safe(des_mod, "designInfo", "primaryPurpose"),
        "enrollment_count":     _safe(des_mod, "enrollmentInfo", "count"),
        "sponsor_name":         _safe(spon_mod, "leadSponsor", "name"),
        "sponsor_class":        _safe(spon_mod, "leadSponsor", "class"),
        "conditions":           _join(cond_mod.get("conditions", [])),
        "interventions":        _join(interventions_list, key="name"),
        "countries":            _join(countries),
        "minimum_age":          elig_mod.get("minimumAge", ""),
        "maximum_age":          elig_mod.get("maximumAge", ""),
        "sex":                  elig_mod.get("sex", ""),
        "eligibility_criteria": elig_mod.get("eligibilityCriteria", ""),
        "primary_outcomes":     _join(primary_outcomes, key="measure"),
        "brief_summary":        brief_summary,
        "trial_text":           f"{brief_title} {brief_summary}".strip(),
    }


# ── Validation ──────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    print(f"\n  Shape            : {df.shape[0]} rows x {df.shape[1]} columns")

    # Required columns present
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing_cols, f"Required columns missing from output: {missing_cols}"

    # No duplicate NCT IDs
    dupes = df["nct_id"].duplicated().sum()
    if dupes > 0:
        print(f"  Warning          : {dupes} duplicate nct_id rows found")

    # Row count sanity
    assert len(df) > 0, "Output DataFrame is empty."

    # Missing value report for key columns
    key_cols = ["nct_id", "brief_title", "overall_status", "study_type",
                "sponsor_name", "conditions", "enrollment_count"]
    print("\n  Missing value summary (key columns):")
    for col in key_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum() + (df[col] == "").sum()
            pct = 100 * n_missing / len(df)
            print(f"    {col:<25} {n_missing:>4} missing  ({pct:.1f}%)")

    # Overall status distribution
    print("\n  overall_status distribution:")
    for status, count in df["overall_status"].value_counts().head(8).items():
        print(f"    {status:<30} {count}")

    print("\n  Validation passed.")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[CleanClinicalTrials] Reading raw file...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    studies = raw.get("studies", [])
    print(f"  Raw records loaded: {len(studies)}")

    print("  Extracting fields...")
    rows = [extract_row(s) for s in studies]
    df = pd.DataFrame(rows)

    # Normalise types
    df["enrollment_count"] = pd.to_numeric(df["enrollment_count"], errors="coerce")
    df["start_date"]       = pd.to_datetime(df["start_date"],       errors="coerce")
    df["completion_date"]  = pd.to_datetime(df["completion_date"],  errors="coerce")

    # Drop rows with no NCT ID (should not happen, but guard anyway)
    before = len(df)
    df = df[df["nct_id"].notna() & (df["nct_id"] != "")].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with no nct_id.")

    validate(df)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n  Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
