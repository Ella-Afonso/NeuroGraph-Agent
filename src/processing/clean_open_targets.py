"""
Open Targets Cleaning Script
------------------------------
Reads : Data/raw/open_targets_raw.json
Writes: Data/processed/target_evidence_clean.csv

Extraction map (raw path -> column name)
-----------------------------------------
target.id                        -> target_id
target.approvedSymbol            -> approved_symbol
target.approvedName              -> approved_name
target.biotype                   -> biotype
score                            -> association_score
datasourceScores (serialised)    -> datasource_scores  (JSON string)
<top-level disease_id>           -> disease_id
<top-level disease_name>         -> disease_name

Individual datasource score columns are also pivoted out where present,
prefixed with "ds_" (e.g. ds_europepmc, ds_gwas_credible_sets).
This makes them directly usable for analysis without JSON parsing.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_PROCESSED_DIR, DATA_RAW_DIR

INPUT_FILE  = os.path.join(DATA_RAW_DIR,       "open_targets_raw.json")
OUTPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "target_evidence_clean.csv")

REQUIRED_COLUMNS = ["target_id", "approved_symbol", "association_score", "disease_id"]


# ── Extraction helper ───────────────────────────────────────────────────────────

def extract_row(assoc: dict, disease_id: str, disease_name: str) -> dict:
    """Extract and flatten one association record."""
    target = assoc.get("target") or {}
    ds_scores = assoc.get("datasourceScores") or []

    # Serialise full datasource score list as a JSON string for archival
    ds_scores_json = json.dumps(
        {item["id"]: item["score"] for item in ds_scores if "id" in item and "score" in item}
    )

    row = {
        "target_id":        target.get("id", ""),
        "approved_symbol":  target.get("approvedSymbol", ""),
        "approved_name":    target.get("approvedName", ""),
        "biotype":          target.get("biotype", ""),
        "association_score": assoc.get("score"),
        "datasource_scores": ds_scores_json,
        "disease_id":       disease_id,
        "disease_name":     disease_name,
    }

    # Pivot individual datasource scores into their own columns (ds_<id>)
    for item in ds_scores:
        ds_id    = item.get("id", "")
        ds_score = item.get("score")
        if ds_id:
            row[f"ds_{ds_id}"] = ds_score

    return row


# ── Validation ──────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    print(f"\n  Shape            : {df.shape[0]} rows x {df.shape[1]} columns")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing_cols, f"Required columns missing from output: {missing_cols}"

    assert len(df) > 0, "Output DataFrame is empty."

    dupes = df["target_id"].duplicated().sum()
    if dupes > 0:
        print(f"  Warning          : {dupes} duplicate target_id rows found")

    # Score range check
    scores = df["association_score"].dropna()
    assert not scores.empty, "No valid association scores found."
    assert scores.between(0.0, 1.0).all(), (
        "One or more association_score values are outside [0, 1]."
    )
    print(f"\n  association_score: min={scores.min():.4f}  "
          f"median={scores.median():.4f}  max={scores.max():.4f}")

    # Missing value report for core columns
    core_cols = ["target_id", "approved_symbol", "approved_name", "biotype", "association_score"]
    print("\n  Missing value summary (core columns):")
    for col in core_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            pct = 100 * n_missing / len(df)
            print(f"    {col:<25} {n_missing:>4} missing  ({pct:.1f}%)")

    # Biotype distribution
    print("\n  biotype distribution:")
    for btype, count in df["biotype"].value_counts().head(6).items():
        print(f"    {btype:<30} {count}")

    # Datasource columns present
    ds_cols = [c for c in df.columns if c.startswith("ds_")]
    print(f"\n  Datasource score columns: {len(ds_cols)}")
    print(f"    {', '.join(ds_cols)}")

    # Top 10 targets by score
    print("\n  Top 10 targets by association_score:")
    top10 = df.nlargest(10, "association_score")[
        ["approved_symbol", "approved_name", "association_score"]
    ]
    for _, r in top10.iterrows():
        print(f"    {r['approved_symbol']:<12} {r['association_score']:.4f}  {r['approved_name']}")

    print("\n  Validation passed.")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[CleanOpenTargets] Reading raw file...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    disease_id   = raw.get("disease_id", "")
    disease_name = raw.get("disease_name", "")
    associations = raw.get("associations", [])
    print(f"  Disease          : {disease_name} ({disease_id})")
    print(f"  Raw records      : {len(associations)}")

    print("  Extracting fields...")
    rows = [extract_row(a, disease_id, disease_name) for a in associations]
    df = pd.DataFrame(rows)

    # Normalise association score to float
    df["association_score"] = pd.to_numeric(df["association_score"], errors="coerce")

    # Normalise all ds_* columns to float
    ds_cols = [c for c in df.columns if c.startswith("ds_")]
    for col in ds_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no target_id
    before = len(df)
    df = df[df["target_id"].notna() & (df["target_id"] != "")].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with no target_id.")

    # Sort by association score descending
    df = df.sort_values("association_score", ascending=False).reset_index(drop=True)

    validate(df)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n  Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
