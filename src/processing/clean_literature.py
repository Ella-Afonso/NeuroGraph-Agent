"""
Semantic Scholar Cleaning Script
----------------------------------
Reads : Data/raw/semantic_scholar_raw.json
Writes: Data/processed/literature_clean.csv

Extraction map (raw path -> column name)
-----------------------------------------
paperId                          -> paper_id
title                            -> title
abstract                         -> abstract
year                             -> year
citationCount                    -> citation_count
venue                            -> venue
url                              -> url
authors[].name                   -> authors  ("|"-joined)
fieldsOfStudy[]                  -> fields_of_study  ("|"-joined)
externalIds.DOI                  -> doi
externalIds.PubMed               -> pmid
<title> + " " + <abstract>       -> paper_text  (NLP input field)
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, SEMANTIC_SCHOLAR_YEAR_FROM

INPUT_FILE  = os.path.join(DATA_RAW_DIR,       "semantic_scholar_raw.json")
OUTPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "literature_clean.csv")

REQUIRED_COLUMNS = ["paper_id", "title", "year"]


# ── Extraction helper ───────────────────────────────────────────────────────────

def extract_row(paper: dict) -> dict:
    """Extract and flatten one paper record."""
    ext_ids = paper.get("externalIds") or {}
    authors = paper.get("authors") or []
    fields  = paper.get("fieldsOfStudy") or []

    title    = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()

    return {
        "paper_id":       paper.get("paperId", ""),
        "title":          title,
        "abstract":       abstract,
        "year":           paper.get("year"),
        "citation_count": paper.get("citationCount"),
        "venue":          paper.get("venue") or "",
        "url":            paper.get("url") or "",
        "authors":        "|".join(
                              a.get("name", "") for a in authors
                              if isinstance(a, dict) and a.get("name")
                          ),
        "fields_of_study": "|".join(str(f) for f in fields if f),
        "doi":            ext_ids.get("DOI") or "",
        "pmid":           ext_ids.get("PubMed") or "",
        "paper_text":     f"{title} {abstract}".strip(),
    }


# ── Validation ──────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    print(f"\n  Shape            : {df.shape[0]} rows x {df.shape[1]} columns")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing_cols, f"Required columns missing from output: {missing_cols}"

    assert len(df) > 0, "Output DataFrame is empty."

    dupes = df["paper_id"].duplicated().sum()
    if dupes > 0:
        print(f"  Warning          : {dupes} duplicate paper_id rows found")

    # Missing value report
    key_cols = ["paper_id", "title", "abstract", "year",
                "citation_count", "venue", "doi"]
    print("\n  Missing value summary (key columns):")
    for col in key_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
            pct = 100 * n_missing / len(df)
            print(f"    {col:<25} {n_missing:>4} missing  ({pct:.1f}%)")

    # Year range
    years = df["year"].dropna()
    if not years.empty:
        print(f"\n  Year range       : {int(years.min())} – {int(years.max())}")

        old = (years < SEMANTIC_SCHOLAR_YEAR_FROM).sum()
        if old > 0:
            print(f"  Warning          : {old} papers older than {SEMANTIC_SCHOLAR_YEAR_FROM}")

    # Citation count summary
    cit = df["citation_count"].dropna()
    if not cit.empty:
        print(f"\n  Citation count   : min={int(cit.min())}  "
              f"median={int(cit.median())}  max={int(cit.max())}")

    # Abstract coverage
    has_abstract = (df["abstract"].fillna("").str.strip() != "").sum()
    print(f"\n  Has abstract     : {has_abstract} / {len(df)} "
          f"({100*has_abstract/len(df):.1f}%)")

    print("\n  Validation passed.")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[CleanLiterature] Reading raw file...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    papers = raw.get("papers", [])
    print(f"  Raw records loaded: {len(papers)}")

    print("  Extracting fields...")
    rows = [extract_row(p) for p in papers]
    df = pd.DataFrame(rows)

    # Normalise types
    df["year"]           = pd.to_numeric(df["year"],           errors="coerce").astype("Int64")
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").astype("Int64")

    # Drop rows with no paper_id
    before = len(df)
    df = df[df["paper_id"].notna() & (df["paper_id"] != "")].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with no paper_id.")

    # Drop exact duplicate paper_id rows, keep first
    before = len(df)
    df = df.drop_duplicates(subset=["paper_id"], keep="first")
    deduped = before - len(df)
    if deduped:
        print(f"  Removed {deduped} duplicate paper_id rows.")

    validate(df)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n  Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
