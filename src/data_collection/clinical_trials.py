"""
ClinicalTrials.gov Data Collection
-----------------------------------
Fetches Alzheimer's disease clinical trial records from ClinicalTrials.gov API v2.
Saves raw JSON to: Data/raw/clinical_trials_raw.json

Expected output : JSON file, list of up to CLINICAL_TRIALS_MAX_RECORDS trial records
Typical run time: 30–60 seconds for 1000 records

Common errors
-------------
- ConnectionError       : check internet connection
- HTTPError 429         : rate limited — script retries automatically
- AssertionError        : empty results — verify DISEASE_NAME in config.py
- KeyError on nctId     : API response structure changed — check API v2 docs
"""

import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    CLINICAL_TRIALS_BASE_URL,
    CLINICAL_TRIALS_MAX_RECORDS,
    DATA_RAW_DIR,
    DISEASE_NAME,
)

OUTPUT_FILE = os.path.join(DATA_RAW_DIR, "clinical_trials_raw.json")
PAGE_SIZE = 100  # ClinicalTrials.gov v2 allows up to 1000; 100 is a safe default


def fetch_clinical_trials(disease: str, max_records: int) -> list:
    """Fetch clinical trial records from ClinicalTrials.gov API v2 with pagination."""
    all_studies = []
    page_token = None

    print(f"[ClinicalTrials] Fetching trials for: {disease}")

    while len(all_studies) < max_records:
        params = {
            "query.cond": disease,
            "format": "json",
            "pageSize": min(PAGE_SIZE, max_records - len(all_studies)),
        }
        if page_token:
            params["pageToken"] = page_token

        for attempt in range(3):
            try:
                response = requests.get(
                    CLINICAL_TRIALS_BASE_URL, params=params, timeout=30
                )
                if response.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"  Rate limited. Waiting {wait}s (attempt {attempt + 1}/3)...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.ConnectionError:
                print("  Connection error. Retrying in 5s...")
                time.sleep(5)
        else:
            print("  Max retries reached. Stopping early.")
            break

        data = response.json()
        studies = data.get("studies", [])

        if not studies:
            break

        all_studies.extend(studies)
        print(f"  Fetched {len(all_studies)} records so far...")

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        time.sleep(0.5)

    return all_studies[:max_records]


def validate(studies: list) -> None:
    """Validate the fetched study records."""
    assert len(studies) > 0, (
        "No studies returned. Check DISEASE_NAME in config.py or API availability."
    )

    first = studies[0]
    assert "protocolSection" in first, (
        "Unexpected response structure. The ClinicalTrials.gov API format may have changed."
    )

    id_module = first["protocolSection"].get("identificationModule", {})
    assert "nctId" in id_module, (
        "NCT ID missing from first record. Check API response structure."
    )

    nct_id = id_module["nctId"]
    print(f"  Validation passed: {len(studies)} studies fetched, first NCT ID = {nct_id}")


def main():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    studies = fetch_clinical_trials(DISEASE_NAME, CLINICAL_TRIALS_MAX_RECORDS)
    validate(studies)

    output = {
        "disease": DISEASE_NAME,
        "source": "ClinicalTrials.gov API v2",
        "total_fetched": len(studies),
        "studies": studies,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(studies)} studies -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
