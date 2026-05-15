"""
Open Targets Data Collection
----------------------------
Fetches Alzheimer's disease target/gene associations from the Open Targets
Platform GraphQL API.

Saves raw JSON to:
    Data/raw/open_targets_raw.json

For Alzheimer's disease, use:
    MONDO_0004975

This script intentionally uses a conservative GraphQL query to avoid fields
that may change across Open Targets API releases.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    DATA_RAW_DIR,
    DISEASE_ID_OPEN_TARGETS,
    OPEN_TARGETS_GRAPHQL_URL,
    OPEN_TARGETS_MAX_RECORDS,
)

OUTPUT_FILE = os.path.join(DATA_RAW_DIR, "open_targets_raw.json")
PAGE_SIZE = 25


QUERY = """
query DiseaseTargetAssociations($diseaseId: String!, $index: Int!, $size: Int!) {
  disease(efoId: $diseaseId) {
    id
    name
    associatedTargets(page: { index: $index, size: $size }) {
      count
      rows {
        score
        target {
          id
          approvedSymbol
          approvedName
          biotype
        }
        datasourceScores {
          id
          score
        }
      }
    }
  }
}
"""


def post_graphql_query(
    query: str,
    variables: Dict[str, Any],
    max_retries: int = 5,
) -> Optional[Dict[str, Any]]:
    """Send a GraphQL POST request to Open Targets with useful debugging."""

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "NeuroGraph-Agent/0.1 academic portfolio project",
    }

    payload = {
        "query": query,
        "variables": variables,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                OPEN_TARGETS_GRAPHQL_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code in {429, 500, 502, 503, 504}:
                wait = min(60, 10 * attempt)
                print(
                    f"  Temporary error {response.status_code}. "
                    f"Waiting {wait}s (attempt {attempt}/{max_retries})..."
                )
                time.sleep(wait)
                continue

            if response.status_code == 400:
                print("\n  Open Targets returned 400 Bad Request.")
                print("  This usually means the GraphQL query contains an invalid field.")
                print("\n  Response body:")
                print(response.text[:3000])
                return None

            response.raise_for_status()

            data = response.json()

            if "errors" in data:
                print("\n  GraphQL returned errors:")
                print(json.dumps(data["errors"], indent=2))
                return None

            return data

        except requests.exceptions.Timeout:
            wait = min(60, 10 * attempt)
            print(
                f"  Timeout. Waiting {wait}s "
                f"(attempt {attempt}/{max_retries})..."
            )
            time.sleep(wait)

        except requests.exceptions.ConnectionError:
            wait = min(60, 10 * attempt)
            print(
                f"  Connection error. Waiting {wait}s "
                f"(attempt {attempt}/{max_retries})..."
            )
            time.sleep(wait)

        except requests.exceptions.HTTPError as error:
            print(f"\n  HTTP error: {error}")
            try:
                print("  Response body:")
                print(response.text[:3000])
            except NameError:
                pass
            return None

    return None


def fetch_target_associations(disease_id: str, max_records: int) -> dict:
    """Fetch disease-target association rows from Open Targets GraphQL API."""

    all_rows: List[Dict[str, Any]] = []
    total_available: Optional[int] = None
    disease_name_returned: Optional[str] = None
    page_index = 0

    print(f"[OpenTargets] Fetching target associations for disease ID: {disease_id}")

    while len(all_rows) < max_records:
        size = min(PAGE_SIZE, max_records - len(all_rows))

        variables = {
            "diseaseId": disease_id,
            "index": page_index,
            "size": size,
        }

        data = post_graphql_query(QUERY, variables)

        if data is None:
            raise RuntimeError(
                "Open Targets request failed. Check the printed response body above. "
                "Most likely cause: invalid GraphQL field or schema mismatch."
            )

        disease_node = data.get("data", {}).get("disease")

        if disease_node is None:
            raise RuntimeError(
                f"No disease node returned for disease ID: {disease_id}. "
                "Check DISEASE_ID_OPEN_TARGETS in config.py. "
                "For Alzheimer's disease, use MONDO_0004975."
            )

        disease_name_returned = disease_node.get("name")
        associated = disease_node.get("associatedTargets", {})

        if total_available is None:
            total_available = associated.get("count", 0)
            print(f"  Disease: {disease_name_returned}")
            print(f"  Total associations available: {total_available}")

        rows = associated.get("rows", [])

        if not rows:
            print("  No more associations returned.")
            break

        all_rows.extend(rows)
        print(f"  Fetched {len(all_rows)} associations so far...")

        page_index += 1
        time.sleep(0.5)

        if total_available is not None and len(all_rows) >= total_available:
            break

    return {
        "disease_id": disease_id,
        "disease_name": disease_name_returned,
        "source": "Open Targets GraphQL API",
        "total_available": total_available,
        "total_fetched": len(all_rows[:max_records]),
        "associations": all_rows[:max_records],
    }


def validate(result: dict) -> None:
    """Validate the fetched association records."""

    if result["total_fetched"] <= 0:
        raise RuntimeError(
            "No associations returned. Check DISEASE_ID_OPEN_TARGETS in config.py. "
            "For Alzheimer's disease, use MONDO_0004975."
        )

    first = result["associations"][0]

    if "target" not in first:
        raise RuntimeError("Target field missing. Check GraphQL query fields.")

    if "score" not in first:
        raise RuntimeError("Score field missing. Check GraphQL query fields.")

    target = first["target"]

    if "approvedSymbol" not in target:
        raise RuntimeError("approvedSymbol missing from target.")

    scores = [
        row["score"]
        for row in result["associations"]
        if row.get("score") is not None
    ]

    if not scores:
        raise RuntimeError("No valid association scores found.")

    if not all(0.0 <= score <= 1.0 for score in scores):
        raise RuntimeError(
            "One or more scores are outside the expected 0–1 range."
        )

    print(
        f"  Validation passed: {result['total_fetched']} associations fetched, "
        f"disease = {result['disease_name']}, "
        f"score range {min(scores):.3f}–{max(scores):.3f}"
    )


def main() -> None:
    """Run Open Targets collection and save raw JSON."""

    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    result = fetch_target_associations(
        disease_id=DISEASE_ID_OPEN_TARGETS,
        max_records=OPEN_TARGETS_MAX_RECORDS,
    )

    validate(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(f"  Saved {result['total_fetched']} associations -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()