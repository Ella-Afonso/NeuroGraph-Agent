"""
Semantic Scholar Data Collection
--------------------------------
Fetches Alzheimer's disease research papers from the Semantic Scholar API.
Saves raw JSON to: Data/raw/semantic_scholar_raw.json

Important:
- Do NOT paste your API key into this file.
- Set it as an environment variable instead:

PowerShell temporary:
    $env:SEMANTIC_SCHOLAR_API_KEY="your_api_key_here"

PowerShell permanent:
    setx SEMANTIC_SCHOLAR_API_KEY "your_api_key_here"

Then restart VS Code if using setx.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    DATA_RAW_DIR,
    DISEASE_NAME,
    SEMANTIC_SCHOLAR_BASE_URL,
    SEMANTIC_SCHOLAR_MAX_RECORDS,
    SEMANTIC_SCHOLAR_YEAR_FROM,
)

OUTPUT_FILE = os.path.join(DATA_RAW_DIR, "semantic_scholar_raw.json")

# Keep this conservative because the authenticated introductory limit is 1 request/second.
PAGE_SIZE = 20

FIELDS = (
    "paperId,title,abstract,year,citationCount,authors,venue,url,"
    "externalIds,fieldsOfStudy"
)


def get_api_key() -> Optional[str]:
    """Read Semantic Scholar API key from environment variable."""
    return os.getenv("SEMANTIC_SCHOLAR_API_KEY")


def build_headers() -> Dict[str, str]:
    """Build request headers, including API key if available."""
    headers = {
        "User-Agent": "NeuroGraph-Agent/0.1 academic portfolio project"
    }

    api_key = get_api_key()

    if api_key:
        headers["x-api-key"] = api_key
        print("[SemanticScholar] API key detected. Using authenticated requests.")
    else:
        print(
            "[SemanticScholar] No API key found. "
            "Unauthenticated requests may be rate-limited."
        )

    return headers


def request_with_backoff(
    url: str,
    params: Dict[str, Any],
    headers: Dict[str, str],
    max_retries: int = 8,
) -> Optional[requests.Response]:
    """Send request with retry/backoff for rate limits and temporary errors."""

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=60,
            )

            if response.status_code == 429:
                wait = min(120, 15 * attempt)
                print(
                    f"  Rate limited: 429 Too Many Requests. "
                    f"Waiting {wait}s (attempt {attempt}/{max_retries})..."
                )
                time.sleep(wait)
                continue

            if response.status_code in {500, 502, 503, 504}:
                wait = min(120, 10 * attempt)
                print(
                    f"  Server error {response.status_code}. "
                    f"Waiting {wait}s (attempt {attempt}/{max_retries})..."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            wait = min(120, 10 * attempt)
            print(
                f"  Timeout. Waiting {wait}s "
                f"(attempt {attempt}/{max_retries})..."
            )
            time.sleep(wait)

        except requests.exceptions.ConnectionError:
            wait = min(120, 10 * attempt)
            print(
                f"  Connection error. Waiting {wait}s "
                f"(attempt {attempt}/{max_retries})..."
            )
            time.sleep(wait)

        except requests.exceptions.HTTPError as error:
            print(f"  HTTP error: {error}")
            return None

    return None


def fetch_papers(
    disease: str,
    max_records: int,
    year_from: int,
) -> List[Dict[str, Any]]:
    """Fetch paper records from Semantic Scholar with offset pagination."""

    all_papers: List[Dict[str, Any]] = []
    offset = 0
    headers = build_headers()

    search_url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search"

    # Broader query helps because papers may use either spelling.
    query = 'Alzheimer disease OR Alzheimer\'s disease'

    print(
        f"[SemanticScholar] Fetching papers for: {query} "
        f"(from {year_from} onwards)"
    )

    while len(all_papers) < max_records:
        limit = min(PAGE_SIZE, max_records - len(all_papers))

        params = {
            "query": query,
            "fields": FIELDS,
            "limit": limit,
            "offset": offset,
            "year": f"{year_from}-",
        }

        response = request_with_backoff(
            url=search_url,
            params=params,
            headers=headers,
        )

        if response is None:
            print("  Request failed after retries. Stopping early.")
            break

        data = response.json()
        papers = data.get("data", [])

        if not papers:
            print("  No more papers returned by API.")
            break

        all_papers.extend(papers)
        offset += len(papers)

        print(f"  Fetched {len(all_papers)} papers so far...")

        # Must stay below 1 request/second. 1.2s gives a small safety buffer.
        time.sleep(1.2)

    return all_papers[:max_records]


def validate(papers: List[Dict[str, Any]]) -> None:
    """Validate fetched paper records."""

    if len(papers) == 0:
        raise RuntimeError(
            "\nNo Semantic Scholar papers were returned.\n\n"
            "Possible causes:\n"
            "1. API key is not set correctly.\n"
            "2. Semantic Scholar is still rate-limiting the request.\n"
            "3. Query returned no results.\n\n"
            "Check your key with:\n"
            "python -c \"from dotenv import load_dotenv; import os; load_dotenv(); print(bool(os.getenv('SEMANTIC_SCHOLAR_API_KEY')))\"\n"
        )

    first = papers[0]

    if "title" not in first:
        raise RuntimeError("Title field missing. Check FIELDS parameter.")

    if "year" not in first:
        raise RuntimeError("Year field missing. Check FIELDS parameter.")

    years = [paper.get("year") for paper in papers if paper.get("year")]

    if len(years) == 0:
        raise RuntimeError("No year data found in any records.")

    old_years = [year for year in years if year < SEMANTIC_SCHOLAR_YEAR_FROM]

    if old_years:
        print(
            f"  Warning: {len(old_years)} papers are older than "
            f"{SEMANTIC_SCHOLAR_YEAR_FROM}. Keeping them for now."
        )

    print(
        f"  Validation passed: {len(papers)} papers fetched, "
        f"year range {min(years)}–{max(years)}"
    )


def main() -> None:
    """Run Semantic Scholar collection and save raw JSON."""

    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    papers = fetch_papers(
        disease=DISEASE_NAME,
        max_records=SEMANTIC_SCHOLAR_MAX_RECORDS,
        year_from=SEMANTIC_SCHOLAR_YEAR_FROM,
    )

    validate(papers)

    output = {
        "disease": DISEASE_NAME,
        "source": "Semantic Scholar API",
        "year_from": SEMANTIC_SCHOLAR_YEAR_FROM,
        "total_fetched": len(papers),
        "papers": papers,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"  Saved {len(papers)} papers -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()