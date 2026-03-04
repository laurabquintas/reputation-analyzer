#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
google_reviews.py

Fetch Google reviews for Ananea Castelo Suites Hotel via the Google Places
API (New) and classify each review by topic using Ollama (mistral:7b).

Reviews are stored in a JSON file and deduplicated by review ID across runs.
Each review is classified into zero or more topics with positive/negative
sentiment.  A single review can mention the same topic both positively and
negatively.

NOTE: The Google Places API returns a maximum of 5 reviews per request.
There is no pagination for reviews.

USAGE
-----
Basic (API key via env var, Ollama running locally):
    python src/sites/google_reviews.py

Skip classification (Ollama not available):
    python src/sites/google_reviews.py --skip-classification

Reclassify previously unclassified reviews:
    python src/sites/google_reviews.py --reclassify

REQUIREMENTS
------------
requests, PyYAML
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from time import sleep

import requests
import yaml

from src.classification import (
    classify_review,
    is_ollama_available,
)

logger = logging.getLogger(__name__)

# ---------------------- Configuration ---------------------- #

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "hotels.yaml"

DEFAULT_JSON_PATH = str(DATA_DIR / "google_reviews.json")

ANANEA_HOTEL = "Ananea Castelo Suites Hotel"

PLACES_SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"


def _load_google_query() -> str:
    """Load the google_query for Ananea from config."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for h in cfg["hotels"]:
        if h["name"] == ANANEA_HOTEL:
            return h.get("google_query", "")
    return ""


ANANEA_GOOGLE_QUERY = _load_google_query()


# ---------------------- Google Places API ---------------------- #

def google_get_reviews(query: str, api_key: str, timeout: int = 15) -> list[dict]:
    """Fetch reviews for a hotel via Google Places Text Search.

    Returns up to 5 reviews (Google API limitation).
    """
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.reviews",
    }
    payload = {"textQuery": query}

    resp = requests.post(
        PLACES_SEARCH_TEXT_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    places = data.get("places", [])
    if not places:
        return []

    # Take the first (best match) place
    place = places[0]
    return place.get("reviews", [])


def _extract_review_id(review: dict) -> str:
    """Extract a unique review ID from the Google review resource name.

    Google reviews have a ``name`` field like ``places/PLACE_ID/reviews/REVIEW_ID``.
    """
    name = review.get("name", "")
    if "/" in name:
        return name.rsplit("/", 1)[-1]
    return name


def _extract_review_text(review: dict) -> str:
    """Extract review text, preferring originalText over translated text."""
    original = review.get("originalText", {})
    if isinstance(original, dict) and original.get("text"):
        return original["text"]
    text_obj = review.get("text", {})
    if isinstance(text_obj, dict):
        return text_obj.get("text", "")
    return str(text_obj) if text_obj else ""


def _extract_publish_date(review: dict) -> str:
    """Extract and normalize publish date from ISO 8601 publishTime."""
    publish_time = review.get("publishTime", "")
    if not publish_time:
        return ""
    try:
        dt = datetime.fromisoformat(publish_time.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return publish_time[:10] if len(publish_time) >= 10 else publish_time


# ---------------------- JSON storage ---------------------- #

def load_reviews(json_path: str) -> list[dict]:
    path = Path(json_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("reviews", [])


def save_reviews(reviews: list[dict], json_path: str) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": {
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "total_reviews": len(reviews),
        },
        "reviews": reviews,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def deduplicate_reviews(existing: list[dict], new_reviews: list[dict]) -> list[dict]:
    """Merge new reviews into existing, skipping duplicates by ID."""
    existing_ids = {str(r["id"]) for r in existing}
    merged = list(existing)
    for r in new_reviews:
        rid = str(r["id"])
        if rid not in existing_ids:
            merged.append(r)
            existing_ids.add(rid)
    return merged


# ---------------------- CLI ---------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Google reviews and classify topics via Ollama.")
    p.add_argument("--json", default=DEFAULT_JSON_PATH,
                   help=f"Output JSON path (default: {DEFAULT_JSON_PATH})")
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                   help="Scrape date tag (YYYY-MM-DD). Default: today.")
    p.add_argument("--api-key", default=None,
                   help="Google Maps API key. If omitted, uses GOOGLE_MAPS_API_KEY env var.")
    p.add_argument("--ollama-url", default="http://localhost:11434",
                   help="Ollama API base URL (default: http://localhost:11434)")
    p.add_argument("--skip-classification", action="store_true",
                   help="Skip Ollama classification, store reviews without topics.")
    p.add_argument("--reclassify", action="store_true",
                   help="Reclassify reviews that have classified=false.")
    p.add_argument("--min-delay", type=float, default=2.5,
                   help="Min delay (s) between requests (default: 2.5)")
    p.add_argument("--max-delay", type=float, default=5.0,
                   help="Max delay (s) between requests (default: 5.0)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    api_key = args.api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError("No API key provided. Use --api-key or set GOOGLE_MAPS_API_KEY.")

    # Only scrape Ananea
    if not ANANEA_GOOGLE_QUERY:
        logger.error("No google_query configured for %s", ANANEA_HOTEL)
        return 1

    existing_reviews = load_reviews(args.json)

    # Check Ollama
    ollama_ok = False if args.skip_classification else is_ollama_available(args.ollama_url)
    if not ollama_ok and not args.skip_classification:
        logger.warning("Ollama not available at %s. Reviews will be stored without classification.", args.ollama_url)

    # --- Reclassify mode ---
    if args.reclassify:
        if not ollama_ok:
            logger.error("Cannot reclassify: Ollama is not available.")
            return 1
        reclassified = 0
        for review in existing_reviews:
            if not review.get("classified", False) and review.get("text"):
                try:
                    topics = classify_review(review["text"], args.ollama_url)
                    review["topics"] = topics
                    review["classified"] = True
                    reclassified += 1
                    logger.info("Reclassified review %s: %d topics", review["id"], len(topics))
                except Exception as e:
                    logger.warning("Failed to reclassify review %s: %s", review["id"], e)
        save_reviews(existing_reviews, args.json)
        logger.info("Reclassified %d reviews.", reclassified)
        return 0

    # --- Normal scrape mode ---
    logger.info("Fetching Google reviews for %s (query=%s)", ANANEA_HOTEL, ANANEA_GOOGLE_QUERY)

    try:
        raw_reviews = google_get_reviews(ANANEA_GOOGLE_QUERY, api_key)
    except Exception as e:
        logger.error("Failed to fetch reviews: %s", e)
        return 1

    logger.info("API returned %d reviews", len(raw_reviews))

    existing_ids = {str(r["id"]) for r in existing_reviews}
    new_reviews: list[dict] = []

    for raw in raw_reviews:
        review_id = _extract_review_id(raw)
        if not review_id or review_id in existing_ids:
            logger.debug("Skipping duplicate review %s", review_id)
            continue

        text = _extract_review_text(raw)
        topics: list[dict] = []
        classified = False

        if ollama_ok and text:
            try:
                topics = classify_review(text, args.ollama_url)
                classified = True
                logger.info("  Review %s: %d topics classified", review_id, len(topics))
            except Exception as e:
                logger.warning("  Classification failed for review %s: %s", review_id, e)

        # Extract author info
        author_attr = raw.get("authorAttribution", {})
        author_name = author_attr.get("displayName", "") if isinstance(author_attr, dict) else ""

        review = {
            "id": review_id,
            "hotel": ANANEA_HOTEL,
            "source": "google",
            "rating": raw.get("rating"),
            "title": "",  # Google reviews have no title
            "text": text,
            "published_date": _extract_publish_date(raw),
            "author_name": author_name,
            "scraped_date": args.date,
            "topics": topics,
            "classified": classified,
        }

        new_reviews.append(review)
        existing_ids.add(review_id)

    all_reviews = deduplicate_reviews(existing_reviews, new_reviews)
    save_reviews(all_reviews, args.json)
    logger.info("Saved %d total reviews (%d new) to %s", len(all_reviews), len(new_reviews), args.json)

    # Rate limit politeness
    delay = random.uniform(args.min_delay, args.max_delay)
    sleep(delay)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
