#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tripadvisor_reviews.py

Fetch TripAdvisor review text for Ananea Castelo Suites Hotel via the
TripAdvisor Content API and classify each review by topic using Ollama
(mistral:7b).

Reviews are stored in a JSON file and deduplicated by review ID across runs.
Each review is classified into zero or more topics with positive/negative
sentiment.  A single review can mention the same topic both positively and
negatively (e.g. "breakfast was great but dinner was poor" -> meals positive
AND meals negative).

USAGE
-----
Basic (API key via env var, Ollama running locally):
    python src/sites/tripadvisor_reviews.py

Skip classification (Ollama not available):
    python src/sites/tripadvisor_reviews.py --skip-classification

Reclassify previously unclassified reviews:
    python src/sites/tripadvisor_reviews.py --reclassify

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
    VALID_TOPICS,
    VALID_SENTIMENTS,
    classify_review,
    is_ollama_available,
    _parse_classification,
)

logger = logging.getLogger(__name__)

# ---------------------- Configuration ---------------------- #

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "hotels.yaml"

DEFAULT_JSON_PATH = str(DATA_DIR / "tripadvisor_reviews.json")

ANANEA_HOTEL = "Ananea Castelo Suites Hotel"


def _load_location_ids() -> dict[str, str]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {h["name"]: h["tripadvisor_location_id"] for h in cfg["hotels"] if h.get("tripadvisor_location_id")}


LOCATION_IDS = _load_location_ids()


# Languages to fetch reviews in (covers most hotel review languages)
DEFAULT_LANGUAGES = ["en", "pt", "de", "fr", "es", "it", "nl"]


# ---------------------- TripAdvisor API ---------------------- #

def ta_get_reviews_page(
    location_id: str,
    api_key: str,
    language: str = "en",
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Fetch a single page of reviews.  Returns (reviews, total_results)."""
    url = f"https://api.content.tripadvisor.com/api/v1/location/{location_id}/reviews"
    params = {
        "key": api_key,
        "language": language,
        "offset": str(offset),
    }
    resp = requests.get(url, params=params, timeout=15)
    logger.debug("Reviews [lang=%s offset=%d] status: %d", language, offset, resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    paging = data.get("paging", {})
    total = int(paging.get("total_results", 0))
    return data.get("data", []), total


def ta_get_reviews(
    location_id: str,
    api_key: str,
    languages: list[str] | None = None,
    max_pages: int = 5,
) -> list[dict]:
    """Fetch reviews across multiple languages with pagination.

    Deduplicates by review ID (same review may appear in multiple languages).
    Returns up to ``max_pages * 5`` reviews per language.
    """
    if languages is None:
        languages = DEFAULT_LANGUAGES

    seen_ids: set[str] = set()
    all_reviews: list[dict] = []

    for lang in languages:
        offset = 0
        pages_fetched = 0
        while pages_fetched < max_pages:
            try:
                page, total = ta_get_reviews_page(location_id, api_key, lang, offset)
            except requests.HTTPError as exc:
                logger.warning("HTTP error for lang=%s offset=%d: %s", lang, offset, exc)
                break

            if not page:
                break

            for review in page:
                rid = str(review.get("id", ""))
                if rid and rid not in seen_ids:
                    review["_language"] = lang
                    all_reviews.append(review)
                    seen_ids.add(rid)

            pages_fetched += 1
            logger.info(
                "  [%s] page %d: %d reviews (total available: %d)",
                lang, pages_fetched, len(page), total,
            )

            # Stop if we've fetched everything for this language
            offset += len(page)
            if offset >= total:
                break

            # Polite delay between pages
            sleep(random.uniform(1.0, 2.0))

    logger.info("Fetched %d unique reviews across %d languages", len(all_reviews), len(languages))
    return all_reviews


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
    p = argparse.ArgumentParser(description="Fetch TripAdvisor reviews and classify topics via Ollama.")
    p.add_argument("--json", default=DEFAULT_JSON_PATH,
                   help=f"Output JSON path (default: {DEFAULT_JSON_PATH})")
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                   help="Scrape date tag (YYYY-MM-DD). Default: today.")
    p.add_argument("--api-key", default=None,
                   help="TripAdvisor API key. If omitted, uses TRIPADVISOR_API_KEY env var.")
    p.add_argument("--ollama-url", default="http://localhost:11434",
                   help="Ollama API base URL (default: http://localhost:11434)")
    p.add_argument("--skip-classification", action="store_true",
                   help="Skip Ollama classification, store reviews without topics.")
    p.add_argument("--reclassify", action="store_true",
                   help="Reclassify reviews that have classified=false.")
    p.add_argument("--languages", nargs="*", default=None,
                   help="Languages to fetch reviews in (default: en pt de fr es it nl)")
    p.add_argument("--max-pages", type=int, default=5,
                   help="Max API pages per language (5 reviews/page, default: 5)")
    p.add_argument("--min-delay", type=float, default=2.5,
                   help="Min delay (s) between hotel requests (default: 2.5)")
    p.add_argument("--max-delay", type=float, default=5.0,
                   help="Max delay (s) between hotel requests (default: 5.0)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    api_key = args.api_key or os.getenv("TRIPADVISOR_API_KEY")
    if not api_key:
        raise RuntimeError("No API key provided. Use --api-key or set TRIPADVISOR_API_KEY.")

    # Only scrape Ananea
    location_id = LOCATION_IDS.get(ANANEA_HOTEL)
    if not location_id:
        logger.error("No TripAdvisor location ID configured for %s", ANANEA_HOTEL)
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
    logger.info("Fetching reviews for %s (location_id=%s)", ANANEA_HOTEL, location_id)

    try:
        raw_reviews = ta_get_reviews(
            location_id, api_key,
            languages=args.languages,
            max_pages=args.max_pages,
        )
    except Exception as e:
        logger.error("Failed to fetch reviews: %s", e)
        return 1

    logger.info("API returned %d reviews", len(raw_reviews))

    existing_ids = {str(r["id"]) for r in existing_reviews}
    new_reviews: list[dict] = []

    for raw in raw_reviews:
        review_id = str(raw.get("id", ""))
        if not review_id or review_id in existing_ids:
            logger.debug("Skipping duplicate review %s", review_id)
            continue

        text = raw.get("text", "")
        topics: list[dict] = []
        classified = False

        if ollama_ok and text:
            try:
                topics = classify_review(text, args.ollama_url)
                classified = True
                logger.info("  Review %s: %d topics classified", review_id, len(topics))
            except Exception as e:
                logger.warning("  Classification failed for review %s: %s", review_id, e)

        # Parse subratings (API returns list of dicts or dict)
        raw_subratings = raw.get("subratings") or {}
        if isinstance(raw_subratings, list):
            subratings = {
                sr.get("name", ""): sr.get("value")
                for sr in raw_subratings
                if isinstance(sr, dict)
            }
        elif isinstance(raw_subratings, dict):
            subratings = {k: v.get("value") if isinstance(v, dict) else v for k, v in raw_subratings.items()}
        else:
            subratings = {}

        review = {
            "id": review_id,
            "hotel": ANANEA_HOTEL,
            "location_id": location_id,
            "rating": raw.get("rating"),
            "title": raw.get("title", ""),
            "text": text,
            "published_date": raw.get("published_date", ""),
            "travel_date": raw.get("travel_date", ""),
            "trip_type": raw.get("trip_type", ""),
            "subratings": subratings,
            "helpful_votes": raw.get("helpful_votes", 0),
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
