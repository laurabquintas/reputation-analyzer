#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expedia_reviews.py

Scrape Expedia review text for hotels and classify each review by topic
using Ollama (mistral:7b).

Reviews are stored in a JSON file and deduplicated by review ID across runs.
Each review is classified into zero or more topics with positive/negative
sentiment.

Expedia does not offer a public review API, so this scraper fetches the hotel
page and parses the HTML with BeautifulSoup. Anti-bot handling (URL candidates,
UA rotation, retry logic, proxy support) is reused from the score scraper.

USAGE
-----
Basic (Ollama running locally):
    python src/sites/expedia_reviews.py

Skip classification (Ollama not available):
    python src/sites/expedia_reviews.py --skip-classification

Reclassify previously unclassified reviews:
    python src/sites/expedia_reviews.py --reclassify

REQUIREMENTS
------------
requests, beautifulsoup4, PyYAML
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from time import sleep

import requests
import yaml
from bs4 import BeautifulSoup, Tag

from src.classification import (
    classify_review,
    is_ollama_available,
)
from src.sites.expedia import (
    USER_AGENTS,
    HEADERS,
    _expedia_url_candidates,
    fetch_page,
)

logger = logging.getLogger(__name__)

# ---------------------- Configuration ---------------------- #

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "hotels.yaml"

DEFAULT_JSON_PATH = str(DATA_DIR / "expedia_reviews.json")

ANANEA_HOTEL = "Ananea Castelo Suites Hotel"


def _load_expedia_urls() -> dict[str, str]:
    """Load Expedia hotel URLs from config."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        h["name"]: h["expedia_url"]
        for h in cfg["hotels"]
        if h.get("expedia_url")
    }


EXPEDIA_URLS = _load_expedia_urls()


def _hotel_url_to_reviews_url(hotel_url: str) -> str:
    """Ensure the hotel URL points to the reviews dialog.

    Expedia hotel URLs may already contain ``?pwaDialog=product-reviews``; if
    not, we append it so the reviews section is rendered server-side.
    """
    if "pwaDialog" not in hotel_url:
        sep = "&" if "?" in hotel_url else "?"
        return f"{hotel_url}{sep}pwaDialog=product-reviews"
    return hotel_url


# ---------------------- HTML scraping ---------------------- #

def _parse_rating(text: str) -> float | None:
    """Extract a numeric rating from text like '8.6' or '9,0'."""
    if not text:
        return None
    cleaned = text.strip().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if m:
        try:
            score = float(m.group(1))
            if 0.0 <= score <= 10.0:
                return score
        except ValueError:
            pass
    return None


def _parse_date(text: str) -> str:
    """Parse a date string to YYYY-MM-DD.

    Handles formats like:
      - "2025-03-12"  (ISO)
      - "Mar 12, 2025" (English)
      - "12 Mar 2025"
    """
    if not text:
        return ""
    text = text.strip()

    # ISO format
    if re.match(r"\d{4}-\d{2}-\d{2}", text):
        return text[:10]

    # "Mar 12, 2025" or "March 12, 2025"
    en_months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    lower = text.lower()
    for month_key, month_num in en_months.items():
        if month_key in lower:
            day_m = re.search(r"(\d{1,2})", text)
            year_m = re.search(r"(\d{4})", text)
            if year_m:
                day = day_m.group(1).zfill(2) if day_m else "01"
                return f"{year_m.group(1)}-{month_num}-{day}"

    return text


def _extract_review_id(element: Tag) -> str:
    """Try to extract a unique ID from a review HTML element."""
    for attr in ("data-review-id", "data-stid", "data-id", "id"):
        val = element.get(attr)
        if val:
            return str(val)
    return ""


def _scrape_reviews_from_html(html: str) -> list[dict]:
    """Parse review data from an Expedia page HTML.

    Returns a list of raw review dicts with keys:
      id, rating, title, text, travel_date, author_name
    """
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # --- Strategy 1: JSON-LD structured data --- #
    for script in soup.find_all("script", type="application/ld+json"):
        raw = (script.string or "").strip()
        if not raw:
            continue
        try:
            ld_data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        items = ld_data if isinstance(ld_data, list) else [ld_data]
        for item in items:
            if isinstance(item, dict) and item.get("@type") == "Review":
                review = _parse_jsonld_review(item)
                if review.get("text"):
                    reviews.append(review)

            # Reviews nested inside a Hotel/LodgingBusiness
            if isinstance(item, dict) and "review" in item:
                nested = item["review"]
                if isinstance(nested, list):
                    for r in nested:
                        review = _parse_jsonld_review(r)
                        if review.get("text"):
                            reviews.append(review)
                elif isinstance(nested, dict):
                    review = _parse_jsonld_review(nested)
                    if review.get("text"):
                        reviews.append(review)

    if reviews:
        return reviews

    # --- Strategy 2: HTML parsing --- #
    review_selectors = [
        {"attrs": {"data-stid": re.compile(r"review", re.I)}},
        {"attrs": {"itemprop": "review"}},
        {"attrs": {"itemtype": "http://schema.org/Review"}},
        {"class_": re.compile(r"review-card|review-content", re.I)},
    ]

    review_elements: list[Tag] = []
    for sel in review_selectors:
        found = soup.find_all(["div", "article", "section", "li"], **sel)
        if found:
            review_elements = found
            break

    for element in review_elements:
        review = _parse_html_review(element)
        if review.get("text"):
            reviews.append(review)

    return reviews


def _parse_jsonld_review(item: dict) -> dict:
    """Extract review fields from a JSON-LD Review object."""
    review_id = item.get("@id", "") or item.get("url", "")
    if "/" in review_id:
        review_id = review_id.rstrip("/").rsplit("/", 1)[-1]

    rating_obj = item.get("reviewRating", {})
    rating = None
    if isinstance(rating_obj, dict):
        rating = _parse_rating(str(rating_obj.get("ratingValue", "")))

    author_obj = item.get("author", {})
    author = ""
    if isinstance(author_obj, dict):
        author = author_obj.get("name", "")
    elif isinstance(author_obj, str):
        author = author_obj

    body = item.get("reviewBody", "") or item.get("description", "")
    title = item.get("name", "") or item.get("headline", "")

    date_published = item.get("datePublished", "") or item.get("dateCreated", "")

    return {
        "id": review_id,
        "rating": rating,
        "title": title,
        "text": body,
        "travel_date": _parse_date(date_published),
        "author_name": author,
    }


def _parse_html_review(element: Tag) -> dict:
    """Extract review fields from an HTML review element."""
    review_id = _extract_review_id(element)

    # Rating
    rating = None
    rating_el = (
        element.find(attrs={"itemprop": "ratingValue"})
        or element.find(class_=re.compile(r"rating", re.I))
        or element.find(attrs={"data-stid": re.compile(r"rating", re.I)})
    )
    if rating_el:
        rating = _parse_rating(rating_el.get_text())
        if rating is None:
            rating = _parse_rating(str(rating_el.get("content", "")))

    # Also try "X out of 10" or "X/10" patterns in the element
    if rating is None:
        el_text = element.get_text(" ", strip=True)
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10", el_text, re.I)
        if m:
            rating = _parse_rating(m.group(1))

    # Title
    title = ""
    title_el = (
        element.find(attrs={"itemprop": "name"})
        or element.find(class_=re.compile(r"review.*title|title", re.I))
        or element.find(["h2", "h3", "h4"])
    )
    if title_el:
        title = title_el.get_text(strip=True)

    # Review text
    text = ""
    text_el = (
        element.find(attrs={"itemprop": "reviewBody"})
        or element.find(attrs={"itemprop": "description"})
        or element.find(class_=re.compile(r"review.*text|review.*body", re.I))
        or element.find(attrs={"data-stid": re.compile(r"content|text|body", re.I)})
    )
    if text_el:
        text = text_el.get_text(strip=True)
    else:
        paragraphs = element.find_all("p")
        if paragraphs:
            text = max((p.get_text(strip=True) for p in paragraphs), key=len, default="")

    # Date
    travel_date = ""
    date_el = (
        element.find(attrs={"itemprop": "datePublished"})
        or element.find(class_=re.compile(r"date", re.I))
        or element.find("time")
    )
    if date_el:
        travel_date = _parse_date(
            date_el.get("datetime", "") or date_el.get("content", "") or date_el.get_text(strip=True)
        )

    # Author
    author_name = ""
    author_el = (
        element.find(attrs={"itemprop": "author"})
        or element.find(class_=re.compile(r"author|user", re.I))
    )
    if author_el:
        author_name = author_el.get_text(strip=True)

    return {
        "id": review_id,
        "rating": rating,
        "title": title,
        "text": text,
        "travel_date": travel_date,
        "author_name": author_name,
    }


def expedia_get_reviews(
    hotel_url: str,
    max_pages: int = 5,
    timeout: int = 20,
    retries: int = 2,
    min_delay: float = 2.5,
    max_delay: float = 5.0,
) -> list[dict]:
    """Fetch reviews from Expedia hotel page.

    Uses anti-bot measures from the score scraper: URL candidates,
    UA rotation, retry logic, proxy support.
    """
    reviews_url = _hotel_url_to_reviews_url(hotel_url)
    all_reviews: list[dict] = []
    seen_ids: set[str] = set()

    for page in range(1, max_pages + 1):
        url = reviews_url if page == 1 else f"{reviews_url}&startIndex={10 * (page - 1)}"
        logger.info("Fetching Expedia reviews page %d: %s", page, url)

        html = fetch_page(url, timeout=timeout, retries=retries)
        if html is None:
            logger.warning("Failed to fetch page %d, stopping.", page)
            break

        count_before = len(all_reviews)
        page_reviews = _scrape_reviews_from_html(html)

        for review in page_reviews:
            rid = review.get("id", "")
            if not rid:
                content = (review.get("text", "") + review.get("title", "")).encode()
                rid = hashlib.md5(content).hexdigest()[:16]
                review["id"] = rid
            if rid not in seen_ids:
                all_reviews.append(review)
                seen_ids.add(rid)

        new_on_page = len(all_reviews) - count_before
        logger.info("Page %d: %d new reviews (%d total)", page, new_on_page, len(all_reviews))

        if new_on_page == 0:
            logger.info("No new reviews on page %d, stopping.", page)
            break

        sleep(random.uniform(min_delay, max_delay))

    logger.info("Fetched %d unique reviews total", len(all_reviews))
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
    p = argparse.ArgumentParser(
        description="Scrape Expedia reviews and classify topics via Ollama."
    )
    p.add_argument(
        "--json", default=DEFAULT_JSON_PATH,
        help=f"Output JSON path (default: {DEFAULT_JSON_PATH})",
    )
    p.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"),
        help="Scrape date tag (YYYY-MM-DD). Default: today.",
    )
    p.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    p.add_argument(
        "--skip-classification", action="store_true",
        help="Skip Ollama classification, store reviews without topics.",
    )
    p.add_argument(
        "--reclassify", action="store_true",
        help="Reclassify reviews that have classified=false.",
    )
    p.add_argument(
        "--max-pages", type=int, default=5,
        help="Max review pages to fetch (default: 5)",
    )
    p.add_argument(
        "--timeout", type=int, default=20,
        help="HTTP timeout per request in seconds (default: 20)",
    )
    p.add_argument(
        "--retries", type=int, default=2,
        help="Retries per URL candidate (default: 2)",
    )
    p.add_argument(
        "--min-delay", type=float, default=2.5,
        help="Min delay (s) between page requests (default: 2.5)",
    )
    p.add_argument(
        "--max-delay", type=float, default=5.0,
        help="Max delay (s) between page requests (default: 5.0)",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    hotel_url = EXPEDIA_URLS.get(ANANEA_HOTEL)
    if not hotel_url:
        logger.error("No Expedia URL configured for %s", ANANEA_HOTEL)
        return 1

    existing_reviews = load_reviews(args.json)

    # Check Ollama
    ollama_ok = (
        False if args.skip_classification
        else is_ollama_available(args.ollama_url)
    )
    if not ollama_ok and not args.skip_classification:
        logger.warning(
            "Ollama not available at %s. Reviews will be stored without classification.",
            args.ollama_url,
        )

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
                    logger.info(
                        "Reclassified review %s: %d topics",
                        review["id"], len(topics),
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to reclassify review %s: %s",
                        review["id"], e,
                    )
        save_reviews(existing_reviews, args.json)
        logger.info("Reclassified %d reviews.", reclassified)
        return 0

    # --- Normal scrape mode ---
    logger.info("Scraping Expedia reviews for %s", ANANEA_HOTEL)

    try:
        raw_reviews = expedia_get_reviews(
            hotel_url,
            max_pages=args.max_pages,
            timeout=args.timeout,
            retries=args.retries,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
        )
    except Exception as e:
        logger.error("Failed to scrape reviews: %s", e)
        return 1

    logger.info("Scraped %d reviews from Expedia", len(raw_reviews))

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
                logger.info(
                    "  Review %s: %d topics classified",
                    review_id, len(topics),
                )
            except Exception as e:
                logger.warning(
                    "  Classification failed for review %s: %s",
                    review_id, e,
                )

        review = {
            "id": review_id,
            "hotel": ANANEA_HOTEL,
            "source": "expedia",
            "rating": raw.get("rating"),
            "title": raw.get("title", ""),
            "text": text,
            "published_date": raw.get("travel_date", ""),
            "author_name": raw.get("author_name", ""),
            "scraped_date": args.date,
            "topics": topics,
            "classified": classified,
        }

        new_reviews.append(review)
        existing_ids.add(review_id)

    all_reviews = deduplicate_reviews(existing_reviews, new_reviews)
    save_reviews(all_reviews, args.json)
    logger.info(
        "Saved %d total reviews (%d new) to %s",
        len(all_reviews), len(new_reviews), args.json,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
