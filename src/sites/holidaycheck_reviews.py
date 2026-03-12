#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
holidaycheck_reviews.py

Scrape HolidayCheck review text for Ananea Castelo Suites Hotel and classify
each review by topic using Ollama (mistral:7b).

Reviews are stored in a JSON file and deduplicated by review ID across runs.
Each review is classified into zero or more topics with positive/negative
sentiment.

HolidayCheck does not offer a public review API, so this scraper fetches the
hotel reviews page (``/hrd/`` path) and parses the HTML with BeautifulSoup.

USAGE
-----
Basic (Ollama running locally):
    python src/sites/holidaycheck_reviews.py

Skip classification (Ollama not available):
    python src/sites/holidaycheck_reviews.py --skip-classification

Reclassify previously unclassified reviews:
    python src/sites/holidaycheck_reviews.py --reclassify

REQUIREMENTS
------------
requests, beautifulsoup4, PyYAML
"""

from __future__ import annotations

import argparse
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

logger = logging.getLogger(__name__)

# ---------------------- Configuration ---------------------- #

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "hotels.yaml"

DEFAULT_JSON_PATH = str(DATA_DIR / "holidaycheck_reviews.json")

ANANEA_HOTEL = "Ananea Castelo Suites Hotel"

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
}


def _load_holidaycheck_urls() -> dict[str, str]:
    """Load HolidayCheck hotel URLs from config."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        h["name"]: h["holidaycheck_url"]
        for h in cfg["hotels"]
        if h.get("holidaycheck_url")
    }


HC_URLS = _load_holidaycheck_urls()


def _hotel_url_to_reviews_url(hotel_url: str) -> str:
    """Convert a hotel info URL (/hi/...) to the reviews URL (/hrd/...).

    HolidayCheck uses /hi/{slug}/{uuid} for hotel info pages and
    /hrd/{slug}/{uuid} for the review listing pages.
    """
    return hotel_url.replace("/hi/", "/hrd/", 1)


# ---------------------- HTML scraping ---------------------- #

def _parse_rating(text: str) -> float | None:
    """Extract a numeric rating from text like '5,5' or '4.0'."""
    if not text:
        return None
    cleaned = text.strip().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _parse_date(text: str) -> str:
    """Parse a German-style date string to YYYY-MM-DD.

    Handles formats like:
      - "12.03.2025"  (DD.MM.YYYY)
      - "März 2025"   (month year)
      - "2025-03-12"  (ISO)
    """
    if not text:
        return ""
    text = text.strip()

    # ISO format
    if re.match(r"\d{4}-\d{2}-\d{2}", text):
        return text[:10]

    # DD.MM.YYYY
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"

    # German month name + year
    german_months = {
        "januar": "01", "februar": "02", "märz": "03", "april": "04",
        "mai": "05", "juni": "06", "juli": "07", "august": "08",
        "september": "09", "oktober": "10", "november": "11", "dezember": "12",
    }
    for month_name, month_num in german_months.items():
        if month_name in text.lower():
            year_m = re.search(r"(\d{4})", text)
            if year_m:
                return f"{year_m.group(1)}-{month_num}-01"

    return text


def _extract_review_id_from_element(element: Tag) -> str:
    """Try to extract a unique ID from a review HTML element."""
    # Check data attributes
    for attr in ("data-review-id", "data-id", "id"):
        val = element.get(attr)
        if val:
            return str(val)

    # Check for links containing /hrd/ with review UUID
    link = element.find("a", href=re.compile(r"/hrd/"))
    if link:
        href = link.get("href", "")
        parts = href.rstrip("/").split("/")
        if len(parts) >= 2:
            return parts[-1]

    return ""


def _scrape_reviews_from_html(html: str) -> list[dict]:
    """Parse review data from a HolidayCheck reviews page HTML.

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

        # Handle @graph arrays
        items = ld_data if isinstance(ld_data, list) else [ld_data]
        for item in items:
            if isinstance(item, dict) and item.get("@type") == "Review":
                review = _parse_jsonld_review(item)
                if review.get("id") or review.get("text"):
                    reviews.append(review)

            # Reviews nested inside a Hotel/LodgingBusiness
            if isinstance(item, dict) and "review" in item:
                nested = item["review"]
                if isinstance(nested, list):
                    for r in nested:
                        review = _parse_jsonld_review(r)
                        if review.get("id") or review.get("text"):
                            reviews.append(review)
                elif isinstance(nested, dict):
                    review = _parse_jsonld_review(nested)
                    if review.get("id") or review.get("text"):
                        reviews.append(review)

    if reviews:
        return reviews

    # --- Strategy 2: HTML parsing with common selectors --- #
    # HolidayCheck review containers typically use these patterns
    review_selectors = [
        {"attrs": {"data-test": re.compile(r"review", re.I)}},
        {"class_": re.compile(r"review", re.I)},
        {"attrs": {"itemtype": "http://schema.org/Review"}},
        {"attrs": {"itemprop": "review"}},
    ]

    review_elements: list[Tag] = []
    for sel in review_selectors:
        found = soup.find_all(["div", "article", "section"], **sel)
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
    review_id = _extract_review_id_from_element(element)

    # Rating: look for rating value in text or data attributes
    rating = None
    rating_el = (
        element.find(attrs={"itemprop": "ratingValue"})
        or element.find(class_=re.compile(r"rating", re.I))
        or element.find(attrs={"data-test": re.compile(r"rating", re.I)})
    )
    if rating_el:
        rating = _parse_rating(rating_el.get_text())
        if rating is None:
            rating = _parse_rating(str(rating_el.get("content", "")))

    # Title
    title = ""
    title_el = (
        element.find(attrs={"itemprop": "name"})
        or element.find(class_=re.compile(r"title", re.I))
        or element.find(["h2", "h3", "h4"])
    )
    if title_el:
        title = title_el.get_text(strip=True)

    # Review text: look for review body content
    text = ""
    text_el = (
        element.find(attrs={"itemprop": "reviewBody"})
        or element.find(attrs={"itemprop": "description"})
        or element.find(class_=re.compile(r"review.*text|review.*body|description", re.I))
        or element.find(attrs={"data-test": re.compile(r"body|text|content", re.I)})
    )
    if text_el:
        text = text_el.get_text(strip=True)
    else:
        # Fallback: longest paragraph in the element
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


def hc_get_reviews(
    hotel_url: str,
    max_pages: int = 10,
    timeout: int = 15,
    min_delay: float = 2.5,
    max_delay: float = 5.0,
) -> list[dict]:
    """Fetch reviews from HolidayCheck hotel page with pagination.

    Converts the hotel info URL (/hi/) to the reviews URL (/hrd/) and
    paginates through available review pages.
    """
    reviews_url = _hotel_url_to_reviews_url(hotel_url)
    all_reviews: list[dict] = []
    seen_ids: set[str] = set()

    for page in range(1, max_pages + 1):
        url = reviews_url if page == 1 else f"{reviews_url}?p={page}"
        logger.info("Fetching page %d: %s", page, url)

        try:
            resp = requests.get(url, headers=UA_HEADERS, timeout=timeout)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            logger.warning("HTTP error on page %d: %s", page, exc)
            break
        except requests.RequestException as exc:
            logger.warning("Request error on page %d: %s", page, exc)
            break

        page_reviews = _scrape_reviews_from_html(resp.text)

        if not page_reviews:
            logger.info("No reviews found on page %d, stopping pagination.", page)
            break

        new_on_page = 0
        for review in page_reviews:
            rid = review.get("id", "")
            if not rid:
                # Generate a fallback ID from content hash
                import hashlib
                content = (review.get("text", "") + review.get("title", "")).encode()
                rid = hashlib.md5(content).hexdigest()[:16]
                review["id"] = rid

            if rid not in seen_ids:
                all_reviews.append(review)
                seen_ids.add(rid)
                new_on_page += 1

        logger.info("Page %d: %d reviews (%d new)", page, len(page_reviews), new_on_page)

        if new_on_page == 0:
            logger.info("No new reviews on page %d, stopping.", page)
            break

        # Polite delay between pages
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
        description="Scrape HolidayCheck reviews and classify topics via Ollama."
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
        "--max-pages", type=int, default=10,
        help="Max review pages to fetch (default: 10)",
    )
    p.add_argument(
        "--timeout", type=int, default=15,
        help="HTTP timeout per request in seconds (default: 15)",
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

    hotel_url = HC_URLS.get(ANANEA_HOTEL)
    if not hotel_url:
        logger.error("No HolidayCheck URL configured for %s", ANANEA_HOTEL)
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
    logger.info("Scraping HolidayCheck reviews for %s", ANANEA_HOTEL)

    try:
        raw_reviews = hc_get_reviews(
            hotel_url,
            max_pages=args.max_pages,
            timeout=args.timeout,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
        )
    except Exception as e:
        logger.error("Failed to scrape reviews: %s", e)
        return 1

    logger.info("Scraped %d reviews from HolidayCheck", len(raw_reviews))

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
            "source": "holidaycheck",
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
