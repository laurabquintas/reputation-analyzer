#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expedia_reviews.py

Scrape Expedia review text for hotels and classify each review by topic
using Ollama (qwen2.5:7b).

Reviews are stored in a JSON file and deduplicated by review ID across runs.
Each review is classified into zero or more topics with positive/negative
sentiment.

Expedia is a JavaScript SPA, so this scraper uses Playwright (headless
Chromium) to render the page and extract reviews from the rendered DOM.

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
playwright, beautifulsoup4, PyYAML
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, Tag
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

from src.classification import (
    classify_review,
    is_ollama_available,
    warm_up_model,
)
from src.sites.expedia import (
    USER_AGENTS,
    _expedia_url_candidates,
)

logger = logging.getLogger(__name__)

# ---------------------- Language / country detection ---------------------- #

_EXPEDIA_TRIP_TYPE_MAP: dict[str, str] = {
    "couple": "Couple",
    "family": "Family",
    "group": "Friends",
    "solo": "Solo",
    "business": "Business",
}

_LANG_MARKERS: dict[str, tuple[str, ...]] = {
    "French": ("très", "avec", "nous", "était", "pour", "hôtel"),
    "German": ("sehr", "wir", "nicht", "auch", "waren", "aber"),
    "Portuguese": ("muito", "estava", "para", "uma", "mais", "foi"),
    "Spanish": ("muy", "pero", "para", "una", "estaba", "todo"),
    "Italian": ("molto", "anche", "erano", "una", "sono", "stato"),
    "Dutch": ("heel", "maar", "waren", "voor", "niet", "onze"),
    "Swedish": ("mycket", "hotell", "inte", "var", "men", "från"),
}

_LANG_TO_COUNTRY: dict[str, str] = {
    "French": "France",
    "German": "Germany",
    "Portuguese": "Portugal",
    "Spanish": "Spain",
    "Italian": "Italy",
    "Dutch": "Netherlands",
    "Swedish": "Sweden",
}


def _detect_country(text: str) -> str:
    """Infer reviewer country from review text language using keyword markers."""
    lower = text.lower()
    for lang, markers in _LANG_MARKERS.items():
        if sum(1 for m in markers if m in lower) >= 2:
            return _LANG_TO_COUNTRY.get(lang, "Unknown")
    return "England/USA"


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


def _hotel_url_to_base_url(hotel_url: str) -> str:
    """Strip any ``pwaDialog`` query params to get the base hotel page URL."""
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

    parsed = urlsplit(hotel_url)
    params = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
              if not k.lower().startswith("pwadialog")]
    new_query = urlencode(params)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment))


# ---------------------- Playwright fetch ---------------------- #

async def _async_fetch_reviews_page(url: str, timeout: int = 30000) -> str | None:
    """Async version of fetch_reviews_page for use inside event loops (e.g. Jupyter)."""
    base_url = _hotel_url_to_base_url(url)
    candidates = _expedia_url_candidates(base_url)

    for candidate_url in candidates:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    locale="en-US",
                )
                page = await context.new_page()
                try:
                    await page.goto(candidate_url, timeout=timeout, wait_until="networkidle")

                    reviews_btn = page.locator('[data-stid="reviews-link"]')
                    if await reviews_btn.count() == 0:
                        logger.warning("No reviews button found on %s", candidate_url)
                        continue

                    await reviews_btn.click()
                    await page.wait_for_selector(
                        '[data-stid="product-reviews-list-item"]',
                        timeout=15000,
                    )

                    html = await page.content()
                    if candidate_url != base_url:
                        logger.info("Fetched via fallback URL: %s", candidate_url)
                    return html
                finally:
                    await browser.close()
        except Exception as e:
            logger.warning("Playwright fetch failed for %s: %s", candidate_url, e)
            continue

    logger.error("Failed to fetch reviews on all Expedia URL variants")
    return None


def _sync_fetch_reviews_page(url: str, timeout: int = 30000) -> str | None:
    """Sync version of fetch_reviews_page (used outside event loops)."""
    base_url = _hotel_url_to_base_url(url)
    candidates = _expedia_url_candidates(base_url)

    for candidate_url in candidates:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    locale="en-US",
                )
                page = context.new_page()
                try:
                    page.goto(candidate_url, timeout=timeout, wait_until="networkidle")

                    reviews_btn = page.locator('[data-stid="reviews-link"]')
                    if reviews_btn.count() == 0:
                        logger.warning("No reviews button found on %s", candidate_url)
                        continue

                    reviews_btn.click()

                    page.wait_for_selector(
                        '[data-stid="product-reviews-list-item"]',
                        timeout=15000,
                    )

                    html = page.content()
                    if candidate_url != base_url:
                        logger.info("Fetched via fallback URL: %s", candidate_url)
                    return html
                finally:
                    browser.close()
        except Exception as e:
            logger.warning("Playwright fetch failed for %s: %s", candidate_url, e)
            continue

    logger.error("Failed to fetch reviews on all Expedia URL variants")
    return None


def fetch_reviews_page(url: str, timeout: int = 30000) -> str | None:
    """Render an Expedia hotel page and click the reviews button.

    Automatically uses the async Playwright API when called inside an
    existing event loop (e.g. Jupyter notebooks), and the sync API otherwise.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Inside an event loop (Jupyter) — use nest_asyncio or run coroutine
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(_async_fetch_reviews_page(url, timeout))
    else:
        return _sync_fetch_reviews_page(url, timeout)


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
    """Parse review data from a rendered Expedia page HTML.

    Looks for ``data-stid="product-reviews-list-item"`` elements that
    Expedia renders inside the reviews dialog.

    Returns a list of raw review dicts with keys:
      id, rating, title, text, travel_date, author_name
    """
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # Expedia review items
    review_items = soup.find_all(attrs={"data-stid": "product-reviews-list-item"})
    for item in review_items:
        review = _parse_expedia_review_item(item)
        reviews.append(review)

    return reviews


def _parse_expedia_review_item(item: Tag) -> dict:
    """Extract review fields from an Expedia ``product-reviews-list-item``.

    Structure (as of 2026):
      <article id="<review-id>">
        <h3 aria-label="8 out of 10 Good">8/10 Good</h3>
        <h4>Author Name</h4>
        <div class="uitk-type-300">Travelled with group</div>
        <div class="uitk-type-300">6 Oct 2025</div>
        <span>Liked: cleanliness, ...</span>
        <div class="uitk-expando-peek">review text...</div>
      </article>
    """
    # Review ID from the <article> element
    article = item.find("article")
    review_id = article.get("id", "") if article else ""
    if not review_id:
        review_id = _extract_review_id(item)

    # Rating from h3 (e.g. "8/10 Good")
    rating = None
    h3 = item.find("h3")
    if h3:
        h3_text = h3.get_text(strip=True)
        m = re.match(r"(\d+(?:\.\d+)?)/10", h3_text)
        if m:
            rating = _parse_rating(m.group(1))

    # Author from h4
    author_name = ""
    h4 = item.find("h4")
    if h4:
        author_name = h4.get_text(strip=True)

    # Date and travel type from uitk-type-300 divs
    travel_date = ""
    trip_type = ""
    type_300_divs = item.find_all("div", class_="uitk-type-300")
    date_pattern = re.compile(
        r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\w*\s+\d{4}",
        re.I,
    )
    for div in type_300_divs:
        div_text = div.get_text(strip=True)
        if date_pattern.search(div_text):
            travel_date = _parse_date(div_text)
        elif not trip_type:
            lower = div_text.lower()
            for key, value in _EXPEDIA_TRIP_TYPE_MAP.items():
                if key in lower:
                    trip_type = value
                    break

    # Review text from uitk-expando-peek
    text = ""
    expando = item.find(class_="uitk-expando-peek")
    if expando:
        # Get text from spans inside, or fallback to full text
        spans = expando.find_all("span", class_="uitk-type-300")
        if spans:
            text = " ".join(s.get_text(strip=True) for s in spans)
        else:
            text = expando.get_text(" ", strip=True)

    # Title from h3 quality label (e.g. "Good", "Excellent")
    title = ""
    if h3:
        h3_text = h3.get_text(strip=True)
        m = re.match(r"\d+(?:\.\d+)?/10\s+(.*)", h3_text)
        if m:
            title = m.group(1).strip()

    country = _detect_country(text) if text else "Unknown"

    return {
        "id": review_id,
        "rating": rating,
        "title": title,
        "text": text,
        "travel_date": travel_date,
        "author_name": author_name,
        "trip_type": trip_type or "Unknown",
        "country": country,
    }


def expedia_get_reviews(
    hotel_url: str,
    max_pages: int = 5,
    timeout: int = 30,
    retries: int = 2,
    min_delay: float = 2.5,
    max_delay: float = 5.0,
) -> list[dict]:
    """Fetch reviews from Expedia hotel page using Playwright.

    Navigates to the hotel page, clicks the reviews button, and
    extracts reviews from the rendered dialog.
    """
    all_reviews: list[dict] = []
    seen_ids: set[str] = set()
    timeout_ms = timeout * 1000

    logger.info("Fetching Expedia reviews: %s", hotel_url)

    html = fetch_reviews_page(hotel_url, timeout=timeout_ms)
    if html is None:
        logger.warning("Failed to fetch reviews page.")
        return all_reviews

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

    logger.info("Fetched %d unique reviews", len(all_reviews))
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

    # Backfill missing fields for old reviews
    for r in existing_reviews:
        if "trip_type" not in r:
            r["trip_type"] = "Unknown"
        if "country" not in r:
            r["country"] = "Unknown"

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
    if ollama_ok:
        logger.info("Warming up Ollama model...")
        warm_up_model(args.ollama_url)

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
            "country": raw.get("country", "") or "Unknown",
            "trip_type": raw.get("trip_type", "") or "Unknown",
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
