#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
booking_reviews.py

Scrape Booking.com review text for hotels and classify each review by topic
using Ollama (mistral:7b).

Reviews are stored in a JSON file and deduplicated by review ID across runs.
Each review is classified into zero or more topics with positive/negative
sentiment.

Booking.com renders reviews client-side via JavaScript, so this scraper uses
Playwright (headless Chromium) to render the page, click "Read all reviews",
and paginate through the review list.

USAGE
-----
Basic (Ollama running locally):
    python src/sites/booking_reviews.py

Skip classification (Ollama not available):
    python src/sites/booking_reviews.py --skip-classification

Reclassify previously unclassified reviews:
    python src/sites/booking_reviews.py --reclassify

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
import time
from datetime import datetime
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, Tag
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

from src.classification import (
    classify_booking_review,
    classify_review,
    is_ollama_available,
)

logger = logging.getLogger(__name__)

# ---------------------- Configuration ---------------------- #

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "hotels.yaml"

DEFAULT_JSON_PATH = str(DATA_DIR / "booking_reviews.json")

ANANEA_HOTEL = "Ananea Castelo Suites Hotel"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _load_booking_urls() -> dict[str, str]:
    """Load Booking hotel URLs from config."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        h["name"]: h["booking_url"]
        for h in cfg["hotels"]
        if h.get("booking_url")
    }


BOOKING_URLS = _load_booking_urls()


# ---------------------- Playwright fetch ---------------------- #

def _dismiss_cookie_banner(page) -> None:
    """Reject non-essential cookies if the banner is present."""
    try:
        reject_btn = page.locator("#onetrust-reject-all-handler")
        if reject_btn.count() > 0:
            reject_btn.click()
            time.sleep(0.5)
    except Exception:
        pass


async def _async_dismiss_cookie_banner(page) -> None:
    """Async version: reject non-essential cookies."""
    import asyncio
    try:
        reject_btn = page.locator("#onetrust-reject-all-handler")
        if await reject_btn.count() > 0:
            await reject_btn.click()
            await asyncio.sleep(0.5)
    except Exception:
        pass


def _sync_fetch_reviews(
    url: str,
    max_pages: int = 10,
    timeout: int = 30000,
    min_delay: float = 2.0,
    max_delay: float = 4.0,
) -> list[str]:
    """Sync: fetch review HTML for each page by navigating, clicking, and paginating."""
    pages_html: list[str] = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="en-GB",
            )
            page = context.new_page()
            try:
                page.goto(url, timeout=timeout, wait_until="networkidle")
                _dismiss_cookie_banner(page)

                # Click "Read all reviews" button
                read_all_btn = page.locator('[data-testid="fr-read-all-reviews"]')
                if read_all_btn.count() == 0:
                    logger.warning("No 'Read all reviews' button found on %s", url)
                    return pages_html

                read_all_btn.click()
                page.wait_for_selector('[data-testid="review-card"]', timeout=15000)
                time.sleep(1)

                # Page 1
                pages_html.append(page.content())
                logger.info("  Page 1: fetched")

                # Paginate through remaining pages
                for page_num in range(2, max_pages + 1):
                    page_btn = page.locator("ol li button", has_text=str(page_num))
                    if page_btn.count() == 0:
                        logger.info("  No page %d button — stopping pagination", page_num)
                        break

                    page_btn.click()
                    time.sleep(random.uniform(min_delay, max_delay))

                    cards = page.locator('[data-testid="review-card"]').count()
                    if cards == 0:
                        logger.info("  Page %d: no review cards — stopping", page_num)
                        break

                    pages_html.append(page.content())
                    logger.info("  Page %d: fetched (%d cards)", page_num, cards)

            finally:
                browser.close()
    except Exception as e:
        logger.warning("Playwright fetch failed for %s: %s", url, e)

    return pages_html


async def _async_fetch_reviews(
    url: str,
    max_pages: int = 10,
    timeout: int = 30000,
    min_delay: float = 2.0,
    max_delay: float = 4.0,
) -> list[str]:
    """Async: fetch review HTML for each page (for Jupyter compatibility)."""
    import asyncio
    pages_html: list[str] = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="en-GB",
            )
            page = await context.new_page()
            try:
                await page.goto(url, timeout=timeout, wait_until="networkidle")
                await _async_dismiss_cookie_banner(page)

                read_all_btn = page.locator('[data-testid="fr-read-all-reviews"]')
                if await read_all_btn.count() == 0:
                    logger.warning("No 'Read all reviews' button found on %s", url)
                    return pages_html

                await read_all_btn.click()
                await page.wait_for_selector('[data-testid="review-card"]', timeout=15000)
                await asyncio.sleep(1)

                pages_html.append(await page.content())
                logger.info("  Page 1: fetched")

                for page_num in range(2, max_pages + 1):
                    page_btn = page.locator("ol li button", has_text=str(page_num))
                    if await page_btn.count() == 0:
                        logger.info("  No page %d button — stopping pagination", page_num)
                        break

                    await page_btn.click()
                    await asyncio.sleep(random.uniform(min_delay, max_delay))

                    cards = await page.locator('[data-testid="review-card"]').count()
                    if cards == 0:
                        logger.info("  Page %d: no review cards — stopping", page_num)
                        break

                    pages_html.append(await page.content())
                    logger.info("  Page %d: fetched (%d cards)", page_num, cards)

            finally:
                await browser.close()
    except Exception as e:
        logger.warning("Playwright fetch failed for %s: %s", url, e)

    return pages_html


def fetch_reviews(
    url: str,
    max_pages: int = 10,
    timeout: int = 30000,
    min_delay: float = 2.0,
    max_delay: float = 4.0,
) -> list[str]:
    """Fetch review page HTML from Booking.com.

    Automatically uses the async Playwright API when called inside an
    existing event loop (e.g. Jupyter notebooks), and the sync API otherwise.

    Returns a list of HTML strings, one per review page.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(
            _async_fetch_reviews(url, max_pages, timeout, min_delay, max_delay)
        )
    else:
        return _sync_fetch_reviews(url, max_pages, timeout, min_delay, max_delay)


# ---------------------- HTML scraping ---------------------- #

_EN_MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _parse_date(text: str) -> str:
    """Parse a date string to YYYY-MM-DD.

    Handles:
      - "Reviewed: 9 November 2025"
      - "9 November 2025"
      - "November 2025" (day defaults to 01)
    """
    if not text:
        return ""
    text = text.strip()

    # Strip "Reviewed:" prefix
    text = re.sub(r"^Reviewed:\s*", "", text, flags=re.I)

    lower = text.lower()
    for month_key, month_num in _EN_MONTHS.items():
        if month_key in lower:
            day_m = re.search(r"(\d{1,2})\b", text)
            year_m = re.search(r"(\d{4})", text)
            if year_m:
                day = day_m.group(1).zfill(2) if day_m else "01"
                return f"{year_m.group(1)}-{month_num}-{day}"

    return text


def _parse_score(text: str) -> float | None:
    """Parse a Booking.com review score.

    Handles English ("Scored 10 10") and Portuguese ("Pontuado com 10 10").
    The duplicate number comes from get_text(" ", strip=True) joining
    the visible score and its aria-label.
    """
    if not text:
        return None
    # English "Scored 9.0" or Portuguese "Pontuado com 9.0"
    m = re.search(r"(?:Scored|Pontuado\s+com)\s+(\d+(?:\.\d+)?)", text, re.I)
    if m:
        try:
            score = float(m.group(1))
            if 0.0 <= score <= 10.0:
                return score
        except ValueError:
            pass
    # Fallback: first number in valid range
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        try:
            score = float(m.group(1))
            if 0.0 <= score <= 10.0:
                return score
        except ValueError:
            pass
    return None


def _parse_reviewer(text: str) -> tuple[str, str]:
    """Parse reviewer name and country from avatar text.

    Input: "Geoffrey\\nAustralia"
    Returns: ("Geoffrey", "Australia")
    """
    if not text:
        return "", ""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    name = lines[0] if lines else ""
    country = lines[1] if len(lines) > 1 else ""
    return name, country


def _scrape_reviews_from_html(html: str) -> list[dict]:
    """Parse review cards from rendered Booking.com HTML.

    Returns a list of raw review dicts.
    """
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    cards = soup.find_all(attrs={"data-testid": "review-card"})
    for card in cards:
        review = _parse_review_card(card)
        reviews.append(review)

    return reviews


def _parse_review_card(card: Tag) -> dict:
    """Extract review fields from a Booking.com review-card element.

    Structure (as of 2026):
      [data-testid="review-card"]
        [data-testid="review-title"]       → title
        [data-testid="review-score"]        → "Scored 10\\n10"
        [data-testid="review-positive-text"] → positive comment
        [data-testid="review-negative-text"] → negative comment
        [data-testid="review-date"]          → "Reviewed: 9 November 2025"
        [data-testid="review-stay-date"]     → "October 2025"
        [data-testid="review-room-name"]     → room type
        [data-testid="review-traveler-type"] → "Couple"
        [data-testid="review-num-nights"]    → "2 nights ·"
        [data-testid="review-avatar"]        → "Geoffrey\\nAustralia"
    """
    def _text(testid: str) -> str:
        el = card.find(attrs={"data-testid": testid})
        return el.get_text(" ", strip=True) if el else ""

    title = _text("review-title")
    score_text = _text("review-score")
    positive_text = _text("review-positive-text")
    negative_text = _text("review-negative-text")
    review_date = _text("review-date")
    stay_date = _text("review-stay-date")
    room_name = _text("review-room-name")
    traveler_type = _text("review-traveler-type")
    num_nights = _text("review-num-nights")
    avatar_text = _text("review-avatar")

    rating = _parse_score(score_text)
    published_date = _parse_date(review_date)
    author_name, country = _parse_reviewer(avatar_text)

    # Combine positive and negative text for classification
    parts = []
    if positive_text:
        parts.append(positive_text)
    if negative_text:
        parts.append(negative_text)
    combined_text = "\n".join(parts)

    # Generate a deterministic ID from author + date + title
    id_seed = f"{author_name}_{published_date}_{title}"
    review_id = hashlib.sha256(id_seed.encode()).hexdigest()[:16]

    return {
        "id": review_id,
        "rating": rating,
        "title": title,
        "text": combined_text,
        "positive_text": positive_text,
        "negative_text": negative_text,
        "published_date": published_date,
        "stay_date": stay_date,
        "room_name": room_name,
        "traveler_type": traveler_type or "Unknown",
        "num_nights": num_nights,
        "author_name": author_name,
        "country": country or "Unknown",
    }


def booking_get_reviews(
    hotel_url: str,
    max_pages: int = 10,
    timeout: int = 30,
    min_delay: float = 2.0,
    max_delay: float = 4.0,
) -> list[dict]:
    """Fetch all reviews from a Booking.com hotel page.

    Opens the reviews dialog, paginates through pages, and extracts
    review data from each page.
    """
    timeout_ms = timeout * 1000

    logger.info("Fetching Booking.com reviews: %s", hotel_url)

    pages_html = fetch_reviews(
        hotel_url,
        max_pages=max_pages,
        timeout=timeout_ms,
        min_delay=min_delay,
        max_delay=max_delay,
    )

    if not pages_html:
        logger.warning("Failed to fetch any review pages.")
        return []

    all_reviews: list[dict] = []
    seen_ids: set[str] = set()

    for html in pages_html:
        page_reviews = _scrape_reviews_from_html(html)
        for review in page_reviews:
            rid = review.get("id", "")
            if rid and rid not in seen_ids:
                all_reviews.append(review)
                seen_ids.add(rid)

    logger.info("Fetched %d unique reviews across %d pages", len(all_reviews), len(pages_html))
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
        description="Scrape Booking.com reviews and classify topics via Ollama."
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
        "--reclassify-all", action="store_true",
        help="Reclassify ALL reviews, even previously classified ones.",
    )
    p.add_argument(
        "--max-pages", type=int, default=10,
        help="Max review pages to fetch (default: 10)",
    )
    p.add_argument(
        "--timeout", type=int, default=30,
        help="Page load timeout in seconds (default: 30)",
    )
    p.add_argument(
        "--min-delay", type=float, default=2.0,
        help="Min delay (s) between page requests (default: 2.0)",
    )
    p.add_argument(
        "--max-delay", type=float, default=4.0,
        help="Max delay (s) between page requests (default: 4.0)",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    hotel_url = BOOKING_URLS.get(ANANEA_HOTEL)
    if not hotel_url:
        logger.error("No Booking URL configured for %s", ANANEA_HOTEL)
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
    if args.reclassify or args.reclassify_all:
        if not ollama_ok:
            logger.error("Cannot reclassify: Ollama is not available.")
            return 1
        reclassified = 0
        for review in existing_reviews:
            needs_reclassify = args.reclassify_all or not review.get("classified", False)
            if needs_reclassify and (
                review.get("positive_text") or review.get("negative_text")
            ):
                try:
                    topics = classify_booking_review(
                        review.get("positive_text", ""),
                        review.get("negative_text", ""),
                        args.ollama_url,
                    )
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
    logger.info("Scraping Booking.com reviews for %s", ANANEA_HOTEL)

    try:
        raw_reviews = booking_get_reviews(
            hotel_url,
            max_pages=args.max_pages,
            timeout=args.timeout,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
        )
    except Exception as e:
        logger.error("Failed to scrape reviews: %s", e)
        return 1

    logger.info("Scraped %d reviews from Booking.com", len(raw_reviews))

    existing_ids = {str(r["id"]) for r in existing_reviews}
    new_reviews: list[dict] = []

    for raw in raw_reviews:
        review_id = str(raw.get("id", ""))
        if not review_id or review_id in existing_ids:
            logger.debug("Skipping duplicate review %s", review_id)
            continue

        text = raw.get("text", "")
        positive_text = raw.get("positive_text", "")
        negative_text = raw.get("negative_text", "")
        topics: list[dict] = []
        classified = False

        if ollama_ok and (positive_text or negative_text):
            try:
                topics = classify_booking_review(
                    positive_text, negative_text, args.ollama_url,
                )
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
            "source": "booking",
            "rating": raw.get("rating"),
            "title": raw.get("title", ""),
            "text": text,
            "positive_text": raw.get("positive_text", ""),
            "negative_text": raw.get("negative_text", ""),
            "published_date": raw.get("published_date", ""),
            "stay_date": raw.get("stay_date", ""),
            "room_name": raw.get("room_name", ""),
            "traveler_type": raw.get("traveler_type", "") or "Unknown",
            "author_name": raw.get("author_name", ""),
            "country": raw.get("country", "") or "Unknown",
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
