#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
holidaycheck_reviews.py

Scrape HolidayCheck review text for Ananea Castelo Suites Hotel and classify
each review by topic using Ollama (mistral:7b).

Reviews are stored in a JSON file and deduplicated by review ID across runs.
Each review is classified into zero or more topics with positive/negative
sentiment.

HolidayCheck does not offer a public review API, so this scraper uses
Playwright to load the hotel reviews page, sort by newest, and then visits
each individual review detail page (``/hrd/`` path) to extract the full
review text — including all topic sections (Allgemein, Zimmer, Service, etc.).

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
from time import sleep

import yaml
from bs4 import BeautifulSoup, Tag
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

from src.classification import (
    classify_holidaycheck_review,
    is_ollama_available,
    warm_up_model,
)

logger = logging.getLogger(__name__)

# ---------------------- Configuration ---------------------- #

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "hotels.yaml"

DEFAULT_JSON_PATH = str(DATA_DIR / "holidaycheck_reviews.json")

ANANEA_HOTEL = "Ananea Castelo Suites Hotel"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# Known HolidayCheck review section headings (German).
HC_SECTION_HEADINGS = [
    "Allgemein",
    "Zimmer",
    "Service",
    "Lage & Umgebung",
    "Lage",
    "Gastronomie",
    "Restaurant & Bars",
    "Sport & Unterhaltung",
    "Hotel",
    "Pool",
    "Strand",
    "Verkehrsanbindung",
    "Preis-Leistung",
]

# German → English translation for reviewer country names.
_COUNTRY_DE_TO_EN: dict[str, str] = {
    "Deutschland": "Germany",
    "Österreich": "Austria",
    "Schweiz": "Switzerland",
    "Niederlande": "Netherlands",
    "Belgien": "Belgium",
    "Frankreich": "France",
    "Italien": "Italy",
    "Spanien": "Spain",
    "Portugal": "Portugal",
    "Großbritannien": "United Kingdom",
    "Irland": "Ireland",
    "Dänemark": "Denmark",
    "Schweden": "Sweden",
    "Norwegen": "Norway",
    "Finnland": "Finland",
    "Polen": "Poland",
    "Tschechien": "Czech Republic",
    "Ungarn": "Hungary",
    "Rumänien": "Romania",
    "Griechenland": "Greece",
    "Türkei": "Turkey",
    "Russland": "Russia",
    "Luxemburg": "Luxembourg",
    "Kroatien": "Croatia",
    "Serbien": "Serbia",
    "Bulgarien": "Bulgaria",
    "Slowenien": "Slovenia",
    "Slowakei": "Slovakia",
    "Estland": "Estonia",
    "Lettland": "Latvia",
    "Litauen": "Lithuania",
    "USA": "USA",
    "Kanada": "Canada",
    "Brasilien": "Brazil",
    "Australien": "Australia",
    "China": "China",
    "Japan": "Japan",
    "Indien": "India",
    "Südafrika": "South Africa",
    "Israel": "Israel",
    "Ägypten": "Egypt",
    "Marokko": "Morocco",
}

# German → English translation for trip type.
_TRIP_TYPE_DE_TO_EN: dict[str, str] = {
    "Paar": "Couple",
    "Freunde": "Friends",
    "Familie": "Family",
    "Allein/Geschäftlich": "Solo/Business",
    "Allein": "Solo",
    "Geschäftlich": "Business",
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
    """Convert a hotel info URL to the reviews URL.

    Hotel info:  /hi/{slug}/{uuid}
    Reviews:     /hr/bewertungen-{slug}/{uuid}
    """
    # Extract slug and uuid from /hi/{slug}/{uuid}
    parts = hotel_url.split("/hi/", 1)
    if len(parts) == 2:
        remainder = parts[1]  # e.g. "ananea-castelo-suites-algarve/069563af-..."
        return f"{parts[0]}/hr/bewertungen-{remainder}"
    return hotel_url


# ---------------------- HTML scraping ---------------------- #

def _normalize_rating(score: float | None) -> float | None:
    """Normalize a HolidayCheck rating to the 0-6 scale.

    The structured data sometimes reports scores on a 0-10 scale,
    but HolidayCheck's real scale is 0-6.
    """
    if score is None:
        return None
    if score > 6.0:
        return round(score * 0.6, 1)
    return round(score, 1)


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


def _extract_all_section_texts(soup: BeautifulSoup) -> str:
    """Extract review text from all topic sections on a detail page.

    HolidayCheck review detail pages show the review split into sections
    (Allgemein, Zimmer, Service, Lage, etc.).  Each section lives in a
    container whose first child holds the heading and whose second child
    holds the review text.

    Returns a combined string with ``[SectionName]`` labels, e.g.::

        [Allgemein] Ein neues Hotel…

        [Zimmer] Moderne Ausstattung…
    """
    parts: list[str] = []

    for heading_text in HC_SECTION_HEADINGS:
        heading_el = soup.find(string=heading_text)
        if not heading_el:
            continue

        # Walk up to the section container: text → <div heading> → <div wrapper> → <div section>
        section = heading_el
        for _ in range(3):
            if section.parent:
                section = section.parent
            else:
                break

        # The section container's second child div holds the review text
        children = [c for c in section.children if hasattr(c, "name") and c.name]
        if len(children) < 2:
            continue

        text = children[1].get_text(strip=True)
        if not text:
            continue

        # Skip sections that only contain rating labels without real prose
        # (e.g. "ZimmergrößeSehr gutSchlafqualitätSehr gut")
        if len(text) < 20 and re.match(r"^[A-ZÄÖÜa-zäöüß\s]+$", text):
            continue

        parts.append(f"[{heading_text}] {text}")

    return "\n\n".join(parts)


def _extract_reviewer_country(soup: BeautifulSoup) -> str:
    """Extract reviewer country from a detail page (translated to English).

    HolidayCheck shows e.g. "Aus Deutschland" in a <span> under the
    reviewer's name.  We strip the "Aus " prefix, translate to English,
    and return the country name.
    """
    span = soup.find("span", string=re.compile(r"^Aus\s+"))
    if span:
        text = span.get_text(strip=True)
        country_de = re.sub(r"^Aus\s+", "", text)
        if country_de:
            return _COUNTRY_DE_TO_EN.get(country_de, country_de)
    return ""


def _extract_trip_type(soup: BeautifulSoup) -> str:
    """Extract trip type from a detail page (translated to English).

    HolidayCheck shows e.g. "Verreist als Freunde • August 2025 • Strand"
    in a <span>.  We extract the portion after "Verreist als " and before
    the first bullet/dot separator (e.g. "Freunde" → "Friends").
    """
    span = soup.find("span", string=re.compile(r"^Verreist\s+als\s+"))
    if span:
        text = span.get_text(strip=True)
        # Remove "Verreist als " prefix
        after_prefix = re.sub(r"^Verreist\s+als\s+", "", text)
        # Take everything before the first bullet (•) or middle-dot (·) separator
        trip_type_de = re.split(r"[•·]", after_prefix)[0].strip()
        if trip_type_de:
            return _TRIP_TYPE_DE_TO_EN.get(trip_type_de, trip_type_de)
    return ""


def _extract_title_from_html(soup: BeautifulSoup) -> str:
    """Extract review title from the detail page HTML.

    On HolidayCheck detail pages the title sits inside
    ``<div class="hotel-review-header">`` as a ``<span>`` child (child #4).
    Falls back to generic heading selectors.
    """
    header = soup.find("div", class_="hotel-review-header")
    if header:
        # The title is the <span> child whose text is longer and is not the author/country/trip
        children = [c for c in header.children if hasattr(c, "name") and c.name]
        for child in children:
            if child.name == "span":
                text = child.get_text(strip=True)
                # Skip avatar (empty), author/age (short with parentheses), country, trip
                if (
                    text
                    and len(text) > 15
                    and not text.startswith("Aus ")
                    and not text.startswith("Verreist ")
                    and "/ 6" not in text
                ):
                    return text

    # Fallback: generic heading selectors
    for sel in [
        {"attrs": {"itemprop": "name"}},
        {"attrs": {"itemprop": "headline"}},
    ]:
        el = soup.find(**sel)
        if el:
            text = el.get_text(strip=True)
            if text and len(text) > 5:
                return text
    return ""


def _extract_rating_from_html(soup: BeautifulSoup) -> float | None:
    """Extract review rating from the detail page HTML.

    On HolidayCheck detail pages the review rating appears as
    ``X,X / 6`` inside ``<div class="hotel-review-header">``.
    The hotel-level overall score lives in a separate
    ``<div class="HotelReviewBar ...">`` and must be excluded.
    """
    header = soup.find("div", class_="hotel-review-header")
    if header:
        # Look for text matching "X,X" followed by "/ 6"
        rating_text = header.find(string=re.compile(r"[0-6][.,]\d"))
        if rating_text:
            return _parse_rating(rating_text.strip())

    # Fallback: look for itemprop="ratingValue" (sometimes present)
    el = soup.find(attrs={"itemprop": "ratingValue"})
    if el:
        return _parse_rating(el.get("content", "") or el.get_text(strip=True))

    return None


def _extract_author_from_html(soup: BeautifulSoup) -> str:
    """Extract reviewer name from the detail page HTML.

    The ``hotel-review-header`` contains a ``<span>`` with text like
    ``"Frank(56-60)"``.  We strip the age-range suffix.
    """
    header = soup.find("div", class_="hotel-review-header")
    if header:
        children = [c for c in header.children if hasattr(c, "name") and c.name]
        for child in children:
            if child.name == "span":
                text = child.get_text(strip=True)
                # Match name optionally followed by (age-range)
                if text and re.match(r"^[A-ZÄÖÜa-zäöüß]", text) and "/ 6" not in text:
                    # Strip age range like "(56-60)" or "(26-30)"
                    name = re.sub(r"\s*\(\d+-\d+\)\s*$", "", text).strip()
                    if name and len(name) < 60 and not name.startswith("Aus ") and not name.startswith("Verreist "):
                        return name
    return ""


def _extract_travel_date_from_html(soup: BeautifulSoup) -> str:
    """Extract travel date from the detail page HTML.

    The trip-type string (e.g. "Verreist als Paar • September 2025 • Strand")
    contains a German month + year segment.  We parse it with ``_parse_date``.
    """
    span = soup.find("span", string=re.compile(r"^Verreist\s+als\s+"))
    if not span:
        # Also check <div> — some pages use div instead of span
        span = soup.find("div", string=re.compile(r"^Verreist\s+als\s+"))
    if span:
        text = span.get_text(strip=True)
        # Split by bullet separators and look for a month+year segment
        parts = re.split(r"[•·]", text)
        for part in parts:
            part = part.strip()
            # Match German month + year (e.g. "September 2025", "August 2025")
            if re.search(r"(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}", part, re.IGNORECASE):
                return _parse_date(part)
    return ""


def _extract_review_detail_links(html: str, base_url: str = "https://www.holidaycheck.de") -> list[str]:
    """Extract individual review detail page links from a reviews listing page.

    Looks for links matching the /hrd/ pattern (individual review pages).
    """
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()

    for a_tag in soup.find_all("a", href=re.compile(r"/hrd/")):
        href = a_tag.get("href", "")
        if not href:
            continue
        if href.startswith("/"):
            href = base_url + href
        if href not in seen:
            links.append(href)
            seen.add(href)

    return links


def _scrape_full_review_from_html(html: str) -> dict:
    """Parse a single review detail page HTML and extract all section texts."""
    soup = BeautifulSoup(html, "html.parser")

    # Get metadata (id, rating, title, date, author) from JSON-LD / HTML
    reviews = _scrape_reviews_from_html(html)
    review = reviews[0] if reviews else {}

    # Fallback: metadata is often missing from the detail page's JSON-LD.
    # Extract from the visible HTML when absent.
    if not review.get("title"):
        review["title"] = _extract_title_from_html(soup)
    if review.get("rating") is None:
        review["rating"] = _extract_rating_from_html(soup)
    if not review.get("author_name"):
        review["author_name"] = _extract_author_from_html(soup)
    if not review.get("travel_date"):
        review["travel_date"] = _extract_travel_date_from_html(soup)

    # Extract the full text from all topic sections
    all_sections_text = _extract_all_section_texts(soup)
    if all_sections_text and len(all_sections_text) > len(review.get("text", "")):
        review["text"] = all_sections_text

    # Extract reviewer country and trip type from the detail page
    review["country"] = _extract_reviewer_country(soup) or "Unknown"
    review["trip_type"] = _extract_trip_type(soup) or "Unknown"

    return review


def _dismiss_cookie_banner(page) -> None:
    """Dismiss the HolidayCheck cookie consent banner if present."""
    try:
        # Common HolidayCheck cookie consent selectors
        for selector in [
            "button[data-testid='uc-accept-all-button']",
            "#onetrust-reject-all-handler",
            "button.uc-btn-accept",
            "button[id*='accept']",
        ]:
            btn = page.locator(selector)
            if btn.count() > 0:
                btn.first.click()
                sleep(0.5)
                return
    except Exception:
        pass


async def _async_dismiss_cookie_banner(page) -> None:
    """Async version: dismiss HolidayCheck cookie consent banner."""
    import asyncio
    try:
        for selector in [
            "button[data-testid='uc-accept-all-button']",
            "#onetrust-reject-all-handler",
            "button.uc-btn-accept",
            "button[id*='accept']",
        ]:
            btn = page.locator(selector)
            if await btn.count() > 0:
                await btn.first.click()
                await asyncio.sleep(0.5)
                return
    except Exception:
        pass


def _sync_hc_get_reviews(
    hotel_url: str,
    max_pages: int = 10,
    timeout: int = 30,
    min_delay: float = 2.5,
    max_delay: float = 5.0,
    sort_newest: bool = True,
) -> list[dict]:
    """Fetch reviews from HolidayCheck using Playwright.

    1. Opens the reviews listing page (/hr/bewertungen-...) sorted by
       newest (``?sort=entrydate``).
    2. Extracts individual review detail links (/hrd/...).
    3. Navigates to each detail page to get the full review text
       including all topic sections (Allgemein, Zimmer, Service, etc.).
    4. Paginates through listing pages.
    """
    reviews_url = _hotel_url_to_reviews_url(hotel_url)
    all_reviews: list[dict] = []
    seen_ids: set[str] = set()
    timeout_ms = timeout * 1000

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="de-DE",
            )
            page = context.new_page()

            try:
                for page_num in range(1, max_pages + 1):
                    # Build listing URL with sort parameter
                    if sort_newest:
                        url = (
                            f"{reviews_url}?sort=entrydate"
                            if page_num == 1
                            else f"{reviews_url}?sort=entrydate&p={page_num}"
                        )
                    else:
                        url = (
                            reviews_url
                            if page_num == 1
                            else f"{reviews_url}?p={page_num}"
                        )

                    logger.info("Fetching listing page %d: %s", page_num, url)
                    page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")

                    if page_num == 1:
                        _dismiss_cookie_banner(page)

                    sleep(random.uniform(1, 2))

                    listing_html = page.content()
                    count_before = len(all_reviews)

                    detail_links = _extract_review_detail_links(listing_html)

                    if detail_links:
                        logger.info(
                            "Found %d review detail links on page %d",
                            len(detail_links), page_num,
                        )
                        for detail_url in detail_links:
                            sleep(random.uniform(min_delay, max_delay))
                            try:
                                page.goto(detail_url, timeout=timeout_ms, wait_until="domcontentloaded")
                                sleep(random.uniform(1, 2))
                                detail_html = page.content()
                                full_review = _scrape_full_review_from_html(detail_html)
                                if full_review and full_review.get("text"):
                                    rid = full_review.get("id", "")
                                    if not rid:
                                        content = (
                                            full_review.get("text", "")
                                            + full_review.get("title", "")
                                        ).encode()
                                        rid = hashlib.md5(content).hexdigest()[:16]
                                        full_review["id"] = rid
                                    if rid not in seen_ids:
                                        all_reviews.append(full_review)
                                        seen_ids.add(rid)
                                        logger.info(
                                            "  Fetched full review: %s",
                                            full_review.get("title", rid)[:50],
                                        )
                            except Exception as exc:
                                logger.warning(
                                    "  Failed to fetch detail page %s: %s",
                                    detail_url, exc,
                                )
                    else:
                        # Fallback: use reviews from the listing page (truncated)
                        page_reviews = _scrape_reviews_from_html(listing_html)
                        for review in page_reviews:
                            rid = review.get("id", "")
                            if not rid:
                                content = (
                                    review.get("text", "")
                                    + review.get("title", "")
                                ).encode()
                                rid = hashlib.md5(content).hexdigest()[:16]
                                review["id"] = rid
                            if rid not in seen_ids:
                                all_reviews.append(review)
                                seen_ids.add(rid)

                    new_on_page = len(all_reviews) - count_before
                    logger.info(
                        "Page %d: %d new reviews (%d total)",
                        page_num, new_on_page, len(all_reviews),
                    )

                    if new_on_page == 0:
                        logger.info("No new reviews on page %d, stopping.", page_num)
                        break

            finally:
                browser.close()

    except Exception as exc:
        logger.warning("Playwright fetch failed: %s", exc)

    logger.info("Fetched %d unique reviews total", len(all_reviews))
    return all_reviews


async def _async_hc_get_reviews(
    hotel_url: str,
    max_pages: int = 10,
    timeout: int = 30,
    min_delay: float = 2.5,
    max_delay: float = 5.0,
    sort_newest: bool = True,
) -> list[dict]:
    """Async: fetch reviews from HolidayCheck (for Jupyter compatibility)."""
    import asyncio

    reviews_url = _hotel_url_to_reviews_url(hotel_url)
    all_reviews: list[dict] = []
    seen_ids: set[str] = set()
    timeout_ms = timeout * 1000

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="de-DE",
            )
            page = await context.new_page()

            try:
                for page_num in range(1, max_pages + 1):
                    if sort_newest:
                        url = (
                            f"{reviews_url}?sort=entrydate"
                            if page_num == 1
                            else f"{reviews_url}?sort=entrydate&p={page_num}"
                        )
                    else:
                        url = (
                            reviews_url
                            if page_num == 1
                            else f"{reviews_url}?p={page_num}"
                        )

                    logger.info("Fetching listing page %d: %s", page_num, url)
                    await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")

                    if page_num == 1:
                        await _async_dismiss_cookie_banner(page)

                    await asyncio.sleep(random.uniform(1, 2))

                    listing_html = await page.content()
                    count_before = len(all_reviews)

                    detail_links = _extract_review_detail_links(listing_html)

                    if detail_links:
                        logger.info(
                            "Found %d review detail links on page %d",
                            len(detail_links), page_num,
                        )
                        for detail_url in detail_links:
                            await asyncio.sleep(random.uniform(min_delay, max_delay))
                            try:
                                await page.goto(
                                    detail_url, timeout=timeout_ms,
                                    wait_until="domcontentloaded",
                                )
                                await asyncio.sleep(random.uniform(1, 2))
                                detail_html = await page.content()
                                full_review = _scrape_full_review_from_html(detail_html)
                                if full_review and full_review.get("text"):
                                    rid = full_review.get("id", "")
                                    if not rid:
                                        content = (
                                            full_review.get("text", "")
                                            + full_review.get("title", "")
                                        ).encode()
                                        rid = hashlib.md5(content).hexdigest()[:16]
                                        full_review["id"] = rid
                                    if rid not in seen_ids:
                                        all_reviews.append(full_review)
                                        seen_ids.add(rid)
                                        logger.info(
                                            "  Fetched full review: %s",
                                            full_review.get("title", rid)[:50],
                                        )
                            except Exception as exc:
                                logger.warning(
                                    "  Failed to fetch detail page %s: %s",
                                    detail_url, exc,
                                )
                    else:
                        page_reviews = _scrape_reviews_from_html(listing_html)
                        for review in page_reviews:
                            rid = review.get("id", "")
                            if not rid:
                                content = (
                                    review.get("text", "")
                                    + review.get("title", "")
                                ).encode()
                                rid = hashlib.md5(content).hexdigest()[:16]
                                review["id"] = rid
                            if rid not in seen_ids:
                                all_reviews.append(review)
                                seen_ids.add(rid)

                    new_on_page = len(all_reviews) - count_before
                    logger.info(
                        "Page %d: %d new reviews (%d total)",
                        page_num, new_on_page, len(all_reviews),
                    )

                    if new_on_page == 0:
                        logger.info("No new reviews on page %d, stopping.", page_num)
                        break

            finally:
                await browser.close()

    except Exception as exc:
        logger.warning("Playwright fetch failed: %s", exc)

    logger.info("Fetched %d unique reviews total", len(all_reviews))
    return all_reviews


def hc_get_reviews(
    hotel_url: str,
    max_pages: int = 10,
    timeout: int = 30,
    min_delay: float = 2.5,
    max_delay: float = 5.0,
    sort_newest: bool = True,
) -> list[dict]:
    """Fetch reviews from HolidayCheck using Playwright.

    Automatically uses the async Playwright API when called inside an
    existing event loop (e.g. Jupyter notebooks), and the sync API otherwise.

    1. Opens the reviews listing page sorted by newest (``?sort=entrydate``).
    2. Navigates to each detail page to extract all topic sections.
    3. Paginates through listing pages.
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
            _async_hc_get_reviews(
                hotel_url, max_pages, timeout, min_delay, max_delay, sort_newest,
            )
        )
    else:
        return _sync_hc_get_reviews(
            hotel_url, max_pages, timeout, min_delay, max_delay, sort_newest,
        )


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
        "--sort-newest", action="store_true", default=True,
        help="Sort reviews by newest first (default: True).",
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

    # Backfill missing country / trip_type for older reviews
    for r in existing_reviews:
        if not r.get("country"):
            r["country"] = "Unknown"
        if not r.get("trip_type"):
            r["trip_type"] = "Unknown"

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
                    topics = classify_holidaycheck_review(review["text"], args.ollama_url)
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
            sort_newest=args.sort_newest,
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
                topics = classify_holidaycheck_review(text, args.ollama_url)
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
            "rating": _normalize_rating(raw.get("rating")),
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
