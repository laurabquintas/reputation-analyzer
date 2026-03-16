#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
google_scraper.py

Playwright-based scraper for Google Maps hotel ratings and reviews.

This module provides Playwright functions to fetch scores and reviews
directly from Google Maps, without requiring a Google API key.  It is
used as the primary data source, with the Google Places API as a
fallback.

Google Maps renders content client-side, so a headless browser is
required.  Reviews are loaded via infinite scroll on the Reviews tab.

REQUIREMENTS
------------
playwright, beautifulsoup4, PyYAML
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

# ---------------------- Configuration ---------------------- #

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "hotels.yaml"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _load_google_maps_urls() -> dict[str, str]:
    """Load Google Maps URLs from config."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        h["name"]: h["google_maps_url"]
        for h in cfg["hotels"]
        if h.get("google_maps_url")
    }


def _load_google_queries() -> dict[str, str]:
    """Load Google search queries from config."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        h["name"]: h["google_query"]
        for h in cfg["hotels"]
        if h.get("google_query")
    }


GOOGLE_MAPS_URLS = _load_google_maps_urls()
GOOGLE_QUERIES = _load_google_queries()


# ---------------------- Cookie consent ---------------------- #

_REJECT_LABELS = ["Reject all", "Rejeitar tudo", "Alle ablehnen", "Rechazar todo"]


def _dismiss_google_consent(page) -> None:
    """Reject non-essential cookies on Google consent page (multi-language)."""
    try:
        for label in _REJECT_LABELS:
            reject_btn = page.locator(f'button:has-text("{label}")')
            if reject_btn.count() > 0:
                reject_btn.first.click()
                time.sleep(2)
                return
    except Exception:
        pass


async def _async_dismiss_google_consent(page) -> None:
    """Async: reject non-essential cookies (multi-language)."""
    import asyncio
    try:
        for label in _REJECT_LABELS:
            reject_btn = page.locator(f'button:has-text("{label}")')
            if await reject_btn.count() > 0:
                await reject_btn.first.click()
                await asyncio.sleep(2)
                return
    except Exception:
        pass


# ---------------------- Sort reviews ---------------------- #

_SORT_TIMEOUT = 5000  # ms – max wait for sort button / menu to appear


def _sort_reviews_by_newest(page) -> None:
    """Click the sort dropdown and select 'Newest' to get most recent reviews first.

    Google Maps defaults to "Most relevant" sort order.  This clicks the
    sort button and selects "Newest" so that the infinite-scroll loop
    fetches the most recent reviews first.

    Uses ``wait_for`` instead of fixed sleeps so it adapts to slow pages.
    """
    try:
        # Wait for the sort button to appear (it loads after the Reviews tab)
        sort_btn = page.locator('button[aria-label*="Sort"]')
        try:
            sort_btn.first.wait_for(state="visible", timeout=_SORT_TIMEOUT)
        except Exception:
            # Fallback: button showing current sort label
            sort_btn = page.locator('button:has-text("Most relevant")')
            try:
                sort_btn.first.wait_for(state="visible", timeout=_SORT_TIMEOUT)
            except Exception:
                logger.warning("Sort button not found, reviews will use default order")
                return

        sort_btn.first.click()

        # Wait for the dropdown menu to appear
        newest = page.locator('[role="menuitemradio"]:has-text("Newest")')
        try:
            newest.first.wait_for(state="visible", timeout=_SORT_TIMEOUT)
        except Exception:
            logger.warning("'Newest' option not found in sort menu")
            return

        newest.first.click()
        # Wait for the review list to reload after sorting
        time.sleep(3)
        logger.info("Sorted reviews by 'Newest'")
    except Exception as e:
        logger.warning("Failed to sort reviews by newest: %s", e)


async def _async_sort_reviews_by_newest(page) -> None:
    """Async: click the sort dropdown and select 'Newest'."""
    import asyncio
    try:
        sort_btn = page.locator('button[aria-label*="Sort"]')
        try:
            await sort_btn.first.wait_for(state="visible", timeout=_SORT_TIMEOUT)
        except Exception:
            sort_btn = page.locator('button:has-text("Most relevant")')
            try:
                await sort_btn.first.wait_for(state="visible", timeout=_SORT_TIMEOUT)
            except Exception:
                logger.warning("Sort button not found, reviews will use default order")
                return

        await sort_btn.first.click()

        newest = page.locator('[role="menuitemradio"]:has-text("Newest")')
        try:
            await newest.first.wait_for(state="visible", timeout=_SORT_TIMEOUT)
        except Exception:
            logger.warning("'Newest' option not found in sort menu")
            return

        await newest.first.click()
        await asyncio.sleep(3)
        logger.info("Sorted reviews by 'Newest'")
    except Exception as e:
        logger.warning("Failed to sort reviews by newest: %s", e)


# ---------------------- Score scraping ---------------------- #

def _extract_score_from_page(page) -> tuple[float | None, int | None]:
    """Extract rating and review count from a rendered Google Maps page.

    Parses aria-label attributes like "4.8 stars " and "51 reviews".
    """
    rating = None
    num_reviews = None

    # Rating: aria-label like "4.8 stars" (with possible trailing space)
    stars_loc = page.locator('[aria-label*="stars"]')
    for i in range(stars_loc.count()):
        label = stars_loc.nth(i).get_attribute("aria-label") or ""
        m = re.match(r"(\d+(?:\.\d+)?)\s+stars?\s*$", label.strip())
        if m:
            try:
                rating = float(m.group(1))
                if 0.0 <= rating <= 5.0:
                    break
                rating = None
            except ValueError:
                pass

    # Review count: aria-label like "51 reviews"
    reviews_loc = page.locator('[aria-label*="reviews"]')
    for i in range(reviews_loc.count()):
        label = reviews_loc.nth(i).get_attribute("aria-label") or ""
        m = re.match(r"(\d+)\s+reviews?", label.strip())
        if m:
            num_reviews = int(m.group(1))
            break

    return rating, num_reviews


# --------- Google Search score cards (Tripadvisor + Google) --------- #

_SCORE_RE = re.compile(r"^(\d+[,\.]\d+)/5$")
_COUNT_RE = re.compile(
    r"^(\d+)\s+(?:críticas|reviews?|Rezensionen|avis|reseñas|"
    r"recensioni|beoordelingen|opiniões)$",
    re.I,
)
_KNOWN_PLATFORMS = {"tripadvisor", "google", "booking.com", "expedia", "holidaycheck"}


def _extract_search_score_cards(page) -> dict[str, dict]:
    """Extract platform score cards from a rendered Google Search page.

    Google Search shows score cards (e.g. Tripadvisor 4.7/5 18 reviews,
    Google 4.8/5 51 reviews) when you search for "<hotel> reviews".

    The cards are parsed from the page's inner text by looking for lines
    matching ``X,Y/5`` (or ``X.Y/5``) and checking the surrounding lines
    for a known platform name and a review count.

    Returns:
        Dict mapping platform name (lowercase) to
        ``{"score": float, "num_reviews": int | None}``.
    """
    body_text = page.inner_text("body")
    lines = [ln.strip() for ln in body_text.split("\n") if ln.strip()]

    result: dict[str, dict] = {}

    for i, line in enumerate(lines):
        m = _SCORE_RE.match(line)
        if not m:
            continue

        score_str = m.group(1).replace(",", ".")
        score = float(score_str)
        if not 0.0 <= score <= 5.0:
            continue

        # Platform name appears ABOVE the score line
        platform = None
        for j in range(i - 1, max(0, i - 4) - 1, -1):
            if lines[j].lower() in _KNOWN_PLATFORMS:
                platform = lines[j].lower()
                break

        # Review count appears BELOW the score line
        num_reviews = None
        for j in range(i + 1, min(len(lines), i + 3)):
            cm = _COUNT_RE.match(lines[j])
            if cm:
                num_reviews = int(cm.group(1))
                break

        if platform and platform not in result:
            result[platform] = {"score": score, "num_reviews": num_reviews}

    return result


def fetch_scores_from_google_search(
    query: str,
    timeout: int = 15,
) -> dict[str, dict]:
    """Scrape platform score cards from a Google Search results page.

    Navigates to ``google.com/search?q=<query>+reviews`` and extracts
    the score cards that Google shows for platforms like Tripadvisor and
    Google itself.

    Args:
        query: Hotel search query (e.g. "Ananea Castelo Suites Algarve").
        timeout: Page-load timeout in seconds.

    Returns:
        Dict mapping platform (e.g. ``"tripadvisor"``, ``"google"``) to
        ``{"score": float, "num_reviews": int | None}``.
    """
    import urllib.parse

    search_url = (
        "https://www.google.com/search?q="
        + urllib.parse.quote(query + " reviews")
    )
    timeout_ms = timeout * 1000

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
            )
            page = context.new_page()
            try:
                page.goto(search_url, timeout=timeout_ms, wait_until="domcontentloaded")
                time.sleep(2)
                _dismiss_google_consent(page)
                time.sleep(2)

                cards = _extract_search_score_cards(page)
                logger.info(
                    "Google Search score cards for '%s': %s",
                    query,
                    {k: v["score"] for k, v in cards.items()},
                )
                return cards
            finally:
                browser.close()
    except Exception as e:
        logger.warning("Google Search score fetch failed for '%s': %s", query, e)
        return {}


def fetch_google_rating_playwright(
    url: str,
    retries: int = 2,
    timeout: int = 30000,
) -> tuple[float | None, int | None]:
    """Fetch Google Maps rating via Playwright.

    Returns (rating, num_reviews) or (None, None) on failure.
    """
    for attempt in range(retries + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    locale="en-GB",
                )
                page = context.new_page()
                try:
                    page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                    time.sleep(2)
                    _dismiss_google_consent(page)
                    time.sleep(3)

                    rating, num_reviews = _extract_score_from_page(page)
                    if rating is not None:
                        return rating, num_reviews
                finally:
                    browser.close()
        except Exception as e:
            logger.warning("Playwright fetch failed for %s (attempt %d): %s", url, attempt + 1, e)

        time.sleep(random.uniform(2.0, 4.0) * (attempt + 1))

    return None, None


# ---------------------- Review scraping ---------------------- #

_RELATIVE_DATE_RE = re.compile(
    r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago",
    re.I,
)


def _parse_relative_date(text: str) -> str:
    """Convert relative date like '4 months ago' to approximate YYYY-MM-DD."""
    m = _RELATIVE_DATE_RE.search(text)
    if not m:
        return ""
    amount = int(m.group(1))
    unit = m.group(2).lower()

    now = datetime.now()
    if unit == "second":
        dt = now - timedelta(seconds=amount)
    elif unit == "minute":
        dt = now - timedelta(minutes=amount)
    elif unit == "hour":
        dt = now - timedelta(hours=amount)
    elif unit == "day":
        dt = now - timedelta(days=amount)
    elif unit == "week":
        dt = now - timedelta(weeks=amount)
    elif unit == "month":
        dt = now - timedelta(days=amount * 30)
    elif unit == "year":
        dt = now - timedelta(days=amount * 365)
    else:
        return ""

    return dt.strftime("%Y-%m-%d")


# Map full (lowercase) language names from Google Translate footer to a
# likely country of origin.  English is intentionally omitted — the caller
# treats "no original_language" (native English) as "England".
_LANG_NAME_TO_COUNTRY: dict[str, str] = {
    "french": "France",
    "german": "Germany",
    "portuguese": "Portugal",
    "spanish": "Spain",
    "italian": "Italy",
    "dutch": "Netherlands",
    "russian": "Russia",
    "polish": "Poland",
    "czech": "Czech Republic",
    "swedish": "Sweden",
    "danish": "Denmark",
    "norwegian": "Norway",
    "finnish": "Finland",
    "greek": "Greece",
    "turkish": "Turkey",
    "romanian": "Romania",
    "hungarian": "Hungary",
}


def _parse_review_element(text: str, review_id: str = "") -> dict | None:
    """Parse a single review element's inner text into a review dict.

    Google Maps review text structure (lines):
      - Author name
      - (optional) "N reviews" or "Local Guide · N reviews"
      - (optional) empty line
      - "X/5"  (rating)
      - "N [units] ago on"
      - "Google" or "Tripadvisor"
      - Review text body
      - (optional) "Like", "Share", "Response from the owner..."

    Args:
        text: The inner text of the review element.
        review_id: The ``data-review-id`` attribute value (already resolved
            by the caller so this works in both sync and async contexts).
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return None

    author_name = lines[0]

    # Find the rating line (e.g., "5/5" or "4/5")
    rating = None
    rating_idx = -1
    for i, line in enumerate(lines):
        m = re.match(r"^(\d)/5$", line)
        if m:
            rating = float(m.group(1))
            rating_idx = i
            break

    if rating is None:
        return None

    # Date is the line after rating (e.g., "4 months ago on")
    date_text = lines[rating_idx + 1] if rating_idx + 1 < len(lines) else ""
    published_date = _parse_relative_date(date_text)

    # Source is the line after date (e.g., "Google" or "Tripadvisor")
    source_line = lines[rating_idx + 2] if rating_idx + 2 < len(lines) else ""

    # Review text starts after the source line
    text_start = rating_idx + 3
    # Filter out UI elements like "Like", "Share", "NEW", "Response from the owner"
    _UI_LABELS = {"Like", "Share", "NEW", "Translated by Google", "Original"}
    review_lines = []
    for line in lines[text_start:]:
        if line in _UI_LABELS or line.startswith("Response from the owner"):
            break
        if line.startswith("Read more on"):
            break
        review_lines.append(line)

    review_text = " ".join(review_lines).strip()

    # Extract original language from Google Translate footer before stripping.
    # e.g. "...great hotel. Translated by Google ・ See original (French)"
    original_language = ""
    lang_match = re.search(r"See original\s*\((\w+)\)", review_text)
    if lang_match:
        original_language = lang_match.group(1).lower()  # e.g. "french", "german"

    # Strip trailing Google Translate footer.
    # The separator dot varies across Unicode variants, so match anything
    # after "Translated by Google" to end of string.
    review_text = re.sub(
        r"\s*Translated by Google\b.*$",
        "",
        review_text,
    ).strip()

    # Extract structured metadata that Google appends after the review body:
    #   "Trip type Vacation Travel group Couple Rooms: 5 Service: 5
    #    Location: 5 Hotel highlights Great value"
    trip_type = ""
    travel_group = ""
    meta_match = re.search(r"\bTrip type\s+(\w+)", review_text)
    if meta_match:
        trip_type = meta_match.group(1).lower()  # e.g. "vacation", "business"
    group_match = re.search(r"\bTravel group\s+(\w+)", review_text)
    if group_match:
        travel_group = group_match.group(1).lower()  # e.g. "couple", "family", "solo"

    # Strip the metadata block from the review text
    review_text = re.sub(
        r"\s*Trip type\b.*$",
        "",
        review_text,
    ).strip()
    review_text = re.sub(
        r"\s*(?:Rooms|Service|Location|Hotel highlights)\s*:?\s*\d.*$",
        "",
        review_text,
    ).strip()

    # Derive country from original language.
    # No original_language means the review is natively English → "England"
    if original_language:
        country = _LANG_NAME_TO_COUNTRY.get(original_language, "Unknown")
    else:
        country = "England/USA"

    # Use provided review_id or generate a deterministic one
    if not review_id:
        id_seed = f"{author_name}_{published_date}_{review_text[:50]}"
        review_id = hashlib.sha256(id_seed.encode()).hexdigest()[:16]

    return {
        "id": review_id,
        "rating": rating,
        "title": "",
        "text": review_text,
        "published_date": published_date,
        "author_name": author_name,
        "source_platform": source_line.lower() if source_line else "google",
        "trip_type": trip_type or "Unknown",
        "travel_group": travel_group,
        "original_language": original_language,
        "country": country,
    }


_EXPAND_AND_EXTRACT_JS = """
() => {
    // Click "More" buttons to expand truncated Google-native review text.
    // Only targets the known Google Maps "More" button class (w8nwRe).
    // NOTE: TripAdvisor reviews on Google Maps are always snippets —
    // full text is only available on tripadvisor.com or via their API.
    document.querySelectorAll('[data-review-id] button.w8nwRe')
        .forEach(btn => { try { btn.click(); } catch(e) {} });

    // Extract review data from all visible review elements.
    // Using JS avoids DOM-virtualisation timeouts that occur when
    // Playwright tries to access .nth(N) on elements that have
    // already been recycled by Google Maps' virtual scroller.
    const results = [];
    document.querySelectorAll('[data-review-id]').forEach(el => {
        results.push({
            rid: el.getAttribute('data-review-id') || '',
            text: el.innerText || '',
        });
    });
    return results;
}
"""


def _sync_fetch_reviews(
    url: str,
    max_reviews: int = 50,
    timeout: int = 30000,
    scroll_pause: float = 2.0,
) -> list[dict]:
    """Sync: fetch reviews from Google Maps by scrolling the reviews panel."""
    reviews: list[dict] = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="en-GB",
            )
            page = context.new_page()
            try:
                page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                time.sleep(2)
                _dismiss_google_consent(page)

                # Wait for the Reviews tab to appear (Google Maps SPA loads slowly)
                reviews_tab = page.locator('button[aria-label*="Reviews"]')
                try:
                    reviews_tab.first.wait_for(state="visible", timeout=15000)
                except Exception:
                    logger.warning("No Reviews tab found on %s", url)
                    return reviews

                reviews_tab.first.click()
                time.sleep(2)

                # Sort by newest first
                _sort_reviews_by_newest(page)

                # Scroll to load reviews
                scrollable = page.locator("div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
                try:
                    scrollable.first.wait_for(state="visible", timeout=10000)
                except Exception:
                    logger.warning("No scrollable reviews panel found")
                    return reviews

                seen_ids: set[str] = set()
                max_scrolls = max_reviews // 10 + 5  # ~10 reviews per scroll
                no_new_count = 0

                for scroll_num in range(max_scrolls):
                    scrollable.first.evaluate("el => el.scrollTop = el.scrollHeight")
                    time.sleep(scroll_pause)

                    # Expand truncated reviews and extract data via JS
                    # (avoids stale-element timeouts from DOM virtualisation)
                    raw_items = page.evaluate(_EXPAND_AND_EXTRACT_JS)
                    # Allow expanded text to settle, then re-extract
                    time.sleep(0.5)
                    raw_items = page.evaluate(_EXPAND_AND_EXTRACT_JS)

                    new_found = 0
                    for item in raw_items:
                        rid = item.get("rid", "")
                        if not rid or rid in seen_ids:
                            continue
                        seen_ids.add(rid)

                        review = _parse_review_element(
                            item.get("text", ""), review_id=rid,
                        )
                        if review:
                            reviews.append(review)
                            new_found += 1

                    if new_found == 0:
                        no_new_count += 1
                        if no_new_count >= 2:
                            break
                    else:
                        no_new_count = 0

                    if len(reviews) >= max_reviews:
                        break

                    logger.info(
                        "  Scroll %d: %d total reviews collected",
                        scroll_num + 1, len(reviews),
                    )

            finally:
                browser.close()
    except Exception as e:
        logger.warning("Playwright review fetch failed for %s: %s", url, e)

    return reviews[:max_reviews]


async def _async_fetch_reviews(
    url: str,
    max_reviews: int = 50,
    timeout: int = 30000,
    scroll_pause: float = 2.0,
) -> list[dict]:
    """Async: fetch reviews from Google Maps (for Jupyter compatibility)."""
    import asyncio
    reviews: list[dict] = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="en-GB",
            )
            page = await context.new_page()
            try:
                await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                await asyncio.sleep(2)
                await _async_dismiss_google_consent(page)

                # Wait for the Reviews tab to appear (Google Maps SPA loads slowly)
                reviews_tab = page.locator('button[aria-label*="Reviews"]')
                try:
                    await reviews_tab.first.wait_for(state="visible", timeout=15000)
                except Exception:
                    logger.warning("No Reviews tab found on %s", url)
                    return reviews

                await reviews_tab.first.click()
                await asyncio.sleep(2)

                # Sort by newest first
                await _async_sort_reviews_by_newest(page)

                # Scroll to load reviews
                scrollable = page.locator("div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
                try:
                    await scrollable.first.wait_for(state="visible", timeout=10000)
                except Exception:
                    logger.warning("No scrollable reviews panel found")
                    return reviews

                seen_ids: set[str] = set()
                max_scrolls = max_reviews // 10 + 5
                no_new_count = 0

                for scroll_num in range(max_scrolls):
                    await scrollable.first.evaluate(
                        "el => el.scrollTop = el.scrollHeight",
                    )
                    await asyncio.sleep(scroll_pause)

                    # Expand truncated reviews and extract data via JS
                    raw_items = await page.evaluate(_EXPAND_AND_EXTRACT_JS)
                    await asyncio.sleep(0.5)
                    raw_items = await page.evaluate(_EXPAND_AND_EXTRACT_JS)

                    new_found = 0
                    for item in raw_items:
                        rid = item.get("rid", "")
                        if not rid or rid in seen_ids:
                            continue
                        seen_ids.add(rid)

                        review = _parse_review_element(
                            item.get("text", ""), review_id=rid,
                        )
                        if review:
                            reviews.append(review)
                            new_found += 1

                    if new_found == 0:
                        no_new_count += 1
                        if no_new_count >= 2:
                            break
                    else:
                        no_new_count = 0

                    if len(reviews) >= max_reviews:
                        break

            finally:
                await browser.close()
    except Exception as e:
        logger.warning("Playwright review fetch failed for %s: %s", url, e)

    return reviews[:max_reviews]


def fetch_reviews(
    url: str,
    max_reviews: int = 50,
    timeout: int = 30000,
    scroll_pause: float = 2.0,
) -> list[dict]:
    """Fetch reviews from Google Maps.

    Automatically uses the async Playwright API when called inside an
    existing event loop (e.g. Jupyter notebooks), and the sync API otherwise.
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
            _async_fetch_reviews(url, max_reviews, timeout, scroll_pause)
        )
    else:
        return _sync_fetch_reviews(url, max_reviews, timeout, scroll_pause)


def google_get_reviews_playwright(
    url: str,
    max_reviews: int = 50,
    timeout: int = 30,
    scroll_pause: float = 2.0,
) -> list[dict]:
    """Fetch all reviews from a Google Maps place page.

    Args:
        url: Google Maps URL (short or full).
        max_reviews: Maximum reviews to collect.
        timeout: Page load timeout in seconds.
        scroll_pause: Seconds between scroll actions.

    Returns:
        List of review dicts with id, rating, text, published_date, author_name.
    """
    timeout_ms = timeout * 1000

    logger.info("Fetching Google Maps reviews: %s", url)

    reviews = fetch_reviews(
        url,
        max_reviews=max_reviews,
        timeout=timeout_ms,
        scroll_pause=scroll_pause,
    )

    # Deduplicate by ID
    seen: set[str] = set()
    unique: list[dict] = []
    for r in reviews:
        rid = r.get("id", "")
        if rid and rid not in seen:
            unique.append(r)
            seen.add(rid)

    logger.info("Fetched %d unique reviews", len(unique))
    return unique


# ---------------------- Platform-filtered helpers ---------------------- #

def get_reviews_by_platform(
    url: str,
    platform: str,
    max_reviews: int = 50,
    timeout: int = 30,
    scroll_pause: float = 2.0,
) -> list[dict]:
    """Fetch reviews from Google Maps filtered by source platform.

    Google Maps reviews include both Google-native and cross-posted
    TripAdvisor reviews.  Each review has a ``source_platform`` field
    ("google" or "tripadvisor").

    Args:
        url: Google Maps URL.
        platform: Filter value for ``source_platform`` (e.g. "tripadvisor").
        max_reviews: Maximum reviews to fetch before filtering.
        timeout: Page load timeout in seconds.
        scroll_pause: Seconds between scroll actions.

    Returns:
        Reviews whose ``source_platform`` matches *platform*.
    """
    all_reviews = google_get_reviews_playwright(
        url,
        max_reviews=max_reviews,
        timeout=timeout,
        scroll_pause=scroll_pause,
    )
    platform_lower = platform.lower()
    return [r for r in all_reviews if r.get("source_platform", "").lower() == platform_lower]


def compute_score_from_reviews(reviews: list[dict]) -> tuple[float | None, int]:
    """Compute average rating and count from a list of review dicts.

    Returns:
        (average_rating, num_reviews).  average_rating is None when there
        are no reviews with a valid rating.
    """
    ratings = [r["rating"] for r in reviews if r.get("rating") is not None]
    if not ratings:
        return None, 0
    avg = round(sum(ratings) / len(ratings), 1)
    return avg, len(ratings)
