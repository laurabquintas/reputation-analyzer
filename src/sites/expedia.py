#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expedia_scraper.py

Fetch Expedia review scores for a fixed list of hotels and update a CSV
with the following layout:
- Index column: "Hotel"
- One column per run date (YYYY-MM-DD) with the score (float, 0–5)
- An "Average Score" column computed across all date columns

This is HTML scraping (no official API); it is inherently brittle and may
break if Expedia changes their layout. Use sparingly and with politeness.

CSV details:
- Default file: expedia_scores.csv
- Separator: ";" (semicolon)
- If the CSV does not exist, it is created with the hotel list as index.

USAGE
-----
Basic:
    python src/sites/expedia.py

Custom CSV path / retries / delays:
    python src/sites/expedia.py --csv data/expedia_scores.csv --retries 2 --min-delay 2.5 --max-delay 5.0

Pin a specific date column (otherwise "today"):
    python src/sites/expedia.py --date 2025-09-20

REQUIREMENTS
------------
pandas
requests
beautifulsoup4
"""

from __future__ import annotations

import logging
import os
import re
import argparse
import random
from time import sleep
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import yaml
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------- Default configuration ---------------------- #

DEFAULT_CSV = "expedia_scores.csv"
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # .../src/sites
    "..", "..", "data", DEFAULT_CSV
)
DEFAULT_SEP = ";"                # semicolon CSV
DEFAULT_TIMEOUT = 20
DEFAULT_MIN_DELAY = 2.0
DEFAULT_MAX_DELAY = 5.0
DEFAULT_RETRIES = 2

DATE_COL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")  # YYYY-MM-DD

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "hotels.yaml")

def _load_expedia_urls() -> Dict[str, str]:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {h["name"]: h["expedia_url"] for h in cfg["hotels"] if h.get("expedia_url")}

# Map of hotel display name -> Expedia URL
EXPEDIA_URLS: Dict[str, str] = _load_expedia_urls()

# ------------------------ Scraper logic ---------------------------- #

HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]


def _expedia_url_candidates(url: str) -> list[str]:
    """
    Generate fallback Expedia URL variants for a single property URL.

    Rationale:
    Expedia may block or rate-limit one host/query combination while another
    equivalent variant still works, especially on automated runners.

    Variant strategy:
    1. Keep the original URL first.
    2. Try all known Expedia hosts for the same path:
       - euro.expedia.net
       - www.expedia.com
       - www.expedia.co.uk
    3. For each host, include:
       - original query params
       - a version with pwaDialog* params removed

    This improves fetch resilience but does not bypass hard anti-bot blocks.
    """
    if not url:
        return []

    out: list[str] = [url]
    parsed = urlsplit(url)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query_no_dialog = [(k, v) for (k, v) in query_pairs if not k.lower().startswith("pwadialog")]

    # Try all known Expedia hostnames, regardless of which one was given
    all_hosts = ["euro.expedia.net", "www.expedia.com", "www.expedia.co.uk"]
    host_variants = [parsed.netloc] + [h for h in all_hosts if h != parsed.netloc]

    for host in host_variants:
        with_query = urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
        if with_query not in out:
            out.append(with_query)
        if query_no_dialog != query_pairs:
            no_dialog = urlunsplit(
                (parsed.scheme, host, parsed.path, urlencode(query_no_dialog), parsed.fragment)
            )
            if no_dialog not in out:
                out.append(no_dialog)
    return out

def fetch_page(url: str, timeout: int, retries: int) -> Optional[str]:
    """
    Fetch the HTML for a given Expedia hotel URL with simple retry logic.

    Returns the response text on success, or None on failure.
    """
    if not url:
        return None

    last_exc: Exception | None = None
    blocked_markers = [
        "captcha",
        "access denied",
        "bot detection",
        "verify you are human",
        "unusual traffic",
    ]
    candidates = _expedia_url_candidates(url)
    proxies = None
    proxy_url = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if proxy_url:
        proxies = {"http": proxy_url, "https": proxy_url}

    for candidate_url in candidates:
        for attempt in range(retries + 1):
            try:
                request_headers = {
                    **HEADERS,
                    "User-Agent": random.choice(USER_AGENTS),
                }
                resp = requests.get(
                    candidate_url,
                    headers=request_headers,
                    timeout=timeout,
                    proxies=proxies,
                    allow_redirects=True,
                )
                if resp.status_code == 403:
                    raise requests.HTTPError(f"403 Client Error: Forbidden for url: {candidate_url}")
                resp.raise_for_status()
                lowered = resp.text.lower()
                if any(marker in lowered for marker in blocked_markers):
                    logger.warning("Possible anti-bot/challenge page for %s", candidate_url)
                if candidate_url != url:
                    logger.info("Fetched via fallback URL: %s", candidate_url)
                return resp.text
            except Exception as e:
                last_exc = e
                if attempt < retries:
                    # Expedia often rate-limits with 429; back off more aggressively.
                    err_text = str(e)
                    if "429" in err_text:
                        wait_s = 15.0 * (attempt + 1)
                        logger.warning("429 received for %s; backing off %.1fs before retry.", candidate_url, wait_s)
                    else:
                        wait_s = 1.5
                    sleep(wait_s)
                else:
                    logger.warning("Fetch failed for %s after %d attempts: %s", candidate_url, retries + 1, e)
                    # try next candidate URL

    if last_exc is not None:
        logger.error("Failed to fetch page on all Expedia URL variants: %s", last_exc)
    return None


def _safe_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score


def validate_expedia_score(score: float | None) -> float | None:
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if 0.0 <= value <= 10.0:
        return value
    logger.warning("Expedia score out of expected range 0-10: %s. Ignoring value.", value)
    return None


def _extract_jsonld_score(soup: BeautifulSoup) -> Optional[float]:
    """
    Parse rating from JSON-LD blocks when available.
    """
    candidates: list[tuple[int, float]] = []

    for tag in soup.find_all("script", type="application/ld+json"):
        raw = (tag.string or "").strip()
        if not raw:
            continue

        for m in re.finditer(
            r'"aggregateRating"\s*:\s*\{(?P<body>[^{}]*?"ratingValue"\s*:\s*"?(?P<score>\d+(?:\.\d+)?)"?[^{}]*?)\}',
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            score = _safe_float(m.group("score"))
            if score is None:
                continue

            body = m.group("body")
            rank = 0
            if re.search(r'"bestRating"\s*:\s*"?(10|10\.0)"?', body, flags=re.IGNORECASE):
                rank += 6
            if re.search(r'"bestRating"\s*:\s*"?(5|5\.0)"?', body, flags=re.IGNORECASE):
                rank -= 6
            if re.search(r"review|ratingCount|reviewCount", body, flags=re.IGNORECASE):
                rank += 3
            if re.search(r"star|class|classification", body, flags=re.IGNORECASE):
                rank -= 4

            candidates.append((rank, score))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][1]


def _extract_semantic_div_score(soup: BeautifulSoup) -> Optional[float]:
    """
    Parse score from known Expedia score div classes.
    """
    for div in soup.find_all("div"):
        classes = div.get("class", [])
        if (
            "uitk-text" in classes
            and "uitk-type-900" in classes
            and "uitk-text-default-theme" in classes
        ):
            score = _safe_float(div.get_text(strip=True))
            if score is not None:
                return score
    return None


def _extract_textual_score(page_text: str) -> Optional[float]:
    """
    Parse common text renderings such as:
    - 8.6 out of 10
    - 8.6/10
    """
    patterns = [
        r"(\d+(?:\.\d+)?)\s+out of\s+10",
        r"(\d+(?:\.\d+)?)/10",
        r"guest rating[^0-9]{0,30}(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, page_text, flags=re.IGNORECASE)
        if not m:
            continue
        score = _safe_float(m.group(1))
        if score is not None:
            return score
    return None


def _extract_embedded_json_score(html: str) -> Optional[float]:
    """
    Parse score from embedded JS/JSON when visible text selectors fail.
    """
    patterns = [
        r'"reviewScore(?:WithDescription)?"\s*:\s*"?(?P<score>\d+(?:\.\d+)?)"?',
        r'"overall(?:Guest)?Rating"\s*:\s*"?(?P<score>\d+(?:\.\d+)?)"?',
        r'"ratingValue"\s*:\s*"?(?P<score>\d+(?:\.\d+)?)"?',
        r'\\"ratingValue\\"\s*:\s*\\"(?P<score>\d+(?:\.\d+)?)\\"',
        r'\\\\"ratingValue\\\\"\s*:\s*\\\\"(?P<score>\d+(?:\.\d+)?)\\\\"',
    ]

    candidates = [
        html,
        html.replace('\\"', '"'),
        html.replace("\\\\", "\\"),
    ]
    scored_matches: list[tuple[int, float]] = []
    for candidate in candidates:
        for pattern in patterns:
            for m in re.finditer(pattern, candidate, flags=re.IGNORECASE):
                score = _safe_float(m.group("score"))
                if score is None:
                    continue

                start = max(0, m.start() - 180)
                end = min(len(candidate), m.end() + 180)
                context = candidate[start:end]

                rank = 0
                if re.search(r'bestRating"\s*:\s*"?(10|10\.0)"?', context, flags=re.IGNORECASE):
                    rank += 6
                if re.search(r'bestRating"\s*:\s*"?(5|5\.0)"?', context, flags=re.IGNORECASE):
                    rank -= 6
                if re.search(r"reviewScore|guestRating|review|out of 10|/10", context, flags=re.IGNORECASE):
                    rank += 4
                if re.search(r"star|property class|classification|hotel class", context, flags=re.IGNORECASE):
                    rank -= 6

                scored_matches.append((rank, score))

    if not scored_matches:
        return None

    scored_matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored_matches[0][1]


def debug_expedia_score_candidates(url: str, timeout: int = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES) -> dict:
    """
    Return candidate extraction results to help diagnose parser misses.
    """
    html = fetch_page(url, timeout=timeout, retries=retries)
    if html is None:
        return {"fetch_ok": False, "error": "Could not fetch page"}

    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    return {
        "fetch_ok": True,
        "jsonld_score": _extract_jsonld_score(soup),
        "semantic_div_score": _extract_semantic_div_score(soup),
        "textual_score": _extract_textual_score(page_text),
        "embedded_json_score": _extract_embedded_json_score(html),
        "contains_8_6": "8.6" in html or "8,6" in html,
    }


def get_expedia_score(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    debug: bool = False,
) -> Optional[float]:
    """
    Extract Expedia guest rating (0–10) from the <div> containing the score.
    Looks for:
        <div class="uitk-text uitk-type-900 uitk-text-default-theme">8.4</div>
    """
    html = fetch_page(url, timeout=timeout, retries=retries)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)

    # 1) Prefer JSON-LD when present
    score = _extract_jsonld_score(soup)
    if score is not None:
        return validate_expedia_score(score)

    # 2) Limit search to the Reviews section
    reviews_section = soup.find("section", id="Reviews")
    if not reviews_section:
        # fallback: whole page (in case markup changes)
        reviews_section = soup

    # 3) Semantic class-based score
    score = _extract_semantic_div_score(reviews_section)
    if score is not None:
        return validate_expedia_score(score)

    # 4) Textual fallback from page text
    score = _extract_textual_score(page_text)
    if score is not None:
        return validate_expedia_score(score)

    # 5) Embedded JSON fallback
    score = _extract_embedded_json_score(html)
    if score is not None:
        return validate_expedia_score(score)

    if debug:
        details = debug_expedia_score_candidates(url, timeout=timeout, retries=retries)
        logger.debug("Extraction candidates: %s", details)
    else:
        logger.warning("No Expedia score pattern matched for this page.")

    return None


# ---------------------------- CSV logic ---------------------------- #

def ensure_csv(csv_path: str, sep: str, hotels: list[str]) -> pd.DataFrame:
    """
    Create or load the CSV. Ensure the index includes all hotels and
    that an 'Average Score' column exists.
    """
    if not os.path.exists(csv_path):
        logger.info("Creating %s", csv_path)
        df = pd.DataFrame(index=hotels)
        df.index.name = "Hotel"
        df["Average Score"] = pd.NA
        df.to_csv(csv_path, sep=sep, index_label="Hotel")
        return df

    df = pd.read_csv(csv_path, sep=sep, index_col="Hotel")

    # Ensure all hotels are present
    for h in hotels:
        if h not in df.index:
            df.loc[h] = pd.Series(dtype="float64")

    if "Average Score" not in df.columns:
        df["Average Score"] = pd.NA

    return df


def update_average(df: pd.DataFrame) -> None:
    """
    Recompute 'Average Score' across all columns that look like YYYY-MM-DD.
    (Non-date columns are ignored.)
    """
    date_cols = [c for c in df.columns if isinstance(c, str) and DATE_COL_RE.fullmatch(c)]
    if date_cols:
        df["Average Score"] = df[date_cols].mean(axis=1, numeric_only=True).round(2)


# ----------------------------- CLI main ---------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Expedia scores and update a semicolon CSV."
    )
    p.add_argument(
        "--csv",
        default=DEFAULT_CSV_PATH,
        help=f"Output CSV path (default: {DEFAULT_CSV_PATH})",
    )
    p.add_argument(
        "--sep",
        default=DEFAULT_SEP,
        help=f"CSV separator (default: '{DEFAULT_SEP}')",
    )
    p.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date column to write (YYYY-MM-DD). Default: today.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout per hotel (default: {DEFAULT_TIMEOUT})",
    )
    p.add_argument(
        "--min-delay",
        type=float,
        default=DEFAULT_MIN_DELAY,
        help=f"Min delay (s) between hotels (default: {DEFAULT_MIN_DELAY})",
    )
    p.add_argument(
        "--max-delay",
        type=float,
        default=DEFAULT_MAX_DELAY,
        help=f"Max delay (s) between hotels (default: {DEFAULT_MAX_DELAY})",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retries per hotel (default: {DEFAULT_RETRIES})",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging for parser candidate details.",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    # Validate date format early
    if not DATE_COL_RE.fullmatch(args.date):
        raise ValueError(f"--date must be YYYY-MM-DD, got: {args.date}")

    hotels = list(EXPEDIA_URLS.keys())
    df = ensure_csv(args.csv, args.sep, hotels)

    today_col = args.date
    new_scores: dict[str, Optional[float]] = {}

    logger.info("Writing Expedia scores into column: %s", today_col)

    for i, (hotel, url) in enumerate(EXPEDIA_URLS.items(), start=1):
        logger.info("%02d/%d → %s", i, len(EXPEDIA_URLS), hotel)
        score = get_expedia_score(
            url,
            timeout=args.timeout,
            retries=args.retries,
            debug=args.debug,
        )
        score = validate_expedia_score(score)
        new_scores[hotel] = score

        if score is not None:
            logger.info("  %s/10", score)
        else:
            logger.warning("  (no score)")

        delay = random.uniform(args.min_delay, args.max_delay)
        sleep(delay)

    # Write column & update average
    df[today_col] = pd.Series(new_scores)
    update_average(df)

    # Save
    df.to_csv(args.csv, sep=args.sep, index_label="Hotel")
    logger.info("Saved %s. Added/updated column: %s", args.csv, today_col)


if __name__ == "__main__":
    main()
