#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
holidaycheck_scraper.py

Fetch HolidayCheck review scores for a fixed list of hotels and update a CSV
with the following layout:
- Index column: "Hotel"
- One column per run date (YYYY-MM-DD) with the score (float, 0-6)
- An "Average Score" column computed across all date columns

The scraper parses each hotel's HolidayCheck page:
1. Preferred: JSON-LD aggregateRating -> ratingValue (normalized to 0-6 scale).
2. Fallback: text pattern like "4,5 / 6" on the page.
If no rating is found, it records NaN for that hotel on that date.

CSV details:
- Default file: holidaycheck_scores.csv
- Separator: ";" (semicolon)
- If the CSV does not exist, it is created with the hotel list as index.

USAGE
-----
Basic:
    python src/sites/holidaycheck.py

Pin a specific date column (otherwise "today"):
    python src/sites/holidaycheck.py --date 2025-09-20

REQUIREMENTS
------------
pandas
requests
beautifulsoup4
PyYAML
"""

from __future__ import annotations

import logging
import os
import re
import argparse
import random
from time import sleep
from datetime import datetime

import yaml
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------- Default configuration ---------------------- #

DEFAULT_CSV = "holidaycheck_scores.csv"
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # .../src/sites
    "..", "..", "data", DEFAULT_CSV)
DEFAULT_SEP = ";"                # you are using semicolon CSV
DEFAULT_TIMEOUT = 15
DEFAULT_MIN_DELAY = 2.5          # seconds between hotel requests (min)
DEFAULT_MAX_DELAY = 5.0          # seconds between hotel requests (max)
UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
}

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "hotels.yaml")

def _load_urls() -> dict[str, str]:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {h["name"]: h["holidaycheck_url"] for h in cfg["hotels"] if h.get("holidaycheck_url")}

# Map of hotel display name -> HolidayCheck URL
URLS = _load_urls()

DATE_COL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")  # YYYY-MM-DD
# -------------------------- Scraper logic -------------------------- #

def _normalize_to_six_scale(score: float, best_rating: float | None) -> float:
    """
    Convert rating to HolidayCheck's 0-6 scale.
    """
    if best_rating is None or best_rating == 6:
        return score
    if best_rating <= 0:
        return score
    return round((score / best_rating) * 6.0, 2)


def sanitize_holidaycheck_score(score: float | None) -> float | None:
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(6.0, value))


def get_holidaycheck_score(url: str, timeout: int = 15) -> float | None:
    """
    Fetch overall HolidayCheck score (0–6 scale) from a hotel page.

    Returns
    -------
    float or None
        Score if found, else None.
    """
    if not url:
        return None

    resp = requests.get(url, headers=UA_HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 1) Preferred: JSON-LD aggregate rating.
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = (tag.string or "").strip()
        if not raw:
            continue
        m = re.search(
            r'"aggregateRating"\s*:\s*\{[^{}]*?"ratingValue"\s*:\s*"?(?P<score>\d+(?:[.,]\d+)?)"?'
            r'(?:[^{}]*?"bestRating"\s*:\s*"?(?P<best>\d+(?:[.,]\d+)?)"?)?[^{}]*?\}',
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            continue
        try:
            score = float(m.group("score").replace(",", "."))
            best = m.group("best")
            best_rating = float(best.replace(",", ".")) if best else None
            normalized = _normalize_to_six_scale(score, best_rating)
            if normalized > 6.0 and score > 6.0:
                # Guard for pages where bestRating is missing but score clearly
                # appears to be on a 10-point scale.
                normalized = round(score * 0.6, 2)
            return sanitize_holidaycheck_score(normalized)
        except ValueError:
            pass

    # 2) Fallback: text pattern like "4,5 / 6"
    text = soup.get_text(" ", strip=True)

    m = re.search(r"(\d+[.,]\d)\s*/\s*6", text)
    if not m:
        return None

    raw = m.group(1).replace(",", ".")
    try:
        return sanitize_holidaycheck_score(float(raw))
    except ValueError:
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
    # Make sure all hotels exist as rows
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
        df["Average Score"] = round(df[date_cols].mean(axis=1, numeric_only=True),2)


# ----------------------------- CLI main ---------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch HolidayCheck scores and update a semicolon CSV.")
    p.add_argument("--csv", default=DEFAULT_CSV_PATH, help=f"Output CSV path (default: {DEFAULT_CSV_PATH})")
    p.add_argument("--sep", default=DEFAULT_SEP, help=f"CSV separator (default: '{DEFAULT_SEP}')")
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                   help="Date column to write (YYYY-MM-DD). Default: today.")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"HTTP timeout per hotel (default: {DEFAULT_TIMEOUT})")
    p.add_argument("--min-delay", type=float, default=2.0, help=f"Min delay (s) between hotels (default: 2.0)")
    p.add_argument("--max-delay", type=float, default=5.0, help=f"Max delay (s) between hotels (default: 5.0)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    # Validate date format early
    if not DATE_COL_RE.fullmatch(args.date):
        raise ValueError(f"--date must be YYYY-MM-DD, got: {args.date}")

    hotels = list(URLS.keys())
    df = ensure_csv(args.csv, args.sep, hotels)

    today_col = args.date
    new_scores: dict[str, float | None] = {}

    logger.info("Writing scores into column: %s", today_col)

    for i, (hotel, url) in enumerate(URLS.items(), start=1):
        logger.info("%02d/%d → %s", i, len(URLS), hotel)
        try:
            score = get_holidaycheck_score(url, timeout=args.timeout)
        except Exception as exc:
            logger.error("  %s", exc)
            score = None
        score = sanitize_holidaycheck_score(score)
        new_scores[hotel] = score
        if score is not None:
            logger.info("  %s/6", score)
        else:
            logger.warning("  (no score)")

        # be polite; jitter within [min-delay, max-delay]
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
