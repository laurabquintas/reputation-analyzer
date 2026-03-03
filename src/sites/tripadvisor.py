#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tripadvisor_scraper.py

Fetch TripAdvisor review scores for a fixed list of hotels via the
TripAdvisor Content API and update a CSV with the following layout:
- Index column: "Hotel"
- One column per run date (YYYY-MM-DD) with the score (float, 0-5)
- An "Average Score" column computed across all date columns

The scraper uses the official TripAdvisor Content API (location details
endpoint) to retrieve the aggregate rating for each hotel.

CSV details:
- Default file: tripadvisor_scores.csv
- Separator: ";" (semicolon)
- If the CSV does not exist, it is created with the hotel list as index.

USAGE
-----
Basic (API key via env var):
    export TRIPADVISOR_API_KEY="YOUR_KEY"
    python src/sites/tripadvisor.py

Pin a specific date column (otherwise "today"):
    python src/sites/tripadvisor.py --date 2025-09-20

REQUIREMENTS
------------
pandas
requests
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

logger = logging.getLogger(__name__)


# ---------------------- Default configuration ---------------------- #

DEFAULT_CSV = "tripadvisor_scores.csv"
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # .../src/sites
    "..", "..", "data", DEFAULT_CSV)
DEFAULT_SEP = ";"                # you are using semicolon CSV

DEFAULT_MIN_DELAY = 2.5          # seconds between hotel requests (min)
DEFAULT_MAX_DELAY = 5.0          # seconds between hotel requests (max)

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "hotels.yaml")

def _load_location_ids() -> dict[str, str]:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {h["name"]: h["tripadvisor_location_id"] for h in cfg["hotels"] if h.get("tripadvisor_location_id")}

# Map of hotel display name -> TripAdvisor location ID
LOCATION_IDS = _load_location_ids()

DATE_COL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")  # YYYY-MM-DD
# -------------------------- Scraper logic -------------------------- #

def sanitize_tripadvisor_score(score: float | None) -> float | None:
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if 0.0 <= value <= 5.0:
        return value
    logger.warning("TripAdvisor score out of expected range 0-5: %s. Ignoring value.", value)
    return None


def ta_get_rating(location_id: str, api_key: str):
    url = f"https://api.content.tripadvisor.com/api/v1/location/{location_id}/details"
    params = {
        "key": api_key,
        "language": "en",
    }
    resp = requests.get(url, params=params, timeout=15)
    logger.debug("Details status: %d", resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    logger.debug("Details json: %s", data)

    rating_raw = data.get("rating")
    num_reviews = data.get("num_reviews") or data.get("review_count")
    try:
        rating = float(rating_raw) if rating_raw is not None else None
    except ValueError:
        rating = None
    rating = sanitize_tripadvisor_score(rating)

    logger.info("Parsed rating: %s, num_reviews: %s", rating, num_reviews)
    return rating, num_reviews


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
    p = argparse.ArgumentParser(description="Fetch TripAdvisor scores and update a semicolon CSV.")
    p.add_argument("--csv", default=DEFAULT_CSV_PATH, help=f"Output CSV path (default: {DEFAULT_CSV_PATH})")
    p.add_argument("--sep", default=DEFAULT_SEP, help=f"CSV separator (default: '{DEFAULT_SEP}')")
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                   help="Date column to write (YYYY-MM-DD). Default: today.")
    p.add_argument("--min-delay", type=float, default=2.0, help=f"Min delay (s) between hotels (default: 2.0)")
    p.add_argument("--max-delay", type=float, default=5.0, help=f"Max delay (s) between hotels (default: 5.0)")
    p.add_argument(
        "--api-key",
        default=None,
        help="Tripadvisor API key. If omitted, uses TRIPADVISOR_API_KEY env var.",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    # Validate date format early
    if not DATE_COL_RE.fullmatch(args.date):
        raise ValueError(f"--date must be YYYY-MM-DD, got: {args.date}")
    api_key = args.api_key or os.getenv("TRIPADVISOR_API_KEY")
    if not api_key:
        raise RuntimeError("No API key provided. Use --api-key or set TRIPADVISOR_API_KEY.")

    hotels = list(LOCATION_IDS.keys())
    df = ensure_csv(args.csv, args.sep, hotels)

    today_col = args.date
    new_scores: dict[str, float | None] = {}

    logger.info("Writing scores into column: %s", today_col)

    for i, (hotel, url) in enumerate(LOCATION_IDS.items(), start=1):
        logger.info("%02d/%d → %s", i, len(LOCATION_IDS), hotel)
        score, n = ta_get_rating(url, api_key=api_key)
        score = sanitize_tripadvisor_score(score)
        new_scores[hotel] = score
        if score is not None:
            logger.info("  %s/5", score)
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
