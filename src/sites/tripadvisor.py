#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
holidaycheck_scraper.py

Fetch Booking.com review scores for a fixed list of hotels and update a CSV
with the following layout:
- Index column: "Hotel"
- One column per run date (YYYY-MM-DD) with the score (float)
- An "Average Score" column computed across all date columns

The scraper is intentionally simple and stable:
- It requests each property page with a desktop User-Agent.
- It parses <script type="application/ld+json"> and reads the JSON-LD's
  "aggregateRating" -> "ratingValue". This is the most reliable source.
- If no rating is found, it records NaN for that hotel on that date.

CSV details:
- Default file: booking_scores.csv
- Separator: ";" (semicolon), matching your current file
- If the CSV does not exist, it is created with the hotel list as index.

USAGE
-----
Basic:
    python src/sites/booking.py

Custom CSV path / retries / delays:
    python src/sites/booking.py --csv data/booking_scores.csv --retries 2 --min-delay 2.5 --max-delay 5.0

Pin a specific date column (otherwise "today"):
    python src/sites/booking.py --date 2025-09-20

REQUIREMENTS
------------
pandas
requests
beautifulsoup4

TIPS
----
- Verify each Booking URL in a normal browser to ensure it points to the exact
  property page you want.
- Be polite with delays; small random sleeps help avoid throttling.
"""

from __future__ import annotations

import os
import re
import json
import argparse
import random
from time import sleep
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ---------------------- Default configuration ---------------------- #

DEFAULT_CSV = "tripadvisor_scores.csv"
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # .../src/sites
    "..", "..", "data", DEFAULT_CSV)
DEFAULT_SEP = ";"                # you are using semicolon CSV

DEFAULT_MIN_DELAY = 2.5          # seconds between hotel requests (min)
DEFAULT_MAX_DELAY = 5.0          # seconds between hotel requests (max)
# Map of hotel display name -> Booking URL
LOCATION_IDS ={
                'Ananea Castelo Suites Hotel': '33299137',
                'PortoBay Falésia': '625806',
                'Regency Salgados Hotel & Spa': '23418643',
                'NAU São Rafael Atlântico': '289104',
                'NAU Salgados Dunas Suites': '1772673',
                'Vidamar Resort Hotel Algarve': '3927147'
                }

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
    print(f"[warn] tripadvisor score out of expected range 0-5: {value}. Ignoring value.")
    return None


def ta_get_rating(location_id: str, api_key: str):
    url = f"https://api.content.tripadvisor.com/api/v1/location/{location_id}/details"
    params = {
        "key": api_key,
        "language": "en",
    }
    resp = requests.get(url, params=params, timeout=15)
    print("details status:", resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    print("details json:", data)

    rating_raw = data.get("rating")
    num_reviews = data.get("num_reviews") or data.get("review_count")
    try:
        rating = float(rating_raw) if rating_raw is not None else None
    except ValueError:
        rating = None
    rating = sanitize_tripadvisor_score(rating)

    print("Parsed rating:", rating, "num_reviews:", num_reviews)
    return rating, num_reviews


# ---------------------------- CSV logic ---------------------------- #

def ensure_csv(csv_path: str, sep: str, hotels: list[str]) -> pd.DataFrame:
    """
    Create or load the CSV. Ensure the index includes all hotels and
    that an 'Average Score' column exists.
    """
    if not os.path.exists(csv_path):
        print(f"Creating {csv_path} …")
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
    p.add_argument("--min-delay", type=float, default=2.0, help=f"Min delay (s) between hotels (default: 2.0)")
    p.add_argument("--max-delay", type=float, default=5.0, help=f"Max delay (s) between hotels (default: 5.0)")
    p.add_argument(
        "--api-key",
        default=None,
        help="Tripadvisor API key. If omitted, uses TRIPADVISOR_API_KEY env var.",
    )
    return p.parse_args()


def main():
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

    print(f"Writing scores into column: {today_col}\n")

    for i, (hotel, url) in enumerate(LOCATION_IDS.items(), start=1):
        print(f"{i:02d}/{len(LOCATION_IDS)} → {hotel}")
        score, n = ta_get_rating(url, api_key=api_key)
        score = sanitize_tripadvisor_score(score)
        new_scores[hotel] = score
        if score is not None:
            print(f"   {score}/5")
        else:
            print("   (no score)")

        # be polite; jitter within [min-delay, max-delay]
        delay = random.uniform(args.min_delay, args.max_delay)
        sleep(delay)

    # Write column & update average
    df[today_col] = pd.Series(new_scores)
    update_average(df)

    # Save
    df.to_csv(args.csv, sep=args.sep, index_label="Hotel")
    print(f"\nSaved {args.csv}. Added/updated column: {today_col}")
    # Show non-null for this run
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(df[[today_col]].dropna())

if __name__ == "__main__":
    main()
