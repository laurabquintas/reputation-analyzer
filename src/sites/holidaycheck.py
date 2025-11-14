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

DEFAULT_CSV = "holidaycheck_scores.csv"
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # .../src/sites
    "..", "..", "data", DEFAULT_CSV)
DEFAULT_SEP = ";"                # you are using semicolon CSV
DEFAULT_TIMEOUT = 15
DEFAULT_MIN_DELAY = 2.5          # seconds between hotel requests (min)
DEFAULT_MAX_DELAY = 5.0          # seconds between hotel requests (max)

# Map of hotel display name -> Booking URL
URLS = {
    "Ananea Castelo Suites Hotel": "https://www.holidaycheck.de/hi/ananea-castelo-suites-algarve/069563af-47db-44a3-bdb1-3441ae3a2ac4",
    "PortoBay Falésia": "https://www.holidaycheck.de/hi/portobay-falesia/44a47534-85c4-3114-a6da-472d82e16e29",
    "Regency Salgados Hotel & Spa": "https://www.holidaycheck.de/hi/regency-salgados-hotel-spa/b0478236-7644-46b4-8fde-bd6cb1832cf8",
    "NAU São Rafael Atlântico": "https://www.holidaycheck.de/hi/nau-sao-rafael-suites-all-inclusive/739da55a-710e-3514-83f6-8e01149442a5",
    "NAU Salgados Dunas Suites": "https://www.holidaycheck.de/hi/nau-salgados-vila-das-lagoas-apartment/602ac74a-9c28-3d74-8dd9-37c47c53cd4a",
    "Vidamar Resort Hotel Algarve": "https://www.holidaycheck.de/hi/vidamar-hotel-resort-algarve/e641bc1e-59d5-37a0-832e-90e6bbb51977",
}

DATE_COL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")  # YYYY-MM-DD
# -------------------------- Scraper logic -------------------------- #

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

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # You need to inspect the page once and adjust this selector.
    # Typical patterns include something like "4,5 / 6".
    text = soup.get_text(" ", strip=True)

    # Find patterns like "4,5 / 6" or "4.5 / 6"
    m = re.search(r"(\d+[.,]\d)\s*/\s*6", text)
    if not m:
        return None

    raw = m.group(1).replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


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
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"HTTP timeout per hotel (default: {DEFAULT_TIMEOUT})")
    p.add_argument("--min-delay", type=float, default=2.0, help=f"Min delay (s) between hotels (default: 2.0)")
    p.add_argument("--max-delay", type=float, default=5.0, help=f"Max delay (s) between hotels (default: 5.0)")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate date format early
    if not DATE_COL_RE.fullmatch(args.date):
        raise ValueError(f"--date must be YYYY-MM-DD, got: {args.date}")

    hotels = list(URLS.keys())
    df = ensure_csv(args.csv, args.sep, hotels)

    today_col = args.date
    new_scores: dict[str, float | None] = {}

    print(f"Writing scores into column: {today_col}\n")

    for i, (hotel, url) in enumerate(URLS.items(), start=1):
        print(f"{i:02d}/{len(URLS)} → {hotel}")
        score = get_holidaycheck_score(url, timeout=args.timeout)
        new_scores[hotel] = score
        if score is not None:
            print(f"   {score}/6")
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
