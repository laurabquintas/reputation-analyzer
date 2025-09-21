#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
booking_scraper.py

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
    python booking_scraper.py

Custom CSV path / retries / delays:
    python booking_scraper.py --csv booking_scores.csv --retries 2 --min-delay 2.5 --max-delay 5.0

Pin a specific date column (otherwise "today"):
    python booking_scraper.py --date 2025-09-20

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

DEFAULT_CSV = "booking_scores.csv"
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # .../src/sites
    "..", "..", "data", DEFAULT_CSV)
DEFAULT_SEP = ";"                # you are using semicolon CSV
DEFAULT_RETRIES = 2
DEFAULT_MIN_DELAY = 2.5          # seconds between hotel requests (min)
DEFAULT_MAX_DELAY = 5.0          # seconds between hotel requests (max)

# Map of hotel display name -> Booking URL
URLS = {
    "Ananea Castelo Suites Hotel": "https://www.booking.com/hotel/pt/castelo-suites.en-gb.html",
    "PortoBay Falésia": "https://www.booking.com/hotel/pt/porto-bay-falesia.en-gb.html",
    "Regency Salgados Hotel & Spa": "https://www.booking.com/hotel/pt/regency-salgados-amp-spa.en-gb.html",
    "NAU São Rafael Atlântico": "https://www.booking.com/hotel/pt/sao-rafael-suites-all-inclusive.en-gb.html",
    "NAU Salgados Dunas Suites": "https://www.booking.com/hotel/pt/westin-salgados-beach-resort-algarve.en-gb.html",
    "Vidamar Resort Hotel Algarve": "https://www.booking.com/hotel/pt/vidamar-algarve-hotel.en-gb.html",
}

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.booking.com/",
}

DATE_COL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# -------------------------- Scraper logic -------------------------- #

def fetch_booking_rating(url: str, session: requests.Session, retries: int = DEFAULT_RETRIES) -> float | None:
    """
    Fetch the Booking rating for a single property page.
    Strategy: parse JSON-LD blocks for "aggregateRating"->"ratingValue".
    Returns a float (e.g., 8.7) or None if not found.
    """
    for attempt in range(retries + 1):
        try:
            r = session.get(url, headers=UA_HEADERS, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            for tag in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(tag.string or "")
                except Exception:
                    continue

                # JSON-LD could be a dict or a list
                items = data if isinstance(data, list) else [data]
                for obj in items:
                    if not isinstance(obj, dict):
                        continue
                    agg = obj.get("aggregateRating")
                    if isinstance(agg, dict) and "ratingValue" in agg:
                        val = str(agg.get("ratingValue"))
                        # Normalize decimal separator & cast
                        return float(val.replace(",", "."))

        except Exception as e:
            print(f"[warn] {url} attempt {attempt + 1} failed: {e}")

        # simple backoff/jitter
        sleep(random.uniform(2.0, 4.0) * (attempt + 1))

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
        df["Average Score"] = df[date_cols].mean(axis=1, numeric_only=True)


# ----------------------------- CLI main ---------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Booking scores and update a semicolon CSV.")
    p.add_argument("--csv", default=DEFAULT_CSV_PATH, help=f"Output CSV path (default: {DEFAULT_CSV_PATH})")
    p.add_argument("--sep", default=DEFAULT_SEP, help=f"CSV separator (default: '{DEFAULT_SEP}')")
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                   help="Date column to write (YYYY-MM-DD). Default: today.")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help=f"HTTP retries per hotel (default: {DEFAULT_RETRIES})")
    p.add_argument("--min-delay", type=float, default=DEFAULT_MIN_DELAY, help=f"Min delay (s) between hotels (default: {DEFAULT_MIN_DELAY})")
    p.add_argument("--max-delay", type=float, default=DEFAULT_MAX_DELAY, help=f"Max delay (s) between hotels (default: {DEFAULT_MAX_DELAY})")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate date format early
    if not DATE_COL_RE.fullmatch(args.date):
        raise ValueError(f"--date must be YYYY-MM-DD, got: {args.date}")

    hotels = list(URLS.keys())
    df = ensure_csv(args.csv, args.sep, hotels)

    session = requests.Session()
    today_col = args.date
    new_scores: dict[str, float | None] = {}

    print(f"Writing scores into column: {today_col}\n")

    for i, (hotel, url) in enumerate(URLS.items(), start=1):
        print(f"{i:02d}/{len(URLS)} → {hotel}")
        score = fetch_booking_rating(url, session, retries=args.retries)
        new_scores[hotel] = score
        if score is not None:
            print(f"   {score}/10")
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
