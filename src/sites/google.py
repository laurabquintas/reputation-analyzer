#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
google_places_scores.py

Fetch Google Maps ratings for a fixed list of hotels (Places API, New) and
update a CSV with the following layout:
- Index column: "Hotel"
- One column per run date (YYYY-MM-DD) with the score (float, 0–5)
- An "Average Score" column computed across all date columns

This script mirrors the HolidayCheck scraper logic you provided, but uses the
official Google Places API instead of HTML scraping.

CSV details:
- Default file: google_scores.csv
- Separator: ";" (semicolon)
- If the CSV does not exist, it is created with the hotel list as index.

USAGE
-----
Basic (API key via env var):
    export GOOGLE_MAPS_API_KEY="YOUR_KEY"
    python src/sites/google_places_scores.py

Custom CSV path / date:
    python src/sites/google_places_scores.py --csv data/google_scores.csv --date 2025-09-20

API key via CLI:
    python src/sites/google_places_scores.py --api-key YOUR_KEY_HERE

REQUIREMENTS
------------
pandas
requests

NOTES
-----
- You must enable the Places API in your Google Cloud project.
- Billing must be enabled on the project.
"""

from __future__ import annotations

import os
import re
import json
import argparse
from datetime import datetime
from typing import Dict

import pandas as pd
import requests


# ---------------------- Default configuration ---------------------- #

DEFAULT_CSV = "google_scores.csv"
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # .../src/sites
    "..", "..", "data", DEFAULT_CSV
)
DEFAULT_SEP = ";"                # semicolon CSV
DEFAULT_TIMEOUT = 15

PLACES_SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"

# Map of hotel display name -> text query for Places search
# Adjust queries to what works best for each hotel.
HOTEL_QUERIES: Dict[str, str] = {
    "Ananea Castelo Suites Hotel": "Ananea Castelo Suites Algarve, Portugal",
    "PortoBay Falésia": "PortoBay Falésia, Albufeira, Portugal",
    "Regency Salgados Hotel & Spa": "Regency Salgados Hotel & Spa, Algarve, Portugal",
    "NAU São Rafael Atlântico": "NAU São Rafael Atlântico, Albufeira, Portugal",
    "NAU Salgados Dunas Suites": "NAU Salgados Dunas Suites, Albufeira, Portugal",
    "Vidamar Resort Hotel Algarve": "Vidamar Resort Hotel Algarve, Albufeira, Portugal",
}
# Map of hotel display name 
URLS = {
    "Ananea Castelo Suites Hotel": "https://maps.app.goo.gl/QsTaS8vLupyrC3hQ8",
    "PortoBay Falésia": "https://maps.app.goo.gl/DxodrUv4ub7qp89eA",
    "Regency Salgados Hotel & Spa": "https://maps.app.goo.gl/UZ6dAot3VC4eWV3U7",
    "NAU São Rafael Atlântico": "https://maps.app.goo.gl/G3Nfg49qBYQkR2xr5",
    "NAU Salgados Dunas Suites": "https://maps.app.goo.gl/CxCEgfZkiXnzAEsy9",
    "Vidamar Resort Hotel Algarve": "https://maps.app.goo.gl/etAzqPDxgnjJ2DDu7",
}

DATE_COL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")  # YYYY-MM-DD


# -------------------------- API logic ------------------------------ #

def get_google_rating(query: str, api_key: str, timeout: int = DEFAULT_TIMEOUT) -> float | None:
    """
    Call Google Places Text Search for the given query and return the rating (0–5).

    Returns
    -------
    float or None
        Rating if found, else None.
    """
    if not query:
        return None

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        # Only request what we need
        "X-Goog-FieldMask": "places.id,places.displayName,places.rating,places.userRatingCount",
    }

    payload = {
        "textQuery": query
    }

    resp = requests.post(
        PLACES_SEARCH_TEXT_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    places = data.get("places", [])
    if not places:
        return None

    place = places[0]
    rating = place.get("rating")
    if rating is None:
        return None

    try:
        return float(rating)
    except (TypeError, ValueError):
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
        description="Fetch Google Maps ratings (Places API) and update a semicolon CSV."
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
        "--api-key",
        default=None,
        help="Google Maps API key. If omitted, uses GOOGLE_MAPS_API_KEY (or GOOGLE_PLACES_API_KEY) env var.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Validate date format early
    if not DATE_COL_RE.fullmatch(args.date):
        raise ValueError(f"--date must be YYYY-MM-DD, got: {args.date}")

    # Resolve API key
    api_key = args.api_key or os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key provided. Use --api-key or set GOOGLE_MAPS_API_KEY (or GOOGLE_PLACES_API_KEY)."
        )

    hotels = list(HOTEL_QUERIES.keys())
    df = ensure_csv(args.csv, args.sep, hotels)

    today_col = args.date
    new_scores: dict[str, float | None] = {}

    print(f"Writing Google ratings into column: {today_col}\n")

    for i, (hotel, query) in enumerate(HOTEL_QUERIES.items(), start=1):
        print(f"{i:02d}/{len(HOTEL_QUERIES)} → {hotel}")
        try:
            score = get_google_rating(query, api_key=api_key, timeout=args.timeout)
        except Exception as e:
            print(f"   ERROR: {e}")
            score = None

        new_scores[hotel] = score

        if score is not None:
            print(f"   {score}/5")
        else:
            print("   (no score)")

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
