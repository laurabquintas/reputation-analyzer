"""Shared review analysis functions used by the dashboard."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

TOPIC_DISPLAY = {
    "employees": "Employees",
    "commodities": "Commodities",
    "comfort": "Comfort",
    "cleaning": "Cleaning",
    "quality_price": "Quality / Price",
    "meals": "Meals",
    "return": "Would Return",
}


def load_reviews(json_path: Path) -> list[dict]:
    if not json_path.exists():
        return []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("reviews", [])


def ytd_topic_summary(reviews: list[dict], hotel: str, year: int | None = None) -> pd.DataFrame:
    """Aggregate positive/negative mention counts per topic for YTD reviews of a hotel.

    A single review can contribute both a positive AND negative count
    for the same topic.
    """
    if year is None:
        year = datetime.now().year

    ytd_reviews = [
        r for r in reviews
        if r.get("hotel") == hotel
        and r.get("published_date", "")[:4] == str(year)
        and r.get("classified", False)
    ]

    rows = []
    for topic_key, topic_display in TOPIC_DISPLAY.items():
        pos = sum(
            1 for r in ytd_reviews
            for t in r.get("topics", [])
            if t["topic"] == topic_key and t["sentiment"] == "positive"
        )
        neg = sum(
            1 for r in ytd_reviews
            for t in r.get("topics", [])
            if t["topic"] == topic_key and t["sentiment"] == "negative"
        )
        rows.append({"Topic": topic_display, "Positive": pos, "Negative": neg})

    return pd.DataFrame(rows)


def ytd_topic_insights(
    reviews: list[dict], hotel: str, year: int | None = None, top_n: int = 2
) -> dict[tuple[str, str], list[str]]:
    """Return the top-N most frequent detail phrases per (display_topic, sentiment).

    Returns a dict like::

        {("Employees", "positive"): ["friendly staff", "helpful reception"],
         ("Employees", "negative"): ["slow check-in"], ...}
    """
    if year is None:
        year = datetime.now().year

    ytd_reviews = [
        r for r in reviews
        if r.get("hotel") == hotel
        and r.get("published_date", "")[:4] == str(year)
        and r.get("classified", False)
    ]

    counters: dict[tuple[str, str], Counter] = {}
    for r in ytd_reviews:
        for t in r.get("topics", []):
            detail = t.get("detail", "").strip().lower()
            if not detail:
                continue
            topic_key = t.get("topic", "")
            sentiment = t.get("sentiment", "")
            display = TOPIC_DISPLAY.get(topic_key)
            if display and sentiment in ("positive", "negative"):
                key = (display, sentiment)
                if key not in counters:
                    counters[key] = Counter()
                counters[key][detail] += 1

    return {
        key: [phrase for phrase, _ in counter.most_common(top_n)]
        for key, counter in counters.items()
    }


def latest_top_reviews(reviews: list[dict], hotel: str, n: int = 3) -> list[dict]:
    """Get the n most recent reviews for a hotel, sorted by date descending."""
    hotel_reviews = [
        r for r in reviews
        if r.get("hotel") == hotel
    ]
    hotel_reviews.sort(key=lambda r: r.get("published_date", ""), reverse=True)
    return hotel_reviews[:n]
