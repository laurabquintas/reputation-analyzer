"""
Integration / smoke tests.

Each test feeds a realistic HTML or JSON response through the full extraction
pipeline of a scraper and asserts that a valid score is returned.  No real
HTTP requests are made — responses are constructed inline so the tests run
instantly and deterministically.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

# ── Booking: full pipeline HTML → JSON-LD → score ────────────────────────────

from src.sites.booking import fetch_booking_rating, sanitize_booking_score

BOOKING_HTML = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Hotel","name":"Ananea Castelo Suites Hotel",
 "aggregateRating":{"@type":"AggregateRating","ratingValue":"8.7","bestRating":"10","reviewCount":"312"}}
</script>
</head><body></body></html>
"""


def test_booking_full_pipeline() -> None:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = BOOKING_HTML
    resp.raise_for_status = MagicMock()
    session = MagicMock()
    session.get.return_value = resp

    score = fetch_booking_rating("https://www.booking.com/hotel/pt/castelo-suites.en-gb.html", session, retries=0)
    score = sanitize_booking_score(score)
    assert score == 8.7


# ── HolidayCheck: full pipeline HTML → JSON-LD → normalize → score ───────────

from src.sites.holidaycheck import get_holidaycheck_score, sanitize_holidaycheck_score

HOLIDAYCHECK_HTML = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Hotel","name":"Ananea Castelo Suites Algarve",
 "aggregateRating":{"@type":"AggregateRating","ratingValue":"4.8","bestRating":"6","reviewCount":"45"}}
</script>
</head><body></body></html>
"""

HOLIDAYCHECK_HTML_TEXT_FALLBACK = """
<html><body>
<div class="rating">Gesamtbewertung 4,5 / 6</div>
</body></html>
"""


def test_holidaycheck_full_pipeline_jsonld() -> None:
    resp = MagicMock()
    resp.text = HOLIDAYCHECK_HTML
    resp.raise_for_status = MagicMock()

    with patch("src.sites.holidaycheck.requests.get", return_value=resp):
        score = get_holidaycheck_score("https://www.holidaycheck.de/hi/example/abc")
    score = sanitize_holidaycheck_score(score)
    assert score == 4.8


def test_holidaycheck_full_pipeline_text_fallback() -> None:
    resp = MagicMock()
    resp.text = HOLIDAYCHECK_HTML_TEXT_FALLBACK
    resp.raise_for_status = MagicMock()

    with patch("src.sites.holidaycheck.requests.get", return_value=resp):
        score = get_holidaycheck_score("https://www.holidaycheck.de/hi/example/abc")
    score = sanitize_holidaycheck_score(score)
    assert score == 4.5


# ── Expedia: full pipeline HTML → multi-strategy extraction → score ──────────

from src.sites.expedia import get_expedia_score, validate_expedia_score

EXPEDIA_HTML_JSONLD = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Hotel","name":"Castelo Suites Hotel",
 "aggregateRating":{"@type":"AggregateRating","ratingValue":"8.6","bestRating":"10","reviewCount":"200"}}
</script>
</head><body></body></html>
"""

EXPEDIA_HTML_SEMANTIC_DIV = """
<html><body>
<section id="Reviews">
<div class="uitk-text uitk-type-900 uitk-text-default-theme">8.4</div>
</section>
</body></html>
"""

EXPEDIA_HTML_TEXTUAL = """
<html><body>
<div>Guest rating 8.6 out of 10 based on 200 reviews</div>
</body></html>
"""


def test_expedia_full_pipeline_jsonld() -> None:
    with patch("src.sites.expedia.fetch_page", return_value=EXPEDIA_HTML_JSONLD):
        score = get_expedia_score("https://euro.expedia.net/example", retries=0)
    score = validate_expedia_score(score)
    assert score == 8.6


def test_expedia_full_pipeline_semantic_div() -> None:
    with patch("src.sites.expedia.fetch_page", return_value=EXPEDIA_HTML_SEMANTIC_DIV):
        score = get_expedia_score("https://euro.expedia.net/example", retries=0)
    score = validate_expedia_score(score)
    assert score == 8.4


def test_expedia_full_pipeline_textual() -> None:
    with patch("src.sites.expedia.fetch_page", return_value=EXPEDIA_HTML_TEXTUAL):
        score = get_expedia_score("https://euro.expedia.net/example", retries=0)
    score = validate_expedia_score(score)
    assert score == 8.6


# ── Google: full pipeline API JSON → score ───────────────────────────────────

from src.sites.google import get_google_rating, sanitize_google_score


def test_google_full_pipeline() -> None:
    api_response = {
        "places": [
            {
                "id": "ChIJabc123",
                "displayName": {"text": "Ananea Castelo Suites Hotel", "languageCode": "en"},
                "rating": 4.5,
                "userRatingCount": 312,
            }
        ]
    }
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = api_response

    with patch("src.sites.google.requests.post", return_value=resp):
        score = get_google_rating("Ananea Castelo Suites Algarve, Portugal", api_key="test-key")
    score = sanitize_google_score(score)
    assert score == 4.5


# ── TripAdvisor: full pipeline API JSON → score ─────────────────────────────

from src.sites.tripadvisor import ta_get_rating, sanitize_tripadvisor_score


def test_tripadvisor_full_pipeline() -> None:
    api_response = {
        "location_id": "33299137",
        "name": "Ananea Castelo Suites Hotel",
        "rating": "4.5",
        "num_reviews": "120",
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = api_response

    with patch("src.sites.tripadvisor.requests.get", return_value=resp):
        rating, num_reviews = ta_get_rating("33299137", api_key="test-key")
    rating = sanitize_tripadvisor_score(rating)
    assert rating == 4.5
    assert num_reviews == "120"


# ── Cross-scraper: no score gracefully returns None ──────────────────────────

EMPTY_HTML = "<html><head></head><body>Nothing useful here</body></html>"


def test_booking_no_score_returns_none() -> None:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = EMPTY_HTML
    resp.raise_for_status = MagicMock()
    session = MagicMock()
    session.get.return_value = resp

    score = fetch_booking_rating("https://www.booking.com/hotel/pt/example", session, retries=0)
    assert score is None


def test_holidaycheck_no_score_returns_none() -> None:
    resp = MagicMock()
    resp.text = EMPTY_HTML
    resp.raise_for_status = MagicMock()

    with patch("src.sites.holidaycheck.requests.get", return_value=resp):
        score = get_holidaycheck_score("https://www.holidaycheck.de/hi/example/abc")
    assert score is None


def test_expedia_no_score_returns_none() -> None:
    with patch("src.sites.expedia.fetch_page", return_value=EMPTY_HTML):
        score = get_expedia_score("https://euro.expedia.net/example", retries=0)
    assert score is None


def test_google_no_places_returns_none() -> None:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"places": []}

    with patch("src.sites.google.requests.post", return_value=resp):
        score = get_google_rating("Unknown Hotel", api_key="test-key")
    assert score is None


def test_tripadvisor_no_rating_returns_none() -> None:
    api_response = {
        "location_id": "12345",
        "name": "Unknown Hotel",
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = api_response

    with patch("src.sites.tripadvisor.requests.get", return_value=resp):
        rating, num_reviews = ta_get_rating("12345", api_key="test-key")
    assert rating is None
