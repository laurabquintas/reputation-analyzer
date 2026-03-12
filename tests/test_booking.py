from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from src.sites.booking import sanitize_booking_score, fetch_booking_rating, _extract_score_from_html


# ── sanitize_booking_score ────────────────────────────────────────────────────

def test_sanitize_none_returns_none() -> None:
    assert sanitize_booking_score(None) is None


def test_sanitize_valid_score() -> None:
    assert sanitize_booking_score(8.7) == 8.7


def test_sanitize_boundary_zero() -> None:
    assert sanitize_booking_score(0.0) == 0.0


def test_sanitize_boundary_ten() -> None:
    assert sanitize_booking_score(10.0) == 10.0


def test_sanitize_out_of_range_returns_none() -> None:
    assert sanitize_booking_score(10.1) is None
    assert sanitize_booking_score(-0.1) is None


def test_sanitize_non_numeric_returns_none() -> None:
    assert sanitize_booking_score("not-a-number") is None  # type: ignore[arg-type]


# ── _extract_score_from_html ─────────────────────────────────────────────────

def test_extract_score_from_jsonld() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@context":"https://schema.org","@type":"Hotel",
     "aggregateRating":{"ratingValue":"8.7","bestRating":"10"}}
    </script>
    </head></html>
    """
    assert _extract_score_from_html(html) == 8.7


def test_extract_score_from_jsonld_list() -> None:
    """JSON-LD may be a JSON array; the first matching object should be used."""
    html = """
    <html><head>
    <script type="application/ld+json">
    [{"@type":"BreadcrumbList"},
     {"@type":"Hotel","aggregateRating":{"ratingValue":"9.2","bestRating":"10"}}]
    </script>
    </head></html>
    """
    assert _extract_score_from_html(html) == 9.2


def test_extract_score_comma_decimal_separator() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">
    {"aggregateRating":{"ratingValue":"8,7","bestRating":"10"}}
    </script>
    </head></html>
    """
    assert _extract_score_from_html(html) == 8.7


def test_extract_score_no_aggregate_rating_returns_none() -> None:
    html = "<html><head></head><body>No rating here</body></html>"
    assert _extract_score_from_html(html) is None


def test_extract_score_malformed_jsonld_skipped() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">INVALID JSON</script>
    </head></html>
    """
    assert _extract_score_from_html(html) is None


# ── fetch_booking_rating (mocked Playwright) ─────────────────────────────────

def test_fetch_rating_from_jsonld() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@context":"https://schema.org","@type":"Hotel",
     "aggregateRating":{"ratingValue":"8.7","bestRating":"10"}}
    </script>
    </head></html>
    """
    with patch("src.sites.booking._fetch_page_playwright", return_value=html):
        assert fetch_booking_rating("https://example.com", retries=0) == 8.7


def test_fetch_rating_playwright_failure_returns_none() -> None:
    with patch("src.sites.booking._fetch_page_playwright", return_value=None), \
         patch("src.sites.booking.sleep"):
        assert fetch_booking_rating("https://example.com", retries=0) is None
