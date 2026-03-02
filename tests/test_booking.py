from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from src.sites.booking import sanitize_booking_score, fetch_booking_rating


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


# ── fetch_booking_rating ──────────────────────────────────────────────────────

def _make_session(html: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = html
    resp.raise_for_status = MagicMock()
    session = MagicMock()
    session.get.return_value = resp
    return session


def test_fetch_rating_from_jsonld() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@context":"https://schema.org","@type":"Hotel",
     "aggregateRating":{"ratingValue":"8.7","bestRating":"10"}}
    </script>
    </head></html>
    """
    session = _make_session(html)
    assert fetch_booking_rating("https://example.com", session, retries=0) == 8.7


def test_fetch_rating_from_jsonld_list() -> None:
    """JSON-LD may be a JSON array; the first matching object should be used."""
    html = """
    <html><head>
    <script type="application/ld+json">
    [{"@type":"BreadcrumbList"},
     {"@type":"Hotel","aggregateRating":{"ratingValue":"9.2","bestRating":"10"}}]
    </script>
    </head></html>
    """
    session = _make_session(html)
    assert fetch_booking_rating("https://example.com", session, retries=0) == 9.2


def test_fetch_rating_comma_decimal_separator() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">
    {"aggregateRating":{"ratingValue":"8,7","bestRating":"10"}}
    </script>
    </head></html>
    """
    session = _make_session(html)
    assert fetch_booking_rating("https://example.com", session, retries=0) == 8.7


def test_fetch_rating_no_aggregate_rating_returns_none() -> None:
    html = "<html><head></head><body>No rating here</body></html>"
    session = _make_session(html)
    assert fetch_booking_rating("https://example.com", session, retries=0) is None


def test_fetch_rating_malformed_jsonld_skipped() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">INVALID JSON</script>
    </head></html>
    """
    session = _make_session(html)
    assert fetch_booking_rating("https://example.com", session, retries=0) is None


def test_fetch_rating_http_error_returns_none() -> None:
    import requests as req
    resp = MagicMock()
    resp.raise_for_status.side_effect = req.HTTPError("404")
    session = MagicMock()
    session.get.return_value = resp
    # retries=0 so only one attempt; should catch the exception and return None
    with patch("src.sites.booking.sleep"):
        assert fetch_booking_rating("https://example.com", session, retries=0) is None
