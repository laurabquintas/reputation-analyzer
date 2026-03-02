from unittest.mock import MagicMock, patch

import pytest

from src.sites.holidaycheck import (
    _normalize_to_six_scale,
    sanitize_holidaycheck_score,
    get_holidaycheck_score,
)


# ── _normalize_to_six_scale ───────────────────────────────────────────────────

def test_normalize_already_six_scale() -> None:
    assert _normalize_to_six_scale(4.5, best_rating=6) == 4.5


def test_normalize_ten_scale_to_six() -> None:
    result = _normalize_to_six_scale(8.0, best_rating=10)
    assert result == pytest.approx(4.8)


def test_normalize_none_best_rating_is_noop() -> None:
    assert _normalize_to_six_scale(4.5, best_rating=None) == 4.5


def test_normalize_zero_best_rating_is_noop() -> None:
    # Guard against division by zero — score returned unchanged
    assert _normalize_to_six_scale(4.5, best_rating=0) == 4.5


# ── sanitize_holidaycheck_score ───────────────────────────────────────────────

def test_sanitize_none_returns_none() -> None:
    assert sanitize_holidaycheck_score(None) is None


def test_sanitize_valid_score() -> None:
    assert sanitize_holidaycheck_score(4.5) == 4.5


def test_sanitize_clamps_above_six() -> None:
    assert sanitize_holidaycheck_score(7.0) == 6.0


def test_sanitize_clamps_below_zero() -> None:
    assert sanitize_holidaycheck_score(-1.0) == 0.0


def test_sanitize_non_numeric_returns_none() -> None:
    assert sanitize_holidaycheck_score("bad") is None  # type: ignore[arg-type]


# ── get_holidaycheck_score (HTML parsing) ─────────────────────────────────────

def _make_response(html: str) -> MagicMock:
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


def test_get_score_from_jsonld() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@context":"https://schema.org","@type":"Hotel",
     "aggregateRating":{"ratingValue":"4.5","bestRating":"6"}}
    </script>
    </head></html>
    """
    with patch("src.sites.holidaycheck.requests.get", return_value=_make_response(html)):
        assert get_holidaycheck_score("https://example.com") == 4.5


def test_get_score_normalizes_ten_scale_jsonld() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">
    {"aggregateRating":{"ratingValue":"8.0","bestRating":"10"}}
    </script>
    </head></html>
    """
    with patch("src.sites.holidaycheck.requests.get", return_value=_make_response(html)):
        result = get_holidaycheck_score("https://example.com")
    assert result == pytest.approx(4.8)


def test_get_score_fallback_text_pattern() -> None:
    html = "<html><body>Overall rating: 4,5 / 6</body></html>"
    with patch("src.sites.holidaycheck.requests.get", return_value=_make_response(html)):
        assert get_holidaycheck_score("https://example.com") == 4.5


def test_get_score_no_rating_returns_none() -> None:
    html = "<html><body>No score here</body></html>"
    with patch("src.sites.holidaycheck.requests.get", return_value=_make_response(html)):
        assert get_holidaycheck_score("https://example.com") is None


def test_get_score_empty_url_returns_none() -> None:
    with patch("src.sites.holidaycheck.requests.get") as mock_get:
        result = get_holidaycheck_score("")
    assert result is None
    mock_get.assert_not_called()
