from unittest.mock import MagicMock, patch

from src.sites.google import sanitize_google_score, get_google_rating


# ── sanitize_google_score ─────────────────────────────────────────────────────

def test_sanitize_none_returns_none() -> None:
    assert sanitize_google_score(None) is None


def test_sanitize_valid_score() -> None:
    assert sanitize_google_score(4.3) == 4.3


def test_sanitize_boundary_zero() -> None:
    assert sanitize_google_score(0.0) == 0.0


def test_sanitize_boundary_five() -> None:
    assert sanitize_google_score(5.0) == 5.0


def test_sanitize_out_of_range_returns_none() -> None:
    assert sanitize_google_score(5.1) is None
    assert sanitize_google_score(-0.1) is None


def test_sanitize_non_numeric_returns_none() -> None:
    assert sanitize_google_score("bad") is None  # type: ignore[arg-type]


# ── get_google_rating ─────────────────────────────────────────────────────────

def _mock_post(places: list) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"places": places}
    return resp


def test_get_rating_success() -> None:
    place = {"id": "abc", "displayName": {"text": "Hotel"}, "rating": 4.3}
    with patch("src.sites.google.requests.post", return_value=_mock_post([place])):
        result = get_google_rating("Some Hotel, Portugal", api_key="test-key")
    assert result == 4.3


def test_get_rating_empty_places_returns_none() -> None:
    with patch("src.sites.google.requests.post", return_value=_mock_post([])):
        result = get_google_rating("Some Hotel, Portugal", api_key="test-key")
    assert result is None


def test_get_rating_no_rating_field_returns_none() -> None:
    place = {"id": "abc", "displayName": {"text": "Hotel"}}  # no "rating" key
    with patch("src.sites.google.requests.post", return_value=_mock_post([place])):
        result = get_google_rating("Some Hotel, Portugal", api_key="test-key")
    assert result is None


def test_get_rating_empty_query_returns_none() -> None:
    # Should short-circuit without making an HTTP call
    with patch("src.sites.google.requests.post") as mock_post:
        result = get_google_rating("", api_key="test-key")
    assert result is None
    mock_post.assert_not_called()


def test_get_rating_uses_first_place_only() -> None:
    places = [
        {"id": "first", "rating": 3.9},
        {"id": "second", "rating": 4.8},
    ]
    with patch("src.sites.google.requests.post", return_value=_mock_post(places)):
        result = get_google_rating("Hotel", api_key="test-key")
    assert result == 3.9
