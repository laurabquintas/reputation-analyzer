"""Tests for Google Maps Playwright scraper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.sites.google_scraper import (
    _extract_score_from_page,
    _extract_search_score_cards,
    _parse_relative_date,
    _parse_review_element,
    compute_score_from_reviews,
    get_reviews_by_platform,
)


# ── _parse_relative_date ─────────────────────────────────────────────────────

def test_parse_relative_date_months_ago() -> None:
    result = _parse_relative_date("4 months ago on")
    assert result  # Should return a YYYY-MM-DD string
    assert len(result) == 10
    assert result[4] == "-"


def test_parse_relative_date_weeks_ago() -> None:
    result = _parse_relative_date("2 weeks ago on")
    assert result
    assert len(result) == 10


def test_parse_relative_date_year_ago() -> None:
    result = _parse_relative_date("1 year ago on")
    assert result
    assert len(result) == 10


def test_parse_relative_date_no_match() -> None:
    assert _parse_relative_date("") == ""
    assert _parse_relative_date("not a date") == ""


# ── _extract_score_from_page ─────────────────────────────────────────────────

def _mock_page_with_labels(stars_labels: list[tuple[str, bool]], reviews_labels: list[tuple[str, bool]]):
    """Create a mock page with aria-label elements."""
    page = MagicMock()

    # Stars locator
    stars_loc = MagicMock()
    stars_loc.count.return_value = len(stars_labels)
    for i, (label, _visible) in enumerate(stars_labels):
        stars_loc.nth(i).get_attribute.return_value = label
    page.locator.side_effect = lambda sel: (
        stars_loc if "stars" in sel else
        _make_reviews_loc(reviews_labels)
    )
    return page


def _make_reviews_loc(reviews_labels):
    loc = MagicMock()
    loc.count.return_value = len(reviews_labels)
    for i, (label, _visible) in enumerate(reviews_labels):
        loc.nth(i).get_attribute.return_value = label
    return loc


def test_extract_score_from_page_normal() -> None:
    page = _mock_page_with_labels(
        [(" stars", False), ("4.8 stars ", True)],
        [("51 reviews", True)],
    )
    rating, num_reviews = _extract_score_from_page(page)
    assert rating == 4.8
    assert num_reviews == 51


def test_extract_score_from_page_no_rating() -> None:
    page = _mock_page_with_labels(
        [(" stars", False)],
        [("0 reviews", True)],
    )
    rating, num_reviews = _extract_score_from_page(page)
    assert rating is None


# ── _parse_review_element ────────────────────────────────────────────────────

def test_parse_review_element_google_review() -> None:
    text = """Raj
7 reviews

4/5
4 months ago on
Google
I booked this hotel on a whim and it was great. The room was spacious.

Like

Share"""

    review = _parse_review_element(text, review_id="abc123")
    assert review is not None
    assert review["id"] == "abc123"
    assert review["author_name"] == "Raj"
    assert review["rating"] == 4.0
    assert "hotel on a whim" in review["text"]
    assert review["published_date"]  # approximate date
    assert "Like" not in review["text"]
    assert "Share" not in review["text"]


def test_parse_review_element_tripadvisor_review() -> None:
    text = """PippaNel
5/5
4 months ago on
Tripadvisor
We travelled as a family of four and really enjoyed our time at this hotel.
Read more on Tripadvisor"""

    review = _parse_review_element(text, review_id="ta_review_456")
    assert review is not None
    assert review["rating"] == 5.0
    assert review["author_name"] == "PippaNel"
    assert "family of four" in review["text"]
    assert "Read more on Tripadvisor" not in review["text"]


def test_parse_review_element_too_short_returns_none() -> None:
    assert _parse_review_element("short") is None


def test_parse_review_element_no_rating_returns_none() -> None:
    text = "Author Name\nNo rating here\nSome text that has no X/5 pattern"
    assert _parse_review_element(text) is None


def test_parse_review_element_filters_owner_response() -> None:
    text = """Guest
5/5
2 months ago on
Google
Great hotel stay.
Response from the owner 1 month ago
Thank you for your review!"""

    review = _parse_review_element(text, review_id="owner_resp_test")
    assert review is not None
    assert "Great hotel stay" in review["text"]
    assert "Thank you" not in review["text"]


def test_parse_review_element_strips_translated_by_google() -> None:
    """Google Translate footer should be stripped; original_language extracted."""
    text = """Martine Labille
5/5
3 months ago on
Google
Super hôtel, personnel très agréable. Translated by Google ・ See original (French)"""

    review = _parse_review_element(text, review_id="translate_test")
    assert review is not None
    assert "Super hôtel" in review["text"]
    assert "Translated by Google" not in review["text"]
    assert "See original" not in review["text"]
    assert "French" not in review["text"]
    assert review["original_language"] == "french"


def test_parse_review_element_strips_metadata_and_translate_footer() -> None:
    """Structured metadata + Translate footer should be stripped, all fields extracted."""
    text = """Anne b
5/5
2 months ago on
Google
Everything was perfect: the cleanliness, the setting, the attentive staff, and the location. Trip type Vacation Travel group Couple Rooms: 5 Service: 5 Location: 5 Hotel highlights Great value Translated by Google \u30fb See original (French) """

    review = _parse_review_element(text, review_id="metadata_test")
    assert review is not None
    assert "Everything was perfect" in review["text"]
    assert "the location" in review["text"]
    assert "Trip type" not in review["text"]
    assert "Rooms:" not in review["text"]
    assert "Translated by Google" not in review["text"]
    assert review["trip_type"] == "vacation"
    assert review["travel_group"] == "couple"
    assert review["original_language"] == "french"


def test_parse_review_element_no_metadata_empty_fields() -> None:
    """Reviews without metadata should default to 'Unknown' or empty."""
    text = """Guest
4/5
1 month ago on
Google
Nice place to stay."""

    review = _parse_review_element(text, review_id="no_meta_test")
    assert review is not None
    assert review["trip_type"] == "Unknown"
    assert review["travel_group"] == ""
    assert review["original_language"] == ""
    assert review["country"] == "England/USA"


def test_parse_review_element_german_translation() -> None:
    """German translated review should extract original_language='german'."""
    text = """Mobilfunk Guru
5/5
4 months ago on
Google
Tolles Hotel mit super Service. Translated by Google ・ See original (German)"""

    review = _parse_review_element(text, review_id="german_test")
    assert review is not None
    assert "Tolles Hotel" in review["text"]
    assert review["original_language"] == "german"


def test_parse_review_element_rating_only_with_new_badge() -> None:
    """Rating-only review with 'NEW' badge should have empty text, not 'NEW'."""
    text = """Breyner Sanchez
Local guide · 8 reviews · 3 photos
5/5
a week ago in
Google
NEW"""

    review = _parse_review_element(text, review_id="new_badge_test")
    assert review is not None
    assert review["rating"] == 5.0
    assert review["author_name"] == "Breyner Sanchez"
    assert "NEW" not in review["text"]
    assert review["text"] == ""


def test_parse_review_element_no_data_review_id() -> None:
    text = """GuestName
4/5
1 month ago on
Google
Nice place"""

    review = _parse_review_element(text)
    assert review is not None
    assert review["id"]  # Should be a SHA256 hash


# ── fetch_google_rating_playwright ───────────────────────────────────────────

def test_fetch_rating_playwright_success() -> None:
    from src.sites.google_scraper import fetch_google_rating_playwright

    with patch("src.sites.google_scraper._sync_fetch_reviews"), \
         patch("src.sites.google_scraper.sync_playwright") as mock_pw:
        # Set up mock chain
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_pw_instance = mock_pw.return_value.__enter__.return_value
        mock_pw_instance.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        # Mock score extraction
        with patch("src.sites.google_scraper._extract_score_from_page", return_value=(4.5, 100)), \
             patch("src.sites.google_scraper._dismiss_google_consent"), \
             patch("src.sites.google_scraper.time"):
            rating, num_reviews = fetch_google_rating_playwright("https://maps.example.com", retries=0)

    assert rating == 4.5
    assert num_reviews == 100


def test_fetch_rating_playwright_failure() -> None:
    from src.sites.google_scraper import fetch_google_rating_playwright

    with patch("src.sites.google_scraper.sync_playwright", side_effect=Exception("fail")), \
         patch("src.sites.google_scraper.time"):
        rating, num_reviews = fetch_google_rating_playwright("https://maps.example.com", retries=0)

    assert rating is None
    assert num_reviews is None


# ── compute_score_from_reviews ────────────────────────────────────────────

def test_compute_score_from_reviews_normal() -> None:
    reviews = [
        {"rating": 5.0, "text": "Great"},
        {"rating": 4.0, "text": "Good"},
        {"rating": 3.0, "text": "OK"},
    ]
    avg, count = compute_score_from_reviews(reviews)
    assert avg == 4.0
    assert count == 3


def test_compute_score_from_reviews_empty() -> None:
    avg, count = compute_score_from_reviews([])
    assert avg is None
    assert count == 0


def test_compute_score_from_reviews_none_ratings() -> None:
    reviews = [
        {"rating": None, "text": "No rating"},
        {"rating": 5.0, "text": "Good"},
    ]
    avg, count = compute_score_from_reviews(reviews)
    assert avg == 5.0
    assert count == 1


# ── get_reviews_by_platform ──────────────────────────────────────────────

def test_get_reviews_by_platform_filters_tripadvisor() -> None:
    mixed_reviews = [
        {"id": "g1", "rating": 4.0, "source_platform": "google", "text": "Google review"},
        {"id": "ta1", "rating": 5.0, "source_platform": "tripadvisor", "text": "TA review 1"},
        {"id": "g2", "rating": 3.0, "source_platform": "google", "text": "Another Google"},
        {"id": "ta2", "rating": 4.0, "source_platform": "tripadvisor", "text": "TA review 2"},
    ]

    with patch("src.sites.google_scraper.google_get_reviews_playwright", return_value=mixed_reviews):
        result = get_reviews_by_platform("https://maps.example.com", platform="tripadvisor")

    assert len(result) == 2
    assert all(r["source_platform"] == "tripadvisor" for r in result)
    assert result[0]["id"] == "ta1"
    assert result[1]["id"] == "ta2"


def test_get_reviews_by_platform_filters_google() -> None:
    mixed_reviews = [
        {"id": "g1", "rating": 4.0, "source_platform": "google", "text": "Google review"},
        {"id": "ta1", "rating": 5.0, "source_platform": "tripadvisor", "text": "TA review"},
    ]

    with patch("src.sites.google_scraper.google_get_reviews_playwright", return_value=mixed_reviews):
        result = get_reviews_by_platform("https://maps.example.com", platform="google")

    assert len(result) == 1
    assert result[0]["source_platform"] == "google"


def test_get_reviews_by_platform_no_matches() -> None:
    reviews = [
        {"id": "g1", "rating": 4.0, "source_platform": "google", "text": "Google only"},
    ]

    with patch("src.sites.google_scraper.google_get_reviews_playwright", return_value=reviews):
        result = get_reviews_by_platform("https://maps.example.com", platform="tripadvisor")

    assert len(result) == 0


# ── _parse_review_element source_platform ─────────────────────────────────

def test_parse_review_element_captures_source_platform_google() -> None:
    text = """AuthorName
4/5
2 weeks ago on
Google
Nice hotel with good service."""

    review = _parse_review_element(text, review_id="g_review_1")
    assert review is not None
    assert review["source_platform"] == "google"


# ── _extract_search_score_cards ─────────────────────────────────────────

def test_extract_search_score_cards_portuguese() -> None:
    """Score cards in Portuguese (comma decimal, 'críticas')."""
    page = MagicMock()
    page.inner_text.return_value = (
        "Outras opções\n"
        "Críticas\n"
        "As críticas não foram validadas\n"
        "Tripadvisor\n"
        "4,7/5\n"
        "18 críticas\n"
        "Google\n"
        "4,8/5\n"
        "51 críticas\n"
        "Adicione uma opinião\n"
    )
    cards = _extract_search_score_cards(page)
    assert "tripadvisor" in cards
    assert "google" in cards
    assert cards["tripadvisor"]["score"] == 4.7
    assert cards["tripadvisor"]["num_reviews"] == 18
    assert cards["google"]["score"] == 4.8
    assert cards["google"]["num_reviews"] == 51


def test_extract_search_score_cards_english() -> None:
    """Score cards in English (period decimal, 'reviews')."""
    page = MagicMock()
    page.inner_text.return_value = (
        "Reviews\n"
        "Tripadvisor\n"
        "4.7/5\n"
        "18 reviews\n"
        "Google\n"
        "4.8/5\n"
        "51 reviews\n"
    )
    cards = _extract_search_score_cards(page)
    assert cards["tripadvisor"]["score"] == 4.7
    assert cards["google"]["score"] == 4.8


def test_extract_search_score_cards_single_platform() -> None:
    """Only one platform score card present."""
    page = MagicMock()
    page.inner_text.return_value = (
        "Reviews\n"
        "Google\n"
        "4.5/5\n"
        "30 reviews\n"
    )
    cards = _extract_search_score_cards(page)
    assert "google" in cards
    assert cards["google"]["score"] == 4.5
    assert "tripadvisor" not in cards


def test_extract_search_score_cards_no_cards() -> None:
    """No score cards on page."""
    page = MagicMock()
    page.inner_text.return_value = "Some random search results without any score cards"
    cards = _extract_search_score_cards(page)
    assert cards == {}


def test_extract_search_score_cards_score_out_of_range() -> None:
    """Score > 5.0 should be ignored."""
    page = MagicMock()
    page.inner_text.return_value = (
        "Google\n"
        "6,0/5\n"
        "10 reviews\n"
    )
    cards = _extract_search_score_cards(page)
    assert cards == {}


def test_parse_review_element_captures_source_platform_tripadvisor() -> None:
    text = """TravelerName
5/5
1 month ago on
Tripadvisor
Wonderful experience at this hotel."""

    review = _parse_review_element(text, review_id="ta_review_1")
    assert review is not None
    assert review["source_platform"] == "tripadvisor"
