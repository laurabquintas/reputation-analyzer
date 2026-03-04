"""Tests for the Google reviews scraper."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.sites.google_reviews import (
    _extract_publish_date,
    _extract_review_id,
    _extract_review_text,
    deduplicate_reviews,
    google_get_reviews,
    load_reviews,
    save_reviews,
)


# ---- _extract_review_id ----

class TestExtractReviewId:
    def test_from_resource_name(self):
        review = {"name": "places/ChIJ123/reviews/abc456"}
        assert _extract_review_id(review) == "abc456"

    def test_simple_name(self):
        review = {"name": "abc456"}
        assert _extract_review_id(review) == "abc456"

    def test_empty_name(self):
        review = {"name": ""}
        assert _extract_review_id(review) == ""

    def test_missing_name(self):
        review = {}
        assert _extract_review_id(review) == ""


# ---- _extract_review_text ----

class TestExtractReviewText:
    def test_prefers_original_text(self):
        review = {
            "originalText": {"text": "Texto original", "languageCode": "pt"},
            "text": {"text": "Translated text", "languageCode": "en"},
        }
        assert _extract_review_text(review) == "Texto original"

    def test_falls_back_to_text(self):
        review = {
            "text": {"text": "Only translated", "languageCode": "en"},
        }
        assert _extract_review_text(review) == "Only translated"

    def test_empty_original_falls_back(self):
        review = {
            "originalText": {"text": "", "languageCode": "pt"},
            "text": {"text": "Translated text", "languageCode": "en"},
        }
        assert _extract_review_text(review) == "Translated text"

    def test_no_text_at_all(self):
        review = {}
        assert _extract_review_text(review) == ""


# ---- _extract_publish_date ----

class TestExtractPublishDate:
    def test_iso_format(self):
        review = {"publishTime": "2026-02-15T10:30:00Z"}
        assert _extract_publish_date(review) == "2026-02-15"

    def test_iso_with_offset(self):
        review = {"publishTime": "2026-01-20T08:00:00+01:00"}
        assert _extract_publish_date(review) == "2026-01-20"

    def test_empty(self):
        review = {"publishTime": ""}
        assert _extract_publish_date(review) == ""

    def test_missing(self):
        review = {}
        assert _extract_publish_date(review) == ""


# ---- deduplicate_reviews ----

class TestDeduplicateReviews:
    def test_no_duplicates(self):
        existing = [{"id": "1", "hotel": "A"}, {"id": "2", "hotel": "A"}]
        new = [{"id": "3", "hotel": "A"}]
        merged = deduplicate_reviews(existing, new)
        assert len(merged) == 3

    def test_skips_duplicates(self):
        existing = [{"id": "1", "hotel": "A"}, {"id": "2", "hotel": "A"}]
        new = [{"id": "2", "hotel": "A"}, {"id": "3", "hotel": "A"}]
        merged = deduplicate_reviews(existing, new)
        assert len(merged) == 3
        assert [r["id"] for r in merged] == ["1", "2", "3"]

    def test_preserves_order(self):
        existing = [{"id": "3"}, {"id": "1"}, {"id": "2"}]
        new = [{"id": "4"}]
        merged = deduplicate_reviews(existing, new)
        assert [r["id"] for r in merged] == ["3", "1", "2", "4"]

    def test_empty_existing(self):
        merged = deduplicate_reviews([], [{"id": "1"}])
        assert len(merged) == 1

    def test_empty_new(self):
        merged = deduplicate_reviews([{"id": "1"}], [])
        assert len(merged) == 1


# ---- load_reviews / save_reviews roundtrip ----

class TestLoadSaveReviews:
    def test_roundtrip(self, tmp_path: Path):
        json_path = str(tmp_path / "reviews.json")
        reviews = [
            {"id": "1", "hotel": "Test", "text": "Great", "topics": [], "classified": True},
            {"id": "2", "hotel": "Test", "text": "Bad", "topics": [], "classified": False},
        ]
        save_reviews(reviews, json_path)
        loaded = load_reviews(json_path)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "1"
        assert loaded[1]["id"] == "2"

    def test_load_nonexistent(self, tmp_path: Path):
        result = load_reviews(str(tmp_path / "missing.json"))
        assert result == []


# ---- google_get_reviews (mocked HTTP) ----

SAMPLE_API_RESPONSE = {
    "places": [
        {
            "id": "ChIJ_test_id",
            "displayName": {"text": "Ananea Castelo Suites Hotel"},
            "reviews": [
                {
                    "name": "places/ChIJ_test_id/reviews/review123",
                    "rating": 5,
                    "text": {"text": "Amazing hotel, loved the breakfast!", "languageCode": "en"},
                    "originalText": {"text": "Amazing hotel, loved the breakfast!", "languageCode": "en"},
                    "publishTime": "2026-02-28T12:00:00Z",
                    "authorAttribution": {
                        "displayName": "John Doe",
                        "uri": "https://maps.google.com/...",
                    },
                },
                {
                    "name": "places/ChIJ_test_id/reviews/review456",
                    "rating": 3,
                    "text": {"text": "Average stay, room was small.", "languageCode": "en"},
                    "publishTime": "2026-02-20T08:30:00Z",
                    "authorAttribution": {
                        "displayName": "Jane Smith",
                    },
                },
            ],
        }
    ]
}


class TestGoogleGetReviews:
    def test_returns_reviews(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = SAMPLE_API_RESPONSE

        with patch("src.sites.google_reviews.requests.post", return_value=resp):
            reviews = google_get_reviews("Ananea Castelo Suites Algarve", api_key="test-key")

        assert len(reviews) == 2
        assert reviews[0]["name"] == "places/ChIJ_test_id/reviews/review123"
        assert reviews[0]["rating"] == 5

    def test_empty_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"places": []}

        with patch("src.sites.google_reviews.requests.post", return_value=resp):
            reviews = google_get_reviews("NonexistentHotel", api_key="test-key")

        assert reviews == []

    def test_no_reviews_field(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "places": [{"id": "ChIJ_test", "displayName": {"text": "Test"}}]
        }

        with patch("src.sites.google_reviews.requests.post", return_value=resp):
            reviews = google_get_reviews("TestHotel", api_key="test-key")

        assert reviews == []
