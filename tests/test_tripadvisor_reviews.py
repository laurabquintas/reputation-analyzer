"""Tests for the TripAdvisor reviews scraper and Ollama classifier."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.classification import (
    VALID_SENTIMENTS,
    VALID_TOPICS,
    _parse_classification,
    is_ollama_available,
)
from src.sites.tripadvisor_reviews import (
    deduplicate_reviews,
    load_reviews,
    save_reviews,
    ta_get_reviews,
    ta_get_reviews_page,
)


# ---- _parse_classification ----

class TestParseClassification:
    def test_valid_json(self):
        raw = '[{"topic": "employees", "sentiment": "positive"}, {"topic": "meals", "sentiment": "negative"}]'
        result = _parse_classification(raw)
        assert len(result) == 2
        assert result[0] == {"topic": "employees", "sentiment": "positive"}
        assert result[1] == {"topic": "meals", "sentiment": "negative"}

    def test_code_fences(self):
        raw = '```json\n[{"topic": "comfort", "sentiment": "negative"}]\n```'
        result = _parse_classification(raw)
        assert len(result) == 1
        assert result[0] == {"topic": "comfort", "sentiment": "negative"}

    def test_invalid_topic_ignored(self):
        raw = '[{"topic": "location", "sentiment": "positive"}, {"topic": "employees", "sentiment": "positive"}]'
        result = _parse_classification(raw)
        assert len(result) == 1
        assert result[0]["topic"] == "employees"

    def test_invalid_sentiment_ignored(self):
        raw = '[{"topic": "employees", "sentiment": "neutral"}]'
        result = _parse_classification(raw)
        assert len(result) == 0

    def test_same_topic_both_sentiments_allowed(self):
        raw = '[{"topic": "meals", "sentiment": "positive"}, {"topic": "meals", "sentiment": "negative"}]'
        result = _parse_classification(raw)
        assert len(result) == 2
        assert result[0] == {"topic": "meals", "sentiment": "positive"}
        assert result[1] == {"topic": "meals", "sentiment": "negative"}

    def test_duplicate_pair_deduplicated(self):
        raw = '[{"topic": "meals", "sentiment": "positive"}, {"topic": "meals", "sentiment": "positive"}]'
        result = _parse_classification(raw)
        assert len(result) == 1

    def test_empty_response(self):
        assert _parse_classification("") == []
        assert _parse_classification("no json here") == []

    def test_json_embedded_in_text(self):
        raw = 'Here is the result:\n[{"topic": "cleaning", "sentiment": "positive"}]\nDone.'
        result = _parse_classification(raw)
        assert len(result) == 1
        assert result[0]["topic"] == "cleaning"


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


# ---- ta_get_reviews (mocked HTTP) ----

SAMPLE_API_RESPONSE = {
    "data": [
        {
            "id": 12345,
            "rating": 5,
            "title": "Great stay",
            "text": "The staff was amazing and breakfast was delicious.",
            "published_date": "2026-02-28",
            "travel_date": "2026-02-15",
            "trip_type": "COUPLES",
            "subratings": [
                {"name": "Service", "value": 5},
                {"name": "Cleanliness", "value": 5},
            ],
            "helpful_votes": 2,
        }
    ],
    "paging": {"total_results": 1, "results": 1},
}


class TestTaGetReviewsPage:
    def test_returns_reviews_and_total(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = SAMPLE_API_RESPONSE

        with patch("src.sites.tripadvisor_reviews.requests.get", return_value=resp):
            reviews, total = ta_get_reviews_page("33299137", api_key="test-key")

        assert len(reviews) == 1
        assert reviews[0]["id"] == 12345
        assert total == 1

    def test_empty_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"data": [], "paging": {"total_results": 0}}

        with patch("src.sites.tripadvisor_reviews.requests.get", return_value=resp):
            reviews, total = ta_get_reviews_page("33299137", api_key="test-key")

        assert reviews == []
        assert total == 0


class TestTaGetReviews:
    def test_returns_reviews_across_languages(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = SAMPLE_API_RESPONSE

        with patch("src.sites.tripadvisor_reviews.requests.get", return_value=resp):
            with patch("src.sites.tripadvisor_reviews.sleep"):
                reviews = ta_get_reviews(
                    "33299137", api_key="test-key", languages=["en"]
                )

        assert len(reviews) == 1
        assert reviews[0]["id"] == 12345
        assert reviews[0]["_language"] == "en"

    def test_deduplicates_across_languages(self):
        """Same review ID returned in en and pt should appear only once."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = SAMPLE_API_RESPONSE

        with patch("src.sites.tripadvisor_reviews.requests.get", return_value=resp):
            with patch("src.sites.tripadvisor_reviews.sleep"):
                reviews = ta_get_reviews(
                    "33299137", api_key="test-key", languages=["en", "pt"]
                )

        assert len(reviews) == 1

    def test_empty_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"data": [], "paging": {"total_results": 0}}

        with patch("src.sites.tripadvisor_reviews.requests.get", return_value=resp):
            with patch("src.sites.tripadvisor_reviews.sleep"):
                reviews = ta_get_reviews(
                    "33299137", api_key="test-key", languages=["en"]
                )

        assert reviews == []


# ---- is_ollama_available ----

class TestIsOllamaAvailable:
    def test_returns_false_on_connection_error(self):
        with patch("src.classification.requests.get", side_effect=Exception("refused")):
            assert is_ollama_available("http://localhost:11434") is False

    def test_returns_true_on_success(self):
        resp = MagicMock()
        resp.status_code = 200
        with patch("src.classification.requests.get", return_value=resp):
            assert is_ollama_available("http://localhost:11434") is True
