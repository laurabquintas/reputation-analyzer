"""Tests for the shared review analysis module."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.reviews import latest_top_reviews, ytd_topic_summary, ytd_topic_insights


SAMPLE_REVIEWS = [
    {
        "id": "1",
        "hotel": "Ananea Castelo Suites Hotel",
        "rating": 5,
        "title": "Amazing",
        "text": "Staff was great, breakfast excellent.",
        "published_date": f"{datetime.now().year}-02-15",
        "topics": [
            {"topic": "employees", "sentiment": "positive", "detail": "friendly staff"},
            {"topic": "meals", "sentiment": "positive", "detail": "excellent breakfast"},
        ],
        "classified": True,
    },
    {
        "id": "2",
        "hotel": "Ananea Castelo Suites Hotel",
        "rating": 3,
        "title": "Mixed",
        "text": "Room was noisy, but staff was helpful. Food was both good and bad.",
        "published_date": f"{datetime.now().year}-02-20",
        "topics": [
            {"topic": "comfort", "sentiment": "negative", "detail": "noisy room"},
            {"topic": "employees", "sentiment": "positive", "detail": "helpful staff"},
            {"topic": "meals", "sentiment": "positive", "detail": "good food"},
            {"topic": "meals", "sentiment": "negative", "detail": "bad food"},
        ],
        "classified": True,
    },
    {
        "id": "3",
        "hotel": "Ananea Castelo Suites Hotel",
        "rating": 4,
        "title": "Good stay",
        "text": "Clean rooms, good value.",
        "published_date": f"{datetime.now().year}-03-01",
        "topics": [
            {"topic": "cleaning", "sentiment": "positive", "detail": "clean rooms"},
            {"topic": "quality_price", "sentiment": "positive", "detail": "good value"},
        ],
        "classified": True,
    },
    {
        "id": "4",
        "hotel": "Ananea Castelo Suites Hotel",
        "rating": 5,
        "title": "Old review",
        "text": "Perfect.",
        "published_date": "2025-06-15",
        "topics": [{"topic": "comfort", "sentiment": "positive", "detail": "perfect stay"}],
        "classified": True,
    },
    {
        "id": "5",
        "hotel": "Other Hotel",
        "rating": 2,
        "title": "Not for us",
        "text": "Bad.",
        "published_date": f"{datetime.now().year}-01-10",
        "topics": [{"topic": "cleaning", "sentiment": "negative", "detail": "dirty rooms"}],
        "classified": True,
    },
    {
        "id": "6",
        "hotel": "Ananea Castelo Suites Hotel",
        "rating": 4,
        "title": "Unclassified",
        "text": "Nice place.",
        "published_date": f"{datetime.now().year}-01-05",
        "topics": [],
        "classified": False,
    },
]


class TestYtdTopicSummary:
    def test_counts_correctly(self):
        df = ytd_topic_summary(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel")
        employees_row = df[df["Topic"] == "Employees"].iloc[0]
        assert employees_row["Positive"] == 2
        assert employees_row["Negative"] == 0

        meals_row = df[df["Topic"] == "Meals"].iloc[0]
        assert meals_row["Positive"] == 2
        assert meals_row["Negative"] == 1

        comfort_row = df[df["Topic"] == "Comfort"].iloc[0]
        assert comfort_row["Positive"] == 0
        assert comfort_row["Negative"] == 1

    def test_filters_by_year(self):
        df = ytd_topic_summary(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel")
        # Review "4" is from 2025 -> its comfort positive should NOT be counted
        comfort_row = df[df["Topic"] == "Comfort"].iloc[0]
        assert comfort_row["Positive"] == 0

    def test_filters_by_hotel(self):
        df = ytd_topic_summary(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel")
        # Review "5" is Other Hotel -> its cleaning negative should NOT be counted
        cleaning_row = df[df["Topic"] == "Cleaning"].iloc[0]
        assert cleaning_row["Negative"] == 0
        assert cleaning_row["Positive"] == 1

    def test_excludes_unclassified(self):
        df = ytd_topic_summary(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel")
        # Review "6" is unclassified -> should not affect any counts
        total = df[["Positive", "Negative"]].sum().sum()
        # employees(2+0) + meals(2+1) + comfort(0+1) + cleaning(1+0) + quality_price(1+0) = 8
        assert total == 8

    def test_explicit_year_filter(self):
        df = ytd_topic_summary(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel", year=2025)
        comfort_row = df[df["Topic"] == "Comfort"].iloc[0]
        assert comfort_row["Positive"] == 1
        assert comfort_row["Negative"] == 0

    def test_all_topics_present(self):
        df = ytd_topic_summary(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel")
        assert len(df) == 7
        expected_topics = {"Employees", "Commodities", "Comfort", "Cleaning", "Quality / Price", "Meals", "Would Return"}
        assert set(df["Topic"].tolist()) == expected_topics


class TestLatestTopReviews:
    def test_sorts_by_date_descending(self):
        result = latest_top_reviews(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel", n=3)
        dates = [r["published_date"] for r in result]
        assert dates == sorted(dates, reverse=True)

    def test_limits_to_n(self):
        result = latest_top_reviews(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel", n=2)
        assert len(result) == 2

    def test_filters_by_hotel(self):
        result = latest_top_reviews(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel", n=10)
        for r in result:
            assert r["hotel"] == "Ananea Castelo Suites Hotel"

    def test_empty_for_unknown_hotel(self):
        result = latest_top_reviews(SAMPLE_REVIEWS, "Unknown Hotel", n=3)
        assert result == []


class TestYtdTopicInsights:
    def test_returns_top_details(self):
        insights = ytd_topic_insights(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel")
        # Two reviews mention employees positive: "friendly staff" and "helpful staff"
        emp_pos = insights.get(("Employees", "positive"), [])
        assert len(emp_pos) == 2
        assert "friendly staff" in emp_pos
        assert "helpful staff" in emp_pos

    def test_limits_to_top_n(self):
        insights = ytd_topic_insights(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel", top_n=1)
        meals_pos = insights.get(("Meals", "positive"), [])
        assert len(meals_pos) == 1

    def test_filters_by_hotel(self):
        insights = ytd_topic_insights(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel")
        # "dirty rooms" is from Other Hotel — should not appear
        cleaning_neg = insights.get(("Cleaning", "negative"), [])
        assert "dirty rooms" not in cleaning_neg

    def test_filters_by_year(self):
        insights = ytd_topic_insights(SAMPLE_REVIEWS, "Ananea Castelo Suites Hotel")
        # Review "4" is from 2025 — "perfect stay" should NOT appear
        comfort_pos = insights.get(("Comfort", "positive"), [])
        assert "perfect stay" not in comfort_pos

    def test_handles_missing_detail(self):
        reviews_no_detail = [
            {
                "id": "x",
                "hotel": "Ananea Castelo Suites Hotel",
                "published_date": f"{datetime.now().year}-01-10",
                "topics": [{"topic": "employees", "sentiment": "positive"}],
                "classified": True,
            },
        ]
        insights = ytd_topic_insights(reviews_no_detail, "Ananea Castelo Suites Hotel")
        assert insights.get(("Employees", "positive"), []) == []

    def test_empty_for_unknown_hotel(self):
        insights = ytd_topic_insights(SAMPLE_REVIEWS, "Unknown Hotel")
        assert insights == {}
