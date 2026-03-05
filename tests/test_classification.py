"""Tests for the classification parser."""

from __future__ import annotations

from src.classification import _parse_classification


class TestParseClassificationDetail:
    def test_extracts_detail_field(self):
        raw = '[{"topic":"employees","sentiment":"positive","detail":"friendly staff"}]'
        result = _parse_classification(raw)
        assert len(result) == 1
        assert result[0]["topic"] == "employees"
        assert result[0]["sentiment"] == "positive"
        assert result[0]["detail"] == "friendly staff"

    def test_missing_detail_omitted(self):
        raw = '[{"topic":"meals","sentiment":"negative"}]'
        result = _parse_classification(raw)
        assert len(result) == 1
        assert "detail" not in result[0]

    def test_empty_detail_omitted(self):
        raw = '[{"topic":"meals","sentiment":"positive","detail":""}]'
        result = _parse_classification(raw)
        assert len(result) == 1
        assert "detail" not in result[0]

    def test_multiple_entries_with_detail(self):
        raw = (
            '[{"topic":"employees","sentiment":"positive","detail":"helpful reception"},'
            '{"topic":"meals","sentiment":"negative","detail":"repetitive breakfast"}]'
        )
        result = _parse_classification(raw)
        assert len(result) == 2
        assert result[0]["detail"] == "helpful reception"
        assert result[1]["detail"] == "repetitive breakfast"

    def test_mixed_with_and_without_detail(self):
        raw = (
            '[{"topic":"comfort","sentiment":"positive","detail":"spacious rooms"},'
            '{"topic":"cleaning","sentiment":"positive"}]'
        )
        result = _parse_classification(raw)
        assert len(result) == 2
        assert result[0]["detail"] == "spacious rooms"
        assert "detail" not in result[1]

    def test_backward_compatible_no_detail(self):
        raw = (
            '[{"topic":"employees","sentiment":"positive"},'
            '{"topic":"meals","sentiment":"positive"}]'
        )
        result = _parse_classification(raw)
        assert len(result) == 2
        for item in result:
            assert "detail" not in item
