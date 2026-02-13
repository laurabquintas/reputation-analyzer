from pathlib import Path

from src.run import normalize_sites, validate_site_csv


def test_normalize_sites_uppercases_and_filters_empty() -> None:
    raw = [" booking ", "TripAdvisor", "", "  ", "google"]
    assert normalize_sites(raw) == ["BOOKING", "TRIPADVISOR", "GOOGLE"]


def test_validate_site_csv_missing_file(tmp_path: Path) -> None:
    exists, has_date, scored, total = validate_site_csv(
        tmp_path / "missing.csv",
        "2026-02-13",
    )
    assert (exists, has_date, scored, total) == (False, False, 0, 0)


def test_validate_site_csv_detects_missing_date_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "Hotel;Average Score;2026-02-12\n"
        "A;4.5;4.5\n"
        "B;4.0;4.0\n",
        encoding="utf-8",
    )

    exists, has_date, scored, total = validate_site_csv(csv_path, "2026-02-13")
    assert exists is True
    assert has_date is False
    assert scored == 0
    assert total == 2


def test_validate_site_csv_counts_scored_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "Hotel;Average Score;2026-02-13\n"
        "A;4.5;4.5\n"
        "B;4.0;\n"
        "C;3.8;3.8\n",
        encoding="utf-8",
    )

    exists, has_date, scored, total = validate_site_csv(csv_path, "2026-02-13")
    assert exists is True
    assert has_date is True
    assert scored == 2
    assert total == 3
