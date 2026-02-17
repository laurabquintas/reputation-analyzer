from pathlib import Path
from subprocess import CompletedProcess

from src.run import SiteConfig, normalize_sites, run_site, validate_site_csv


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


def test_run_site_warns_when_no_scores_collected(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "Hotel;Average Score;2026-02-13\n"
        "A;4.5;\n"
        "B;4.0;\n",
        encoding="utf-8",
    )
    config = SiteConfig(script=tmp_path / "fake.py", csv_path=csv_path)

    monkeypatch.setattr(
        "src.run.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args=args[0], returncode=0, stdout="", stderr=""),
    )

    result = run_site("BOOKING", config, "2026-02-13", "python3", 10)
    assert result.status == "warning"
    assert result.warning is True
    assert "No scores collected" in result.message


def test_run_site_warns_when_partial_scores_collected(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "Hotel;Average Score;2026-02-13\n"
        "A;4.5;4.5\n"
        "B;4.0;\n"
        "C;3.8;3.8\n",
        encoding="utf-8",
    )
    config = SiteConfig(script=tmp_path / "fake.py", csv_path=csv_path)

    monkeypatch.setattr(
        "src.run.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args=args[0], returncode=0, stdout="", stderr=""),
    )

    result = run_site("BOOKING", config, "2026-02-13", "python3", 10)
    assert result.status == "warning"
    assert result.warning is True
    assert result.message == "Collected scores for 2/3 hotels"


def test_run_site_is_ok_when_all_scores_collected(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "Hotel;Average Score;2026-02-13\n"
        "A;4.5;4.5\n"
        "B;4.0;4.0\n",
        encoding="utf-8",
    )
    config = SiteConfig(script=tmp_path / "fake.py", csv_path=csv_path)

    monkeypatch.setattr(
        "src.run.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args=args[0], returncode=0, stdout="", stderr=""),
    )

    result = run_site("BOOKING", config, "2026-02-13", "python3", 10)
    assert result.status == "ok"
    assert result.warning is False
    assert result.message == "Collected scores for 2/2 hotels"
