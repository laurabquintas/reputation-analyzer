from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


@dataclass
class SiteConfig:
    script: Path
    csv_path: Path
    required_env: tuple[str, ...] = ()


@dataclass
class SiteResult:
    site: str
    status: str
    message: str
    warning: bool
    returncode: int
    duration_seconds: float
    csv_path: str
    has_date_column: bool
    scored_hotels: int
    total_hotels: int


SITE_CONFIGS: dict[str, SiteConfig] = {
    "BOOKING": SiteConfig(
        script=ROOT / "src" / "sites" / "booking.py",
        csv_path=DATA_DIR / "booking_scores.csv",
    ),
    "TRIPADVISOR": SiteConfig(
        script=ROOT / "src" / "sites" / "tripadvisor.py",
        csv_path=DATA_DIR / "tripadvisor_scores.csv",
        required_env=("TRIPADVISOR_API_KEY",),
    ),
    "GOOGLE": SiteConfig(
        script=ROOT / "src" / "sites" / "google.py",
        csv_path=DATA_DIR / "google_scores.csv",
    ),
    "EXPEDIA": SiteConfig(
        script=ROOT / "src" / "sites" / "expedia.py",
        csv_path=DATA_DIR / "expedia_scores.csv",
    ),
    "HOLIDAYCHECK": SiteConfig(
        script=ROOT / "src" / "sites" / "holidaycheck.py",
        csv_path=DATA_DIR / "holidaycheck_scores.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run site scrapers and validate data CSV updates.")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date column to expect in CSVs (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=list(SITE_CONFIGS.keys()),
        help=f"Sites to run (default: {' '.join(SITE_CONFIGS.keys())}).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write a JSON execution summary.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run site scripts.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds per site script.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit non-zero if any site produced a warning.",
    )
    return parser.parse_args()


def normalize_sites(raw_sites: list[str]) -> list[str]:
    return [site.strip().upper() for site in raw_sites if site.strip()]


def validate_site_csv(csv_path: Path, date_col: str, sep: str = ";") -> tuple[bool, bool, int, int]:
    if not csv_path.exists():
        return False, False, 0, 0

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        headers = reader.fieldnames or []
        if date_col not in headers:
            total_hotels = sum(1 for _ in reader)
            return True, False, 0, total_hotels

        scored = 0
        total = 0
        for row in reader:
            total += 1
            value = row.get(date_col)
            if value is not None and str(value).strip() != "":
                scored += 1
        return True, True, scored, total


def run_site(site: str, config: SiteConfig, date_col: str, python_cmd: str, timeout: int) -> SiteResult:
    start = time.time()
    missing_env = [name for name in config.required_env if not os.getenv(name)]
    if missing_env:
        message = f"Missing required environment variables: {', '.join(missing_env)}"
        print(f"::error title={site}::{message}")
        return SiteResult(
            site=site,
            status="failed",
            message=message,
            warning=False,
            returncode=1,
            duration_seconds=round(time.time() - start, 2),
            csv_path=str(config.csv_path),
            has_date_column=False,
            scored_hotels=0,
            total_hotels=0,
        )

    cmd = [python_cmd, str(config.script), "--date", date_col]
    print(f"[run] {site}: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        message = f"Timed out after {timeout}s"
        print(f"::error title={site}::{message}")
        return SiteResult(
            site=site,
            status="failed",
            message=message,
            warning=False,
            returncode=124,
            duration_seconds=round(time.time() - start, 2),
            csv_path=str(config.csv_path),
            has_date_column=False,
            scored_hotels=0,
            total_hotels=0,
        )

    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        message = f"Script exited with code {proc.returncode}"
        print(f"::error title={site}::{message}")
        return SiteResult(
            site=site,
            status="failed",
            message=message,
            warning=False,
            returncode=proc.returncode,
            duration_seconds=round(time.time() - start, 2),
            csv_path=str(config.csv_path),
            has_date_column=False,
            scored_hotels=0,
            total_hotels=0,
        )

    exists, has_date, scored, total = validate_site_csv(config.csv_path, date_col)
    if not exists:
        message = f"Expected CSV was not created: {config.csv_path}"
        print(f"::error title={site}::{message}")
        status = "failed"
        warning = False
    elif not has_date:
        message = f"CSV missing expected date column {date_col}: {config.csv_path}"
        print(f"::error title={site}::{message}")
        status = "failed"
        warning = False
    elif scored == 0:
        message = f"No scores collected for date {date_col} in {config.csv_path}"
        print(f"::warning title={site}::{message}")
        status = "warning"
        warning = True
    else:
        message = f"Collected scores for {scored}/{total} hotels"
        status = "ok"
        warning = False

    return SiteResult(
        site=site,
        status=status,
        message=message,
        warning=warning,
        returncode=0,
        duration_seconds=round(time.time() - start, 2),
        csv_path=str(config.csv_path),
        has_date_column=has_date,
        scored_hotels=scored,
        total_hotels=total,
    )


def main() -> int:
    args = parse_args()

    sites = normalize_sites(args.sites)
    unknown_sites = [site for site in sites if site not in SITE_CONFIGS]
    if unknown_sites:
        print(f"Unknown sites ignored: {', '.join(unknown_sites)}")
    selected_sites = [site for site in sites if site in SITE_CONFIGS]

    if not selected_sites:
        print("No valid sites selected.")
        return 1

    results: list[SiteResult] = []
    for site in selected_sites:
        result = run_site(
            site=site,
            config=SITE_CONFIGS[site],
            date_col=args.date,
            python_cmd=args.python,
            timeout=args.timeout,
        )
        results.append(result)

    failed = [r for r in results if r.status == "failed"]
    warned = [r for r in results if r.warning]

    print("\nExecution summary:")
    for result in results:
        print(
            f"- {result.site}: {result.status} | {result.message} "
            f"({result.duration_seconds}s)"
        )

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "date": args.date,
            "results": [asdict(r) for r in results],
            "failed_count": len(failed),
            "warning_count": len(warned),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {summary_path}")

    if failed:
        return 1
    if warned and args.fail_on_warning:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
