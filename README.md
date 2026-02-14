# Reputation Analyzer

Collects hotel ratings from multiple websites and stores weekly snapshots in CSV tables under `data/`.

## Current Flow

1. Each site script in `src/sites/` scrapes or calls an API for all configured hotels.
2. Each script updates one CSV in `data/`:
   - `data/booking_scores.csv`
   - `data/tripadvisor_scores.csv`
   - `data/google_scores.csv`
   - `data/expedia_scores.csv`
   - `data/holidaycheck_scores.csv`
3. `python -m src.run` orchestrates all site scripts, validates outputs, and emits warnings/errors.
4. `.github/workflows/weekly.yml` runs weekly, executes tests, runs scrapers, and commits updated CSVs.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set required API keys:

```bash
export GOOGLE_MAPS_API_KEY="your_google_key"
export TRIPADVISOR_API_KEY="your_tripadvisor_key"
```

Run all sites:

```bash
python -m src.run
```

Run specific sites:

```bash
python -m src.run --sites GOOGLE TRIPADVISOR
```

Run tests:

```bash
pytest -q
```

## GitHub Actions Automation

Workflow: `.github/workflows/weekly.yml`

- Schedule: every Monday at 06:00 UTC
- Steps:
  1. run unit tests
  2. run `python -m src.run --summary-json data/run_summary.json`
  3. upload summary artifact
  4. commit changed `data/*.csv` files

Repository secrets required:

- `GOOGLE_MAPS_API_KEY`
- `TRIPADVISOR_API_KEY`
- Optional alias: `GOOGLE_PLACES_API_KEY`

If a scraper fails, the run exits non-zero and GitHub Actions marks the job as failed.  
If a scraper succeeds but collects zero rows for the date, `src.run` emits a GitHub warning annotation.

### Expedia on Self-Hosted Runner

Expedia scraping runs on `self-hosted` (your own computer) because hosted GitHub runners are commonly blocked by Expedia (403/429).

1. Go to GitHub repo: `Settings -> Actions -> Runners -> New self-hosted runner`.
2. Follow the install/start commands on your computer.
3. Keep the runner online during scheduled workflow time.

The workflow now runs:
- non-Expedia sites on `ubuntu-latest`
- Expedia on `self-hosted`
- then merges and commits all updated `data/*.csv`.

## Project Structure

```text
config/hotels.yaml          # selected websites + hotel list
src/sites/*.py              # one script per source website
src/run.py                  # orchestrator + validation + CI annotations
data/*.csv                  # historical score tables
tests/test_run.py           # unit tests for runner logic
dashboard/app.py            # Streamlit dashboard starter
```

## Dashboard (Public Access)

Use the Streamlit app in `dashboard/app.py`.

Local run:

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

Recommended public deployment:

1. Push this repo to GitHub.
2. Deploy `dashboard/app.py` on Streamlit Community Cloud.
3. Keep the weekly workflow committing `data/*.csv`, and the dashboard will always show fresh data from `main`.
