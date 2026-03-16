# Reputation Analyzer

Turn guest reviews into actionable insights to improve hospitality.

This tool automatically collects hotel ratings and reviews from Booking.com, TripAdvisor, Google, Expedia, and HolidayCheck, then classifies each review by topic (cleanliness, staff, location, food, etc.) and sentiment. The result is a clear picture of what guests love and what needs attention, so hotel teams can prioritize improvements that actually move the needle on guest satisfaction and online reputation.

## How It Works

1. Each site script in `src/sites/` scrapes or calls an API for all configured hotels.
2. Each script updates one CSV in `data/`:
   - `data/booking_scores.csv`
   - `data/tripadvisor_scores.csv`
   - `data/google_scores.csv`
   - `data/expedia_scores.csv`
   - `data/holidaycheck_scores.csv`
3. Review scrapers fetch individual reviews and store them in JSON:
   - `data/tripadvisor_reviews.json`
   - `data/google_reviews.json`
   - `data/holidaycheck_reviews.json`
   - `data/expedia_reviews.json`
4. `python -m src.run` orchestrates all site scripts, validates outputs, and emits warnings/errors.
5. `.github/workflows/biweekly.yml` runs on the 1st and 15th of each month, executes tests, runs scrapers, and commits updated data files.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium  # required for Expedia reviews
```

Set required API keys:

- **Google Maps API** — [Places API documentation](https://developers.google.com/maps/documentation/places/web-service/overview). Enable the Places API in the [Google Cloud Console](https://console.cloud.google.com/apis/library/places-backend.googleapis.com) and create an API key under **Credentials**.
- **TripAdvisor API** — [TripAdvisor Content API](https://tripadvisor-content-api.readme.io/reference/overview). Sign up and request an API key from the [TripAdvisor developer portal](https://www.tripadvisor.com/developers).

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

Run review scrapers individually:

```bash
python src/sites/tripadvisor_reviews.py --skip-classification
python src/sites/google_reviews.py --skip-classification
python src/sites/holidaycheck_reviews.py --skip-classification
python src/sites/expedia_reviews.py --skip-classification
```

Run tests:

```bash
pytest -q
```

## Review Classification

Reviews are classified into topics (e.g. cleanliness, staff, location) with positive/negative sentiment using Ollama with the `qwen2.5:7b` model.

```bash
# Classify reviews (requires Ollama running locally)
python src/sites/expedia_reviews.py

# Reclassify previously unclassified reviews
python src/sites/expedia_reviews.py --reclassify
```

## GitHub Actions Automation

Workflow: `.github/workflows/biweekly.yml`

- Schedule: 1st and 15th of each month at 06:00 UTC
- Steps:
  1. Run unit tests
  2. Run `python -m src.run --summary-json data/run_summary.json`
  3. Scrape reviews from TripAdvisor, Google, HolidayCheck, and Expedia
  4. Upload summary and data artifacts
  5. Commit changed `data/*.csv` and `data/*_reviews.json` files
  6. Send alert email if any scrapers fail

Repository secrets required:

- `GOOGLE_MAPS_API_KEY`
- `TRIPADVISOR_API_KEY`
- `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD` (for alert emails)
- `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM`

### Self-Hosted Runner

All scraping runs on a `self-hosted` runner because hosted GitHub runners are commonly blocked by hotel websites (403/429).

1. Go to GitHub repo: `Settings -> Actions -> Runners -> New self-hosted runner`.
2. Follow the install/start commands on your computer.
3. Keep the runner online during scheduled workflow time.

## Project Structure

```text
config/hotels.yaml          # selected websites + hotel list
src/sites/*.py              # one script per source website (scores + reviews)
src/classification.py       # Ollama-based review topic classification
src/run.py                  # orchestrator + validation + CI annotations
data/*.csv                  # historical score tables
data/*_reviews.json         # scraped reviews with topic classification
tests/                      # unit tests
notebooks/                  # debug & test notebooks for each review scraper
dashboard/app.py            # Streamlit dashboard
```

## Dashboard

Use the Streamlit app in `dashboard/app.py` to view scores, review trends, topic sentiment, and insights.

Local run:

```bash
streamlit run dashboard/app.py
```

Recommended public deployment:

1. Push this repo to GitHub.
2. Deploy `dashboard/app.py` on Streamlit Community Cloud.
3. The biweekly workflow commits updated data files, and the dashboard will always show fresh data from `main`.
