# Reputation Analyzer (Google Sheets)

Automated, low-cost hotel reputation tracker.  
- **One Google Sheet tab per website** (`BOOKING`, `TRIPADVISOR`, `GOOGLE`, `EXPEDIA`, `HOLIDAYCHECK`, `ZOOVER`)
- **One column per run date** (e.g., `2025-09-07`, `2025-09-14`, …)
- **Weekly GitHub Action** updates the sheet.

## 1) Setup (one time)

1. **Google Cloud**
   - Create a project → Enable **Google Sheets API**
   - Create a **Service Account** → Create JSON key
2. **Google Sheets**
   - Create an empty sheet (e.g., “Reputation Tracker”)
   - Share with the service account email (`…@…iam.gserviceaccount.com`) as **Editor**
   - Copy the Sheet ID from URL
3. **GitHub Secrets** (Repo → Settings → Secrets and variables → Actions)
   - `GOOGLE_SERVICE_ACCOUNT_JSON` → entire JSON key content
   - `GOOGLE_SHEET_ID` → the long ID in the sheet URL
   - (optional) `GOOGLE_PLACES_API_KEY` for Google ratings

## 2) Configure hotels

Edit `config/hotels.yaml` with hotel names and (optionally) per-site URLs / IDs.

## 3) Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GOOGLE_SERVICE_ACCOUNT_JSON='{"type": "..."}'
export GOOGLE_SHEET_ID='YOUR_SHEET_ID'
python -m src.run
