"""
Light HTML scraper for Booking rating.
Fill `booking_url` per hotel. Selectors can change often.
If a URL is missing or parse fails, returns None for that hotel.
"""
import time, re, requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (compatible; ReputationBot/1.0)"}

def _parse(html: str):
    soup = BeautifulSoup(html, "html.parser")
    # Try common patterns; adjust as needed
    candidates = [
        "[data-testid='review-score-component']",
        ".b5cd09854e.d10a6220b4",
        "[aria-label*='score']",
    ]
    for sel in candidates:
        el = soup.select_one(sel)
        if not el:
            continue
        m = re.search(r"\d+(?:[.,]\d+)?", el.get_text(" ", strip=True))
        if m:
            return float(m.group(0).replace(",", "."))
    # Fallback search for numbers like 8.7 near 'review' keywords
    m = re.search(r"(?:score|rating)[^\d]{0,10}(\d+(?:[.,]\d+)?)", soup.get_text(" ", strip=True), flags=re.I)
    if m:
        return float(m.group(1).replace(",", "."))
    return None

def fetch(hotels_cfg) -> dict:
    out = {}
    for h in hotels_cfg:
        url = h.get("booking_url") or ""
        if not url:
            out[h["name"]] = None
            continue
        try:
            r = requests.get(url, headers=UA, timeout=25)
            out[h["name"]] = _parse(r.text)
            time.sleep(1.0)
        except Exception:
            out[h["name"]] = None
    return out

