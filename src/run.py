from pathlib import Path
from datetime import datetime
import os

from util import load_config, hotels_list, websites
from sheets_writer import write_results
from sites import google_places, booking, tripadvisor, expedia, holidaycheck, zoover

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "hotels.yaml")
HOTELS_CFG = CFG["hotels"]
HOTELS = hotels_list(CFG)
TODAY = datetime.utcnow()

SITE_FUNCS = {
    "GOOGLE": google_places.fetch,
    "BOOKING": booking.fetch,
    "TRIPADVISOR": tripadvisor.fetch,
    "EXPEDIA": expedia.fetch,
    "HOLIDAYCHECK": holidaycheck.fetch,
    "ZOOVER": zoover.fetch,
}

def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    site_list = websites(CFG) or list(SITE_FUNCS.keys())
    for site in site_list:
        fn = SITE_FUNCS.get(site.upper())
        if not fn:
            continue
        try:
            ratings = fn(HOTELS_CFG)
        except Exception:
            ratings = {h: None for h in HOTELS}
        write_results(sheet_id, site, HOTELS, ratings, TODAY)

if __name__ == "__main__":
    main()

