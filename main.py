from scraper.booking import get_booking_rating
from updater.update_excel import update_excel_rating

vila_maria_url = "https://www.booking.com/hotel/pt/vila-maria-albufeira1.en-gb.html"
rating = get_booking_rating(vila_maria_url)

if rating:
    print(f"⭐ Vila Maria rating on Booking: {rating}")
    update_excel_rating(
        excel_path="data/Reputation Analysis Vila Maria.xlsx",
        hotel="Vila Maria",
        source="Booking",
        rating=rating
    )
else:
    print("❌ Could not retrieve rating.")
