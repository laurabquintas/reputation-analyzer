#!/usr/bin/env python3
"""Export all review JSONs to a single prettified PDF for manual review.

Usage:
    python scripts/export_reviews_pdf.py                   # default output
    python scripts/export_reviews_pdf.py -o my_export.pdf  # custom output path
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Source files in display order — map label -> filename
SOURCES = {
    "TripAdvisor": "tripadvisor_reviews.json",
    "Booking.com": "booking_reviews.json",
    "HolidayCheck": "holidaycheck_reviews.json",
}

TOPIC_LABEL = {
    "employees": "[EMP]",
    "commodities": "[COM]",
    "comfort": "[CMF]",
    "cleaning": "[CLN]",
    "quality_price": "[Q/P]",
    "meals": "[MEA]",
    "return": "[RET]",
}

SENTIMENT_SYMBOL = {"positive": "(+)", "negative": "(-)"}


class ReviewPDF(FPDF):
    """Custom PDF with header/footer for the review export."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_source = ""

        fonts_dir = Path(__file__).parent / "fonts"
        self.add_font("DejaVu", "", str(fonts_dir / "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", str(fonts_dir / "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVu", "I", str(fonts_dir / "DejaVuSans-Oblique.ttf"))

    def header(self):
        self.set_font("DejaVu", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Hotel Review Classification Export  --  {self.current_source}", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _wrap(text: str, width: int = 95) -> str:
    """Wrap long text for PDF cells."""
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def _safe(text: str) -> str:
    """Strip characters that can cause encoding issues (including emojis)."""
    if not text:
        return ""
    import re
    text = text.replace("\r", "").replace("\x00", "")
    # Remove emojis and other characters outside Basic Multilingual Plane
    text = re.sub(r"[\U00010000-\U0010FFFF]", "", text)
    return text


def _cell(pdf: FPDF, w, h, txt, **kwargs):
    """Helper: pdf.cell with new_x/new_y instead of deprecated ln."""
    pdf.cell(w, h, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT, **kwargs)


def add_source_section(pdf: ReviewPDF, source_label: str, reviews: list[dict]):
    """Add a full source section with all its reviews to the PDF."""
    pdf.current_source = source_label

    # Source title page
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 22)
    pdf.set_text_color(30, 80, 160)
    pdf.ln(30)
    pdf.cell(0, 15, source_label, align="C")
    pdf.ln(12)
    pdf.set_font("DejaVu", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, f"{len(reviews)} reviews", align="C")
    pdf.ln(8)

    classified = sum(1 for r in reviews if r.get("classified"))
    unclassified = len(reviews) - classified
    pdf.set_font("DejaVu", "", 11)
    pdf.cell(0, 8, f"Classified: {classified}  |  Unclassified: {unclassified}", align="C")

    # Individual reviews
    for i, review in enumerate(reviews, 1):
        pdf.add_page()

        # Review header
        author = review.get("author_name", "Anonymous")
        rating = review.get("rating", "N/A")
        date = review.get("published_date", review.get("stay_date", "N/A"))
        country = review.get("country", "")
        trip = review.get("trip_type", "")

        pdf.set_font("DejaVu", "B", 13)
        pdf.set_text_color(30, 30, 30)
        _cell(pdf, 0, 8, f"Review {i}/{len(reviews)}  --  {_safe(author)}")

        pdf.set_font("DejaVu", "", 9)
        pdf.set_text_color(100, 100, 100)
        meta_parts = [f"Rating: {rating}"]
        if date:
            meta_parts.append(f"Date: {date}")
        if country:
            meta_parts.append(f"Country: {country}")
        if trip:
            meta_parts.append(f"Trip: {trip}")
        _cell(pdf, 0, 5, "  |  ".join(meta_parts))
        pdf.ln(3)

        # Title
        title = review.get("title", "")
        if title:
            pdf.set_font("DejaVu", "B", 11)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, _safe(title))
            pdf.ln(2)

        # Review text
        text = review.get("text", "")
        positive_text = review.get("positive_text", "")
        negative_text = review.get("negative_text", "")

        if positive_text or negative_text:
            # Booking-style: separate positive/negative
            if positive_text:
                pdf.set_font("DejaVu", "B", 9)
                pdf.set_text_color(30, 130, 50)
                _cell(pdf, 0, 6, "LIKED:")
                pdf.set_font("DejaVu", "", 9)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(0, 5, _safe(_wrap(positive_text)))
                pdf.ln(2)
            if negative_text:
                pdf.set_font("DejaVu", "B", 9)
                pdf.set_text_color(180, 40, 40)
                _cell(pdf, 0, 6, "DISLIKED:")
                pdf.set_font("DejaVu", "", 9)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(0, 5, _safe(_wrap(negative_text)))
                pdf.ln(2)
        elif text:
            pdf.set_font("DejaVu", "", 9)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 5, _safe(_wrap(text)))
            pdf.ln(3)

        # Classification topics
        topics = review.get("topics", [])
        is_classified = review.get("classified", False)

        pdf.set_font("DejaVu", "B", 11)
        pdf.set_text_color(30, 80, 160)
        _cell(pdf, 0, 8, "Classification:")
        pdf.ln(1)

        if not topics:
            pdf.set_font("DejaVu", "I", 10)
            pdf.set_text_color(180, 40, 40)
            label = "UNCLASSIFIED" if not is_classified else "No topics found"
            _cell(pdf, 0, 6, label)
        else:
            for t in topics:
                topic_name = t.get("topic", "?")
                sentiment = t.get("sentiment", "?")
                detail = t.get("detail", "")

                tag = TOPIC_LABEL.get(topic_name, "[???]")
                sent_sym = SENTIMENT_SYMBOL.get(sentiment, "(?)")

                # Topic + sentiment
                pdf.set_font("DejaVu", "B", 10)
                if sentiment == "positive":
                    pdf.set_text_color(30, 130, 50)
                else:
                    pdf.set_text_color(180, 40, 40)

                prefix = f"  {tag} {topic_name}  {sent_sym}"
                if detail:
                    detail_clean = _safe(detail)
                    # Truncate very long details to keep layout clean
                    if len(detail_clean) > 40:
                        detail_clean = detail_clean[:37] + "..."
                    prefix += f'  --  "{detail_clean}"'

                _cell(pdf, 0, 6, prefix)

        # Separator
        pdf.ln(4)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())


def main():
    parser = argparse.ArgumentParser(description="Export review JSONs to a prettified PDF.")
    parser.add_argument(
        "-o", "--output",
        default=str(DATA_DIR / "reviews_export.pdf"),
        help="Output PDF path (default: data/reviews_export.pdf)",
    )
    args = parser.parse_args()

    pdf = ReviewPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    total_reviews = 0
    for source_label, filename in SOURCES.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"  [SKIP] {filename} not found, skipping {source_label}")
            continue

        with open(filepath) as f:
            data = json.load(f)
        reviews = data.get("reviews", [])
        if not reviews:
            print(f"  [SKIP] {source_label}: 0 reviews, skipping")
            continue

        print(f"  [OK] {source_label}: {len(reviews)} reviews")
        total_reviews += len(reviews)
        add_source_section(pdf, source_label, reviews)

    pdf.output(args.output)
    print(f"\nExported {total_reviews} reviews to {args.output}")


if __name__ == "__main__":
    main()
