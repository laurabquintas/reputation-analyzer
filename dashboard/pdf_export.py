"""Generate a PDF snapshot of the dashboard with current filters."""

from __future__ import annotations

import io
import re
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "fonts"

_TOPIC_DISPLAY = {
    "employees": "Employees",
    "commodities": "Commodities",
    "comfort": "Comfort",
    "cleaning": "Cleaning",
    "quality_price": "Quality / Price",
    "meals": "Meals",
    "return": "Would Return",
}

SENTIMENT_SYM = {"positive": "(+)", "negative": "(-)"}

# Source label mapping for review sections
_SOURCE_REVIEW_LABEL = {
    "Booking": "Booking.com",
    "Tripadvisor": "TripAdvisor",
    "Google": "Google",
    "HolidayCheck": "HolidayCheck",
    "Expedia": "Expedia",
}


class DashboardPDF(FPDF):
    """Custom PDF with header/footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._section_title = ""
        self.add_font("DejaVu", "", str(FONTS_DIR / "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", str(FONTS_DIR / "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVu", "I", str(FONTS_DIR / "DejaVuSans-Oblique.ttf"))

    def header(self):
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 7, f"Hotel Reputation Dashboard  --  {self._section_title}", align="C")
        self.ln(9)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  --  Generated {datetime.now():%Y-%m-%d %H:%M}", align="C")


def _cell(pdf: FPDF, w, h, txt, **kwargs):
    pdf.cell(w, h, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT, **kwargs)


def _safe(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "").replace("\x00", "")
    return re.sub(r"[\U00010000-\U0010FFFF]", "", text)


def _fig_to_png(fig: go.Figure, width: int = 900, height: int = 450) -> bytes:
    """Render a Plotly figure to PNG bytes."""
    return fig.to_image(format="png", width=width, height=height, scale=2)


# ------------------------------------------------------------------ #
# Topic sentiment bar chart (reused per source)
# ------------------------------------------------------------------ #

def _build_topic_chart(topic_df: pd.DataFrame, title: str) -> go.Figure:
    """Build a horizontal bar chart identical to the dashboard's."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=topic_df["Topic"], x=topic_df["Positive"],
        name="Positive", orientation="h", marker_color="#15803d",
        text=[f"{v}%" for v in topic_df["Positive"]], textposition="auto",
    ))
    fig.add_trace(go.Bar(
        y=topic_df["Topic"], x=topic_df["Negative"],
        name="Negative", orientation="h", marker_color="#b91c1c",
        text=[f"{v}%" for v in topic_df["Negative"]], textposition="auto",
    ))
    fig.update_layout(
        barmode="group",
        margin={"l": 120, "r": 20, "t": 40, "b": 30},
        height=450,
        xaxis_title="% of Reviews", xaxis_range=[0, 100],
        yaxis_title="",
        title=title,
        plot_bgcolor="white",
    )
    return fig


# ------------------------------------------------------------------ #
# PDF section builders
# ------------------------------------------------------------------ #

def _add_cover(pdf: DashboardPDF, selected_sources: list[str], year: int):
    pdf._section_title = "Cover"
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("DejaVu", "B", 26)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 15, "Hotel Reputation Dashboard", align="C")
    pdf.ln(14)
    pdf.set_font("DejaVu", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, f"YTD {year}", align="C")
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 11)
    pdf.cell(0, 8, f"Sources: {', '.join(selected_sources)}", align="C")
    pdf.ln(8)
    pdf.set_font("DejaVu", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, f"Generated {datetime.now():%Y-%m-%d %H:%M}", align="C")


def _add_scorecard(
    pdf: DashboardPDF,
    scorecard_df: pd.DataFrame,
    kpi: dict,
):
    pdf._section_title = "Competition Comparison"
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(30, 80, 160)
    _cell(pdf, 0, 10, "Competition Comparison  --  Scorecard")
    pdf.ln(4)

    # KPI row
    pdf.set_font("DejaVu", "B", 11)
    pdf.set_text_color(50, 50, 50)
    ananea_idx = kpi.get("ananea_index")
    peers_idx = kpi.get("peers_index")
    edge = kpi.get("edge")
    kpi_text = (
        f"Ananea Index: {ananea_idx:.2f}/100" if pd.notna(ananea_idx) else "Ananea Index: N/A"
    )
    kpi_text += f"   |   Peers Index: {peers_idx:.2f}/100" if pd.notna(peers_idx) else "   |   Peers Index: N/A"
    kpi_text += f"   |   Edge: {edge:+.2f}" if pd.notna(edge) else "   |   Edge: N/A"
    _cell(pdf, 0, 8, kpi_text)
    pdf.ln(4)

    # Scorecard table
    if scorecard_df is not None and not scorecard_df.empty:
        pdf.set_font("DejaVu", "B", 8)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(30, 80, 160)
        col_widths = [50] + [28] * min(len(scorecard_df.columns) - 1, 6)
        for j, col in enumerate(scorecard_df.columns[:7]):
            w = col_widths[j] if j < len(col_widths) else 28
            pdf.cell(w, 7, str(col), fill=True)
        pdf.ln()

        pdf.set_font("DejaVu", "", 7)
        for _, row in scorecard_df.iterrows():
            for j, col in enumerate(scorecard_df.columns[:7]):
                w = col_widths[j] if j < len(col_widths) else 28
                val = row[col]
                pdf.set_text_color(50, 50, 50)
                pdf.cell(w, 6, _safe(str(val)))
            pdf.ln()


def _add_trends(pdf: DashboardPDF, history_df: pd.DataFrame, selected_sources: list[str], year: int, source_year_figure_fn):
    pdf._section_title = f"Source Trends YTD {year}"
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(30, 80, 160)
    _cell(pdf, 0, 10, f"Source Trends  --  YTD {year}")
    pdf.ln(2)

    for source in selected_sources:
        fig = source_year_figure_fn(history_df, source, year)
        if fig is None:
            continue
        png = _fig_to_png(fig, width=900, height=300)
        # Check if we need a new page
        if pdf.get_y() > 160:
            pdf.add_page()
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(50, 50, 50)
        _cell(pdf, 0, 7, source)
        pdf.image(io.BytesIO(png), x=10, w=190)
        pdf.ln(4)


def _add_topic_section(
    pdf: DashboardPDF,
    section_title: str,
    topic_df: pd.DataFrame,
    total: int,
    year: int,
    insights: dict,
    reviews: list[dict],
    hotel: str,
):
    """Add a topic sentiment section: chart + insights + latest reviews."""
    pdf._section_title = section_title
    pdf.add_page()

    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(30, 80, 160)
    label = f"YTD {year}"
    _cell(pdf, 0, 10, f"{section_title}  --  Topic Sentiment ({label})")
    pdf.ln(2)

    # Chart as image
    if topic_df[["Positive", "Negative"]].sum().sum() > 0:
        fig = _build_topic_chart(topic_df, f"{section_title} ({total} reviews)")
        png = _fig_to_png(fig, width=900, height=450)
        pdf.image(io.BytesIO(png), x=10, w=190)
        pdf.ln(4)

        # Insights text
        _SKIP = {"Would Return", "Quality / Price"}
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(30, 80, 160)
        _cell(pdf, 0, 7, "Top Insights")
        pdf.ln(1)

        for topic_display in topic_df["Topic"].tolist():
            if topic_display in _SKIP:
                continue
            pos_details = insights.get((topic_display, "positive"), [])
            neg_details = insights.get((topic_display, "negative"), [])

            pdf.set_font("DejaVu", "B", 8)
            pdf.set_text_color(80, 80, 80)
            _cell(pdf, 0, 5, topic_display)

            pdf.set_font("DejaVu", "", 8)
            if pos_details:
                pdf.set_text_color(30, 130, 50)
                _cell(pdf, 0, 4, f"  (+) {', '.join(pos_details)}")
            if neg_details:
                pdf.set_text_color(180, 40, 40)
                _cell(pdf, 0, 4, f"  (-) {', '.join(neg_details)}")
    else:
        pdf.set_font("DejaVu", "I", 10)
        pdf.set_text_color(120, 120, 120)
        _cell(pdf, 0, 8, f"No classified reviews found for {year}.")

    # Latest reviews
    pdf.ln(4)
    top_reviews = _get_latest_reviews(reviews, hotel, n=3)
    if top_reviews:
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(30, 80, 160)
        _cell(pdf, 0, 7, "Latest Reviews")
        pdf.ln(2)

        for review in top_reviews:
            if pdf.get_y() > 240:
                pdf.add_page()
            author = review.get("author_name", "Anonymous")
            rating = review.get("rating", "N/A")
            date = review.get("published_date", "")
            text = review.get("text", review.get("positive_text", ""))

            pdf.set_font("DejaVu", "B", 9)
            pdf.set_text_color(50, 50, 50)
            _cell(pdf, 0, 5, f"{_safe(author)}  --  Rating: {rating}  --  {date}")

            title = review.get("title", "")
            if title:
                pdf.set_font("DejaVu", "B", 8)
                pdf.set_text_color(70, 70, 70)
                pdf.multi_cell(0, 4, _safe(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            if text:
                pdf.set_font("DejaVu", "", 8)
                pdf.set_text_color(80, 80, 80)
                display = _safe(text[:300] + "..." if len(text) > 300 else text)
                pdf.multi_cell(0, 4, "\n".join(textwrap.wrap(display, 110)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            topics = review.get("topics", [])
            if topics:
                pdf.set_font("DejaVu", "", 7)
                pills = "  ".join(
                    f"{'(+)' if t['sentiment'] == 'positive' else '(-)'} {t['topic']}"
                    + (f' "{t["detail"]}"' if t.get("detail") else "")
                    for t in topics
                )
                detail_clean = _safe(pills)
                if len(detail_clean) > 120:
                    detail_clean = detail_clean[:117] + "..."
                pdf.set_text_color(100, 100, 100)
                _cell(pdf, 0, 4, detail_clean)

            pdf.ln(3)
            pdf.set_draw_color(220, 220, 220)
            pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
            pdf.ln(2)


def _get_latest_reviews(reviews: list[dict], hotel: str, n: int = 3) -> list[dict]:
    hotel_reviews = [r for r in reviews if r.get("hotel") == hotel]
    hotel_reviews.sort(key=lambda r: r.get("published_date", ""), reverse=True)
    return hotel_reviews[:n]


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def generate_dashboard_pdf(
    *,
    history_df: pd.DataFrame,
    selected_sources: list[str],
    source_dfs: dict[str, pd.DataFrame],
    year: int,
    hotel: str,
    scorecard_df: pd.DataFrame,
    kpi: dict,
    source_year_figure_fn,
    ytd_topic_summary_fn,
    ytd_topic_insights_fn,
    reviews_by_source: dict[str, list[dict]],
) -> bytes:
    """Generate a full dashboard PDF and return it as bytes."""

    pdf = DashboardPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # 1. Cover
    _add_cover(pdf, selected_sources, year)

    # 2. Scorecard
    _add_scorecard(pdf, scorecard_df, kpi)

    # 3. Trends
    _add_trends(pdf, history_df, selected_sources, year, source_year_figure_fn)

    # 4. Overall topic sentiment
    all_reviews = []
    for reviews in reviews_by_source.values():
        all_reviews.extend(reviews)

    if all_reviews:
        overall_df, overall_total = ytd_topic_summary_fn(all_reviews, hotel, year=year)
        overall_insights = ytd_topic_insights_fn(all_reviews, hotel, year=year)
        _add_topic_section(
            pdf, "Overall Sources", overall_df, overall_total,
            year, overall_insights, all_reviews, hotel,
        )

    # 5. Per-source sections
    for source in selected_sources:
        source_reviews = reviews_by_source.get(source, [])
        if not source_reviews:
            continue
        label = _SOURCE_REVIEW_LABEL.get(source, source)
        src_df, src_total = ytd_topic_summary_fn(source_reviews, hotel, year=year)
        src_insights = ytd_topic_insights_fn(source_reviews, hotel, year=year)
        _add_topic_section(
            pdf, label, src_df, src_total,
            year, src_insights, source_reviews, hotel,
        )

    return pdf.output()
