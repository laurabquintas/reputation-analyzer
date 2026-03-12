from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import sys
import yaml
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification import classify_review, is_ollama_available


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "hotels.yaml"

SOURCES = {
    "Booking": DATA_DIR / "booking_scores.csv",
    "Tripadvisor": DATA_DIR / "tripadvisor_scores.csv",
    "Google": DATA_DIR / "google_scores.csv",
    "Expedia": DATA_DIR / "expedia_scores.csv",
    "HolidayCheck": DATA_DIR / "holidaycheck_scores.csv",
}

SCALE_MAX = {
    "Booking": 10.0,
    "Tripadvisor": 5.0,
    "Google": 5.0,
    "Expedia": 10.0,
    "HolidayCheck": 6.0,
}

# Map from dashboard source name to YAML field name
_LINK_FIELDS = {
    "Booking": "booking_url",
    "Expedia": "expedia_url",
    "HolidayCheck": "holidaycheck_url",
    "Google": "google_maps_url",
}


def _load_hotel_links() -> dict[str, dict[str, str]]:
    """Build HOTEL_LINKS from config/hotels.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    links: dict[str, dict[str, str]] = {}
    for source, field in _LINK_FIELDS.items():
        source_links = {}
        for hotel in cfg.get("hotels", []):
            url = hotel.get(field, "")
            if url:
                source_links[hotel["name"]] = url
        if source_links:
            links[source] = source_links
    return links


HOTEL_LINKS: dict[str, dict[str, str]] = _load_hotel_links()
ANANEA_HOTEL = "Ananea Castelo Suites Hotel"
REVIEWS_JSON_PATH = DATA_DIR / "tripadvisor_reviews.json"
GOOGLE_REVIEWS_JSON_PATH = DATA_DIR / "google_reviews.json"


def load_source_df(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";")
    if "Hotel" not in df.columns:
        return None
    df["Hotel"] = df["Hotel"].astype(str).str.strip()
    df = df.groupby("Hotel", as_index=False).first()
    return df


DATE_COL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def source_date_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if DATE_COL_RE.fullmatch(str(c))])


def update_average(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = source_date_columns(df)
    if date_cols:
        df["Average Score"] = pd.to_numeric(df[date_cols].stack(), errors="coerce").groupby(level=0).mean().round(2)
    return df


def set_manual_score(source: str, hotel: str, date_col: str, score: float) -> None:
    csv_path = SOURCES[source]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV for {source}: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", index_col="Hotel")
    df.index = df.index.astype(str).str.strip()
    df = df.groupby(level=0).first()
    hotel = hotel.strip()
    if hotel not in df.index:
        df.loc[hotel] = pd.NA

    if date_col not in df.columns:
        df[date_col] = pd.NA

    df.loc[hotel, date_col] = score
    df = update_average(df)
    df.to_csv(csv_path, sep=";", index_label="Hotel")


def scores_over_time(df: pd.DataFrame, source: str) -> pd.DataFrame:
    date_cols = [c for c in df.columns if DATE_COL_RE.fullmatch(str(c))]
    if not date_cols:
        return pd.DataFrame(columns=["Hotel", "Source", "Date", "Score"])

    long_df = df[["Hotel", *date_cols]].melt(
        id_vars=["Hotel"],
        value_vars=date_cols,
        var_name="Date",
        value_name="Score",
    )
    long_df["Source"] = source
    long_df["Score"] = pd.to_numeric(long_df["Score"], errors="coerce")
    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    return long_df.dropna(subset=["Date"])


def latest_scorecard_table(history_df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()

    competitors = sorted([h for h in history_df["Hotel"].dropna().unique().tolist() if h != ANANEA_HOTEL])
    rows = []
    for source in sources:
        src = history_df[(history_df["Source"] == source) & history_df["Score"].notna()]
        if src.empty:
            continue
        unique_dates = sorted(src["Date"].unique())
        latest_date = unique_dates[-1]

        latest = src[src["Date"] == latest_date]
        ananea = latest[latest["Hotel"] == ANANEA_HOTEL]
        if ananea.empty:
            continue

        ananea_score = round(float(ananea["Score"].iloc[0]), 2)

        # Compute Ananea delta vs the last value from a previous month.
        # Walk backwards through dates until we find one whose month differs
        # from latest_date's month.
        ananea_delta = None
        latest_month = (latest_date.year, latest_date.month)
        for d in reversed(unique_dates[:-1]):
            if (d.year, d.month) == latest_month:
                continue
            prev = src[(src["Date"] == d) & (src["Hotel"] == ANANEA_HOTEL)]
            if not prev.empty:
                ananea_delta = round(ananea_score - float(prev["Score"].iloc[0]), 2)
                break

        row: dict[str, object] = {
            "Source": source,
            "Date": latest_date.date().isoformat(),
            ANANEA_HOTEL: ananea_score,
            "Ananea \u0394": ananea_delta,
        }
        for competitor in competitors:
            value = latest.loc[latest["Hotel"] == competitor, "Score"]
            row[competitor] = round(float(value.iloc[0]), 2) if not value.empty else None
        rows.append(row)

    return pd.DataFrame(rows)


def _format_delta(val: object) -> str:
    """Format a delta value with ▲/▼ arrows."""
    if pd.isna(val):
        return "-"
    v = float(val)
    if v > 0:
        return f"\u25b2 +{v:.2f}"
    if v < 0:
        return f"\u25bc {v:.2f}"
    return "= 0.00"


def style_scorecard(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    delta_col = "Ananea \u0394"
    meta_cols = {"Source", "Date", ANANEA_HOTEL, delta_col}
    competitor_cols = [c for c in df.columns if c not in meta_cols]

    def style_row(row: pd.Series) -> list[str]:
        styles = []
        ananea = pd.to_numeric(pd.Series([row.get(ANANEA_HOTEL)]), errors="coerce").iloc[0]
        for col in df.columns:
            if col in {"Source", "Date"}:
                styles.append("")
            elif col == ANANEA_HOTEL:
                styles.append("font-weight: 700;")
            elif col == delta_col:
                delta = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                if pd.isna(delta):
                    styles.append("")
                elif delta > 0:
                    styles.append("color: #15803d; font-weight: 600;")
                elif delta < 0:
                    styles.append("color: #b91c1c; font-weight: 600;")
                else:
                    styles.append("")
            else:
                value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                if pd.isna(value) or pd.isna(ananea):
                    styles.append("")
                elif value > ananea:
                    styles.append("color: #b91c1c; font-weight: 600;")
                elif value < ananea:
                    styles.append("color: #15803d; font-weight: 600;")
                else:
                    styles.append("")
        return styles

    format_dict: dict[str, str | Callable] = {
        ANANEA_HOTEL: "{:.2f}",
        **{c: "{:.2f}" for c in competitor_cols},
    }
    if delta_col in df.columns:
        format_dict[delta_col] = _format_delta

    return (
        df.style.apply(style_row, axis=1)
        .format(format_dict, na_rep="-", subset=[ANANEA_HOTEL, delta_col, *competitor_cols] if delta_col in df.columns else [ANANEA_HOTEL, *competitor_cols])
    )


def ananea_competitive_index(history_df: pd.DataFrame, sources: list[str]) -> dict[str, float | int]:
    """
    KPI on 0-100 scale:
    - ananea_index: average normalized Ananea score across selected sources
    - peers_index: average normalized peers score across selected sources
    - edge_pp: Ananea advantage/disadvantage in percentage points
    """
    rows = latest_scorecard_table(history_df, sources)
    if rows.empty:
        return {"ananea_index": float("nan"), "peers_index": float("nan"), "edge_pp": float("nan"), "sources_used": 0}

    normalized = []
    for _, row in rows.iterrows():
        source = str(row["Source"])
        scale = SCALE_MAX.get(source, 10.0)
        ananea_score = row.get(ANANEA_HOTEL)
        competitor_values = pd.to_numeric(
            pd.Series([row[c] for c in rows.columns if c not in {"Source", "Date", ANANEA_HOTEL}]),
            errors="coerce",
        ).dropna()
        peers_avg = competitor_values.mean() if not competitor_values.empty else float("nan")
        if pd.isna(ananea_score):
            continue
        an = float(ananea_score) / float(scale) * 100.0
        peers = float(peers_avg) / float(scale) * 100.0 if pd.notna(peers_avg) else float("nan")
        normalized.append((an, peers))

    if not normalized:
        return {"ananea_index": float("nan"), "peers_index": float("nan"), "edge_pp": float("nan"), "sources_used": 0}

    an_values = [x[0] for x in normalized]
    peers_values = [x[1] for x in normalized if pd.notna(x[1])]
    ananea_index = sum(an_values) / len(an_values)
    peers_index = sum(peers_values) / len(peers_values) if peers_values else float("nan")
    edge_pp = ananea_index - peers_index if pd.notna(peers_index) else float("nan")
    return {
        "ananea_index": round(ananea_index, 2),
        "peers_index": round(peers_index, 2) if pd.notna(peers_index) else float("nan"),
        "edge_pp": round(edge_pp, 2) if pd.notna(edge_pp) else float("nan"),
        "sources_used": len(an_values),
    }


def source_year_figure(history_df: pd.DataFrame, source: str, year: int) -> go.Figure | None:
    src = history_df[(history_df["Source"] == source) & history_df["Score"].notna()].copy()
    if src.empty:
        return None

    min_date = pd.Timestamp(year, 1, 1)
    max_date = pd.Timestamp(year, 12, 31)
    src = src[(src["Date"] >= min_date) & (src["Date"] <= max_date)].sort_values("Date")
    if src.empty:
        return None

    fig = go.Figure()
    competitors = sorted([h for h in src["Hotel"].dropna().unique().tolist() if h != ANANEA_HOTEL])
    for hotel in [ANANEA_HOTEL, *competitors]:
        hotel_df = src[src["Hotel"] == hotel]
        if hotel_df.empty:
            continue
        is_ananea = hotel == ANANEA_HOTEL
        fig.add_trace(
            go.Scatter(
                x=hotel_df["Date"],
                y=hotel_df["Score"],
                mode="lines+markers",
                name=hotel,
                line={"width": 4 if is_ananea else 1.8},
                marker={"size": 9 if is_ananea else 6},
                opacity=1.0 if is_ananea else 0.6,
            )
        )

    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=360,
        legend_title_text="Hotel",
        xaxis_title="Month",
        yaxis_title=f"Score (max {SCALE_MAX.get(source, 10):.0f})",
    )
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    return fig


def missing_or_zero_rows(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        return pd.DataFrame(columns=["Hotel", "Issue", "Current Value"])

    series = pd.to_numeric(df[date_col], errors="coerce")
    missing_mask = series.isna()
    zero_mask = series.eq(0)
    flagged = df.loc[missing_mask | zero_mask, ["Hotel"]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=["Hotel", "Issue", "Current Value"])

    flagged["Issue"] = flagged.index.to_series().map(lambda idx: "Zero value" if bool(zero_mask.loc[idx]) else "Missing")
    flagged["Current Value"] = series.loc[flagged.index].values
    return flagged.sort_values(["Issue", "Hotel"]).reset_index(drop=True)


def manual_pending_summary(source_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, df in source_dfs.items():
        dates = source_date_columns(df)
        if not dates:
            continue
        latest_date = dates[-1]
        flagged = missing_or_zero_rows(df, latest_date)
        if flagged.empty:
            continue
        for _, item in flagged.iterrows():
            rows.append(
                {
                    "Source": source,
                    "Date": latest_date,
                    "Hotel": item["Hotel"],
                    "Issue": item["Issue"],
                    "Current Value": item["Current Value"],
                }
            )
    return pd.DataFrame(rows)


_TOPIC_DISPLAY = {
    "employees": "Employees",
    "commodities": "Commodities",
    "comfort": "Comfort",
    "cleaning": "Cleaning",
    "quality_price": "Quality / Price",
    "meals": "Meals",
    "return": "Would Return",
}


def _load_reviews_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("reviews", [])


def _quarter_topic_comparison(reviews: list[dict], hotel: str) -> pd.DataFrame | None:
    """Compare current quarter vs previous quarter topic sentiment percentages."""
    today = datetime.now()
    current_q = (today.month - 1) // 3 + 1
    current_year = today.year

    # Current quarter start/end
    cq_start = f"{current_year}-{(current_q - 1) * 3 + 1:02d}-01"
    if current_q == 4:
        cq_end = f"{current_year + 1}-01-01"
    else:
        cq_end = f"{current_year}-{current_q * 3 + 1:02d}-01"

    # Previous quarter start/end
    if current_q == 1:
        pq = 4
        py = current_year - 1
    else:
        pq = current_q - 1
        py = current_year
    pq_start = f"{py}-{(pq - 1) * 3 + 1:02d}-01"
    if pq == 4:
        pq_end = f"{py + 1}-01-01"
    else:
        pq_end = f"{py}-{pq * 3 + 1:02d}-01"

    def _filter_quarter(revs: list[dict], start: str, end: str) -> list[dict]:
        return [
            r for r in revs
            if r.get("hotel") == hotel
            and r.get("classified", False)
            and start <= r.get("published_date", "")[:10] < end
        ]

    cq_reviews = _filter_quarter(reviews, cq_start, cq_end)
    pq_reviews = _filter_quarter(reviews, pq_start, pq_end)

    cq_total = len(cq_reviews)
    pq_total = len(pq_reviews)

    if cq_total == 0 and pq_total == 0:
        return None

    cq_label = f"Q{current_q} {current_year}"
    pq_label = f"Q{pq} {py}"

    rows = []
    for topic_key, topic_display in _TOPIC_DISPLAY.items():
        cq_pos = sum(1 for r in cq_reviews for t in r.get("topics", []) if t["topic"] == topic_key and t["sentiment"] == "positive")
        cq_neg = sum(1 for r in cq_reviews for t in r.get("topics", []) if t["topic"] == topic_key and t["sentiment"] == "negative")
        pq_pos = sum(1 for r in pq_reviews for t in r.get("topics", []) if t["topic"] == topic_key and t["sentiment"] == "positive")
        pq_neg = sum(1 for r in pq_reviews for t in r.get("topics", []) if t["topic"] == topic_key and t["sentiment"] == "negative")

        cq_pos_pct = round(cq_pos / cq_total * 100, 1) if cq_total else 0
        cq_neg_pct = round(cq_neg / cq_total * 100, 1) if cq_total else 0
        pq_pos_pct = round(pq_pos / pq_total * 100, 1) if pq_total else 0
        pq_neg_pct = round(pq_neg / pq_total * 100, 1) if pq_total else 0

        # None means we can't compare (one quarter has 0 reviews)
        can_compare = cq_total > 0 and pq_total > 0
        pos_delta = round(cq_pos_pct - pq_pos_pct, 1) if can_compare else None
        neg_delta = round(cq_neg_pct - pq_neg_pct, 1) if can_compare else None

        rows.append({
            "Topic": topic_display,
            "Pos Δ": pos_delta,
            "Neg Δ": neg_delta,
        })

    df = pd.DataFrame(rows)
    df.attrs["cq_label"] = cq_label
    df.attrs["pq_label"] = pq_label
    df.attrs["cq_total"] = cq_total
    df.attrs["pq_total"] = pq_total
    return df


def _render_quarter_comparison(df: pd.DataFrame | None) -> None:
    """Render quarter-over-quarter scorecards side by side above the bar plot."""
    if df is None:
        return
    cq_label = df.attrs.get("cq_label", "")
    pq_label = df.attrs.get("pq_label", "")
    cq_total = df.attrs.get("cq_total", 0)
    pq_total = df.attrs.get("pq_total", 0)

    st.caption(
        f"Comparison to previous quarter: {pq_label} ({pq_total} reviews) → "
        f"{cq_label} ({cq_total} reviews)"
    )

    cols = st.columns(len(df))
    for col, (_, row) in zip(cols, df.iterrows()):
        pos_delta = row["Pos Δ"]
        neg_delta = row["Neg Δ"]

        if pos_delta is None:
            # Can't compare – one quarter has 0 reviews
            pos_text = "--"
            pos_color = "#6b7280"
            neg_text = "--"
            neg_color = "#6b7280"
        else:
            # Positive: up is good (green), down is bad (red)
            if pos_delta > 0:
                pos_color = "#15803d"
                pos_text = f"▲ +{pos_delta}pp"
            elif pos_delta < 0:
                pos_color = "#b91c1c"
                pos_text = f"▼ {pos_delta}pp"
            else:
                pos_color = "#6b7280"
                pos_text = "– 0pp"

            # Negative: down is good (green), up is bad (red)
            if neg_delta < 0:
                neg_color = "#15803d"
                neg_text = f"▼ {neg_delta}pp"
            elif neg_delta > 0:
                neg_color = "#b91c1c"
                neg_text = f"▲ +{neg_delta}pp"
            else:
                neg_color = "#6b7280"
                neg_text = "– 0pp"

        card_html = f"""
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;
                    padding:10px 8px;text-align:center;">
            <div style="font-weight:600;font-size:0.85rem;margin-bottom:6px;">
                {row["Topic"]}
            </div>
            <div style="display:flex;justify-content:center;gap:16px;">
                <div>
                    <span style="color:#b91c1c;font-size:0.75rem;">&#11044;</span>
                    <span style="font-size:0.7rem;color:#6b7280;">Neg</span>
                    <div style="color:{neg_color};font-weight:600;font-size:0.85rem;">
                        {neg_text}
                    </div>
                </div>
                <div>
                    <span style="color:#15803d;font-size:0.75rem;">&#11044;</span>
                    <span style="font-size:0.7rem;color:#6b7280;">Pos</span>
                    <div style="color:{pos_color};font-weight:600;font-size:0.85rem;">
                        {pos_text}
                    </div>
                </div>
            </div>
        </div>
        """
        with col:
            st.markdown(card_html, unsafe_allow_html=True)


def _ytd_topic_summary(reviews: list[dict], hotel: str, year: int | None = None) -> pd.DataFrame:
    target_year = str(year if year is not None else datetime.now().year)
    ytd = [
        r for r in reviews
        if r.get("hotel") == hotel
        and r.get("published_date", "")[:4] == target_year
        and r.get("classified", False)
    ]
    total = len(ytd)
    rows = []
    for topic_key, topic_display in _TOPIC_DISPLAY.items():
        pos = sum(
            1 for r in ytd for t in r.get("topics", [])
            if t["topic"] == topic_key and t["sentiment"] == "positive"
        )
        neg = sum(
            1 for r in ytd for t in r.get("topics", [])
            if t["topic"] == topic_key and t["sentiment"] == "negative"
        )
        rows.append({
            "Topic": topic_display,
            "Positive": round(pos / total * 100, 1) if total else 0,
            "Negative": round(neg / total * 100, 1) if total else 0,
        })
    return pd.DataFrame(rows), total


def _ytd_topic_insights(
    reviews: list[dict], hotel: str, year: int | None = None, top_n: int = 2,
) -> dict[tuple[str, str], list[str]]:
    """Return the top-N most frequent detail phrases per (display_topic, sentiment)."""
    target_year = str(year if year is not None else datetime.now().year)
    ytd = [
        r for r in reviews
        if r.get("hotel") == hotel
        and r.get("published_date", "")[:4] == target_year
        and r.get("classified", False)
    ]
    counters: dict[tuple[str, str], Counter] = {}
    for r in ytd:
        for t in r.get("topics", []):
            detail = t.get("detail", "").strip().lower()
            if not detail:
                continue
            topic_key = t.get("topic", "")
            sentiment = t.get("sentiment", "")
            display = _TOPIC_DISPLAY.get(topic_key)
            if display and sentiment in ("positive", "negative"):
                key = (display, sentiment)
                if key not in counters:
                    counters[key] = Counter()
                counters[key][detail] += 1
    return {
        key: [phrase for phrase, _ in counter.most_common(top_n)]
        for key, counter in counters.items()
    }


def _render_topic_insights(
    topic_df: pd.DataFrame,
    insights: dict[tuple[str, str], list[str]],
) -> None:
    """Render top insight phrases aligned with each bar-chart row.

    The bar chart uses Plotly horizontal bars with topics on the y-axis,
    which renders them bottom-to-top.  We reverse the DataFrame order so
    the first HTML row corresponds to the top-most bar.
    """
    if not insights:
        st.caption("No detail insights available yet. Reclassify reviews to generate them.")
        return

    # Chart layout: 450px total, ~70px top (title+legend), ~50px bottom (axis).
    # Plot area ≈ 330px for 7 topic rows → ~47px per row.
    CHART_HEIGHT = 450
    TOP_OFFSET = 5      # title + legend offset
    BOTTOM_OFFSET = 50  # x-axis ticks + "% of Reviews" label
    n_topics = len(topic_df)
    plot_area = CHART_HEIGHT - TOP_OFFSET - BOTTOM_OFFSET
    row_height = plot_area / n_topics if n_topics else 50

    # Build rows in reversed order to match Plotly's bottom-to-top y-axis
    _SKIP_INSIGHTS = {"Would Return"}
    rows_html = []
    for topic_display in reversed(topic_df["Topic"].tolist()):
        if topic_display in _SKIP_INSIGHTS:
            continue
        pos_details = insights.get((topic_display, "positive"), [])
        neg_details = insights.get((topic_display, "negative"), [])
        pos_str = ", ".join(pos_details) if pos_details else "\u2014"
        neg_str = ", ".join(neg_details) if neg_details else "\u2014"
        rows_html.append(
            f'<div style="margin-bottom:8px;">'
            f'<span style="color:#888;font-weight:600;">{topic_display}</span><br>'
            f'<span style="color:#15803d;">\u25cf</span> '
            f'<span style="color:#444;">{pos_str}</span><br>'
            f'<span style="color:#b91c1c;">\u25cf</span> '
            f'<span style="color:#444;">{neg_str}</span>'
            f'</div>'
        )

    html = (
        '<div style="font-size:0.73rem;font-family:sans-serif;line-height:1.3;">'
        '<div style="font-size:1rem;font-weight:700;margin-bottom:20px;">Top Insights</div>'
        + "".join(rows_html)
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _latest_top_reviews(reviews: list[dict], hotel: str, n: int = 3) -> list[dict]:
    hotel_reviews = [r for r in reviews if r.get("hotel") == hotel]
    hotel_reviews.sort(key=lambda r: r.get("published_date", ""), reverse=True)
    return hotel_reviews[:n]


def _save_reviews_json(reviews: list[dict], path: Path) -> None:
    """Write reviews list back to the JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": {
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "total_reviews": len(reviews),
        },
        "reviews": reviews,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _generate_manual_id(reviewer_name: str, published_date: str, title: str) -> str:
    """Create a deterministic ID for a manual review."""
    seed = f"{reviewer_name}_{published_date}_{title}"
    return "manual_" + hashlib.sha256(seed.encode()).hexdigest()[:12]


def _check_password() -> bool:
    """Verify the dashboard password before granting access.

    The expected password is read from the ``DASHBOARD_PASSWORD`` environment
    variable.  If the variable is not set the dashboard is left unprotected so
    that local development is not blocked, but a sidebar warning is shown.

    Returns ``True`` when the user is authenticated (or no password is
    configured) and ``False`` otherwise.
    """
    expected = os.getenv("DASHBOARD_PASSWORD", "")
    if not expected:
        st.sidebar.warning("No DASHBOARD_PASSWORD env var set – dashboard is unprotected.")
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title("Hotel Reputation Dashboard")
    st.subheader("Login required")
    password = st.text_input("Password", type="password", key="password_input")
    if st.button("Log in", type="primary"):
        if hmac.compare_digest(password, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def main() -> None:
    st.set_page_config(page_title="Hotel Reputation Dashboard", layout="wide")

    if not _check_password():
        return

    st.title("Hotel Reputation Dashboard")
    st.caption("Biweekly reputation scores over time, pulled from source websites.")

    source_dfs: dict[str, pd.DataFrame] = {}
    all_history = []
    for source, path in SOURCES.items():
        df = load_source_df(path)
        if df is None:
            st.warning(f"{source}: no valid data file found at {path}")
            continue
        source_dfs[source] = df
        all_history.append(scores_over_time(df, source))

    if not all_history:
        st.error("No data available.")
        return

    history_df = pd.concat(all_history, ignore_index=True)
    history_df = history_df.sort_values("Date")

    available_sources = sorted(history_df["Source"].dropna().unique().tolist())
    if "source_selector" not in st.session_state:
        st.session_state["source_selector"] = available_sources
    selected_sources = st.multiselect(
        "Sources",
        available_sources,
        key="source_selector",
    )

    filtered = history_df[history_df["Source"].isin(selected_sources)].copy()
    if filtered.empty:
        st.warning("No data for the selected filters.")
        return

    # ================================================================== #
    # Competition Comparison
    # ================================================================== #
    st.header("Competition Comparison")

    st.subheader("Ananea Scorecard")
    st.caption("Latest available score per source. Competitor values are red when higher than Ananea and green when lower.")
    scorecard = latest_scorecard_table(history_df, selected_sources)
    kpi = ananea_competitive_index(history_df, selected_sources)

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        if pd.notna(kpi["ananea_index"]):
            st.metric("Ananea Competitive Index", f"{kpi['ananea_index']:.2f}/100")
        else:
            st.metric("Ananea Competitive Index", "N/A")
    with kpi_col2:
        if pd.notna(kpi["peers_index"]):
            st.metric("Peers Index", f"{kpi['peers_index']:.2f}/100")
        else:
            st.metric("Peers Index", "N/A")
    with kpi_col3:
        if pd.notna(kpi["edge_pp"]):
            st.metric("Edge vs Peers", f"{kpi['edge_pp']:+.2f} pp")
        else:
            st.metric("Edge vs Peers", "N/A")
    st.caption(
        "Competitive Index formula: for each selected source, normalize score to 0-100 "
        f"(score / source max * 100), then average across sources. "
        "Peers Index uses the average competitor score per source."
    )

    if scorecard.empty:
        st.warning("No Ananea scorecard data available for the selected sources.")
    else:
        st.dataframe(style_scorecard(scorecard.sort_values("Source")), use_container_width=True)

    current_year = datetime.now().year
    previous_year = current_year - 1
    trends_option = st.radio(
        "Period",
        [f"YTD {current_year}", f"Full Year {previous_year}"],
        horizontal=True,
        key="trends_year_toggle",
    )
    trends_year = current_year if trends_option.startswith("YTD") else previous_year
    trends_label = f"YTD {current_year}" if trends_year == current_year else str(previous_year)

    st.subheader(f"Source Trends ({trends_label})")
    st.caption("One chart per selected source. Points mark collection dates.")
    for source in selected_sources:
        fig = source_year_figure(history_df, source, trends_year)
        if fig is None:
            st.warning(f"{source}: no score history available for {trends_year}.")
            continue
        st.markdown(f"**{source}**")
        st.plotly_chart(fig, use_container_width=True)

    # ================================================================== #
    # Manual Missing Values (collapsible)
    # ================================================================== #
    with st.expander("Manual Missing Values"):
        st.caption("Use this to fill scores when scraping failed. Changes are written directly to CSV files in data/.")

        pending = manual_pending_summary(source_dfs)
        if pending.empty:
            st.success("No missing or zero values pending on the latest date for each source.")
        else:
            st.warning("Missing/zero values detected (latest date per source):")
            st.dataframe(pending.sort_values(["Source", "Issue", "Hotel"]), use_container_width=True)

        editable_sources = [s for s in SOURCES if s in source_dfs]
        mv_source = st.selectbox("Source", editable_sources, index=0, key="mv_source")
        mv_src_df = source_dfs[mv_source]

        date_options = source_date_columns(mv_src_df)
        if not date_options:
            st.warning(f"No date columns found for {mv_source}.")
        else:
            selected_date = st.selectbox("Date", list(reversed(date_options)), index=0, key="mv_date")
            flagged_rows = missing_or_zero_rows(mv_src_df, selected_date)
            missing_hotels = flagged_rows["Hotel"].astype(str).tolist()
            hotel_options = missing_hotels if missing_hotels else sorted(mv_src_df["Hotel"].astype(str).tolist())
            hotel = st.selectbox("Hotel", hotel_options, index=0, key="mv_hotel")
            hotel_link = HOTEL_LINKS.get(mv_source, {}).get(hotel)

            if hotel_link:
                st.markdown(f"Hotel link: [{hotel_link}]({hotel_link})")
            else:
                st.caption("No direct hotel link configured for this source/hotel.")

            if missing_hotels:
                st.info(
                    f"Needs input for {mv_source} on {selected_date}: "
                    + ", ".join(missing_hotels)
                )
                st.dataframe(flagged_rows, use_container_width=True)
            else:
                st.info(f"No missing/zero values for {mv_source} on {selected_date}. You can still overwrite an existing value.")

            max_scale = SCALE_MAX.get(mv_source, 10.0)
            score = st.number_input(
                "Score",
                min_value=0.0,
                max_value=max_scale,
                value=0.0,
                step=0.1,
                format="%.1f",
                key="mv_score",
            )

            if st.button("Save score", key="mv_save"):
                try:
                    set_manual_score(source=mv_source, hotel=hotel, date_col=selected_date, score=float(score))
                    st.success(f"Saved {score:.1f} for {hotel} in {mv_source} ({selected_date}).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to save score: {exc}")

    # ================================================================== #
    # Internal Analysis – Reviews
    # ================================================================== #
    st.header("Internal Analysis - Reviews")

    # Load review sources based on selection
    reviews_data = _load_reviews_json(REVIEWS_JSON_PATH) if "Tripadvisor" in selected_sources else []
    google_reviews_data = _load_reviews_json(GOOGLE_REVIEWS_JSON_PATH) if "Google" in selected_sources else []

    # Combine selected review sources for overall summary
    all_reviews_data = reviews_data + google_reviews_data

    # ---- Overall Sources Topic Sentiment ---- #
    st.subheader("Overall Sources Topic Sentiment")
    st.caption("Aggregated topic sentiment across selected review sources.")

    if not all_reviews_data:
        st.info("No review data available yet.")
    else:
        review_year_option = st.radio(
            "Period",
            [f"YTD {current_year}", f"Full Year {previous_year}"],
            horizontal=True,
            key="review_year_toggle",
        )
        selected_year = current_year if review_year_option.startswith("YTD") else previous_year
        # Quarter-over-quarter comparison
        overall_qtr_df = _quarter_topic_comparison(all_reviews_data, ANANEA_HOTEL)
        _render_quarter_comparison(overall_qtr_df)

        overall_topic_df, overall_total = _ytd_topic_summary(all_reviews_data, ANANEA_HOTEL, year=selected_year)

        if overall_topic_df[["Positive", "Negative"]].sum().sum() == 0:
            st.info(f"No classified reviews found for {selected_year}.")
        else:
            overall_label = f"YTD {selected_year}" if selected_year == current_year else str(selected_year)
            overall_insights = _ytd_topic_insights(all_reviews_data, ANANEA_HOTEL, year=selected_year)
            chart_col, insights_col = st.columns([3, 2])
            with chart_col:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=overall_topic_df["Topic"],
                    x=overall_topic_df["Positive"],
                    name="Positive",
                    orientation="h",
                    marker_color="#15803d",
                    text=[f"{v}%" for v in overall_topic_df["Positive"]],
                    textposition="auto",
                ))
                fig.add_trace(go.Bar(
                    y=overall_topic_df["Topic"],
                    x=overall_topic_df["Negative"],
                    name="Negative",
                    orientation="h",
                    marker_color="#b91c1c",
                    text=[f"{v}%" for v in overall_topic_df["Negative"]],
                    textposition="auto",
                ))
                fig.update_layout(
                    barmode="group",
                    margin={"l": 20, "r": 20, "t": 30, "b": 20},
                    height=450,
                    xaxis_title="% of Reviews",
                    xaxis_range=[0, 100],
                    yaxis_title="",
                    title=f"Overall Topic Sentiment – {overall_label} ({overall_total} reviews)",
                )
                st.plotly_chart(fig, use_container_width=True)
            with insights_col:
                _render_topic_insights(overall_topic_df, overall_insights)

    # ---- TripAdvisor ---- #
    if "Tripadvisor" in selected_sources:
        st.subheader("TripAdvisor")

        if not reviews_data:
            st.info(
                "No review data available yet. Run the reviews scraper to populate "
                "data/tripadvisor_reviews.json."
            )
        else:
            # Quarter-over-quarter comparison
            ta_qtr_df = _quarter_topic_comparison(reviews_data, ANANEA_HOTEL)
            _render_quarter_comparison(ta_qtr_df)

            ta_topic_df, ta_total = _ytd_topic_summary(reviews_data, ANANEA_HOTEL, year=selected_year)

            if ta_topic_df[["Positive", "Negative"]].sum().sum() == 0:
                st.info(f"No classified TripAdvisor reviews found for {selected_year}.")
            else:
                ta_label = f"YTD {selected_year}" if selected_year == current_year else str(selected_year)
                ta_insights = _ytd_topic_insights(reviews_data, ANANEA_HOTEL, year=selected_year)
                chart_col, insights_col = st.columns([3, 2])
                with chart_col:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=ta_topic_df["Topic"],
                        x=ta_topic_df["Positive"],
                        name="Positive",
                        orientation="h",
                        marker_color="#15803d",
                        text=[f"{v}%" for v in ta_topic_df["Positive"]],
                        textposition="auto",
                    ))
                    fig.add_trace(go.Bar(
                        y=ta_topic_df["Topic"],
                        x=ta_topic_df["Negative"],
                        name="Negative",
                        orientation="h",
                        marker_color="#b91c1c",
                        text=[f"{v}%" for v in ta_topic_df["Negative"]],
                        textposition="auto",
                    ))
                    fig.update_layout(
                        barmode="group",
                        margin={"l": 20, "r": 20, "t": 30, "b": 20},
                        height=450,
                        xaxis_title="% of Reviews",
                        xaxis_range=[0, 100],
                        yaxis_title="",
                        title=f"TripAdvisor Topic Sentiment – {ta_label} ({ta_total} reviews)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with insights_col:
                    _render_topic_insights(ta_topic_df, ta_insights)

            st.markdown("**Latest Reviews**")
            top_reviews = _latest_top_reviews(reviews_data, ANANEA_HOTEL, n=3)

            if not top_reviews:
                st.info("No reviews found.")
            else:
                cols = st.columns(len(top_reviews))
                for col, review in zip(cols, top_reviews):
                    with col:
                        rating = int(review.get("rating") or 0)
                        stars = "\u2605" * rating + "\u2606" * (5 - rating)
                        st.markdown(f"**{stars}** {rating}/5")
                        st.markdown(f"**{review.get('title', 'No title')}**")
                        text = review.get("text", "")
                        display_text = text[:200] + "..." if len(text) > 200 else text
                        st.caption(display_text)

                        pub_date = review.get("published_date", "")
                        trip_type = review.get("trip_type", "")
                        meta_parts = []
                        if pub_date:
                            meta_parts.append(pub_date)
                        if trip_type:
                            meta_parts.append(trip_type.replace("_", " ").title())
                        if meta_parts:
                            st.caption(" | ".join(meta_parts))

                        topics = review.get("topics", [])
                        if topics:
                            pills = " ".join(
                                f"{'🟢' if t['sentiment'] == 'positive' else '🔴'} "
                                f"{t['topic'].replace('_', ' ').title()}"
                                for t in topics
                            )
                            st.caption(pills)

    # ---- Google ---- #
    if "Google" in selected_sources:
        st.subheader("Google")

        if not google_reviews_data:
            st.info(
                "No Google review data available yet. Run the reviews scraper to populate "
                "data/google_reviews.json."
            )
        else:
            # Quarter-over-quarter comparison
            google_qtr_df = _quarter_topic_comparison(google_reviews_data, ANANEA_HOTEL)
            _render_quarter_comparison(google_qtr_df)

            google_topic_df, google_total = _ytd_topic_summary(google_reviews_data, ANANEA_HOTEL, year=selected_year)

            if google_topic_df[["Positive", "Negative"]].sum().sum() == 0:
                st.info(f"No classified Google reviews found for {selected_year}.")
            else:
                google_label = f"YTD {selected_year}" if selected_year == current_year else str(selected_year)
                google_insights = _ytd_topic_insights(google_reviews_data, ANANEA_HOTEL, year=selected_year)
                chart_col, insights_col = st.columns([3, 2])
                with chart_col:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=google_topic_df["Topic"],
                        x=google_topic_df["Positive"],
                        name="Positive",
                        orientation="h",
                        marker_color="#15803d",
                        text=[f"{v}%" for v in google_topic_df["Positive"]],
                        textposition="auto",
                    ))
                    fig.add_trace(go.Bar(
                        y=google_topic_df["Topic"],
                        x=google_topic_df["Negative"],
                        name="Negative",
                        orientation="h",
                        marker_color="#b91c1c",
                        text=[f"{v}%" for v in google_topic_df["Negative"]],
                        textposition="auto",
                    ))
                    fig.update_layout(
                        barmode="group",
                        margin={"l": 20, "r": 20, "t": 30, "b": 20},
                        height=450,
                        xaxis_title="% of Reviews",
                        xaxis_range=[0, 100],
                        yaxis_title="",
                        title=f"Google Topic Sentiment – {google_label} ({google_total} reviews)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with insights_col:
                    _render_topic_insights(google_topic_df, google_insights)

            st.markdown("**Latest Reviews**")
            google_top_reviews = _latest_top_reviews(google_reviews_data, ANANEA_HOTEL, n=3)

            if not google_top_reviews:
                st.info("No Google reviews found.")
            else:
                cols = st.columns(len(google_top_reviews))
                for col, review in zip(cols, google_top_reviews):
                    with col:
                        rating = int(review.get("rating") or 0)
                        stars = "\u2605" * rating + "\u2606" * (5 - rating)
                        st.markdown(f"**{stars}** {rating}/5")
                        author = review.get("author_name", "Anonymous")
                        st.markdown(f"**{author}**")
                        text = review.get("text", "")
                        display_text = text[:200] + "..." if len(text) > 200 else text
                        st.caption(display_text)

                        pub_date = review.get("published_date", "")
                        if pub_date:
                            st.caption(pub_date)

                        topics = review.get("topics", [])
                        if topics:
                            pills = " ".join(
                                f"{'🟢' if t['sentiment'] == 'positive' else '🔴'} "
                                f"{t['topic'].replace('_', ' ').title()}"
                                for t in topics
                            )
                            st.caption(pills)

    # ---- Manual Review Input ---- #
    with st.expander("Add Review Manually"):
        st.caption(
            "Review APIs return a limited number of reviews. "
            "Use this form to add reviews you found on the website that the API missed."
        )

        with st.form("manual_review_form", clear_on_submit=True):
            mr_source = st.selectbox(
                "Source",
                ["TripAdvisor"],
                index=0,
                key="mr_source",
            )
            mr_cols = st.columns([2, 1])
            with mr_cols[0]:
                mr_reviewer = st.text_input("Reviewer name", placeholder="e.g. John D.")
                mr_title = st.text_input("Review title", placeholder="e.g. Amazing stay!")
            with mr_cols[1]:
                mr_rating = st.number_input("Rating", min_value=1, max_value=5, value=5, step=1)
                mr_date = st.date_input("Review date")
                mr_trip = st.selectbox(
                    "Trip type",
                    ["Couples", "Family", "Solo", "Business", "Friends"],
                    index=0,
                )
            mr_text = st.text_area("Review text", height=150, placeholder="Paste the full review text here...")
            mr_submitted = st.form_submit_button("Add review", type="primary")

        if mr_submitted:
            if not mr_reviewer or not mr_text:
                st.error("Reviewer name and review text are required.")
            else:
                pub_date_str = mr_date.strftime("%Y-%m-%d")
                review_id = _generate_manual_id(mr_reviewer, pub_date_str, mr_title)
                current_reviews = _load_reviews_json(REVIEWS_JSON_PATH)
                existing_ids = {r["id"] for r in current_reviews}

                if review_id in existing_ids:
                    st.warning("This review already exists (same name + date + title).")
                else:
                    # Try to classify automatically via Ollama
                    topics: list[dict] = []
                    classified = False
                    ollama_msg = ""
                    if is_ollama_available():
                        try:
                            topics = classify_review(mr_text)
                            classified = True
                            ollama_msg = f" Classified with {len(topics)} topics."
                        except Exception:
                            ollama_msg = " Ollama classification failed; saved without topics."
                    else:
                        ollama_msg = " Ollama not available; saved without classification."

                    new_review = {
                        "id": review_id,
                        "hotel": ANANEA_HOTEL,
                        "location_id": "",
                        "rating": mr_rating,
                        "title": mr_title,
                        "text": mr_text,
                        "published_date": f"{pub_date_str}T00:00:00Z",
                        "travel_date": "",
                        "trip_type": mr_trip,
                        "subratings": {},
                        "helpful_votes": 0,
                        "scraped_date": datetime.now().strftime("%Y-%m-%d"),
                        "topics": topics,
                        "classified": classified,
                        "source": "manual",
                        "review_source": mr_source,
                    }
                    current_reviews.append(new_review)
                    _save_reviews_json(current_reviews, REVIEWS_JSON_PATH)
                    st.success(f"Review added (ID: {review_id}).{ollama_msg}")
                    st.rerun()



if __name__ == "__main__":
    main()
