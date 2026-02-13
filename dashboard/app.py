from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SOURCES = {
    "Booking": DATA_DIR / "booking_scores.csv",
    "Tripadvisor": DATA_DIR / "tripadvisor_scores.csv",
    "Google": DATA_DIR / "google_scores.csv",
    "Expedia": DATA_DIR / "expedia_scores.csv",
    "HolidayCheck": DATA_DIR / "holidaycheck_scores.csv",
}


def load_source_df(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";")
    if "Hotel" not in df.columns:
        return None
    return df


DATE_COL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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


def latest_snapshot(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df
    latest_date = history_df["Date"].max()
    return history_df[history_df["Date"] == latest_date].copy()


def main() -> None:
    st.set_page_config(page_title="Hotel Reputation Dashboard", layout="wide")
    st.title("Hotel Reputation Dashboard")
    st.caption("Weekly reputation scores over time, pulled from source websites.")

    all_history = []
    for source, path in SOURCES.items():
        df = load_source_df(path)
        if df is None:
            st.warning(f"{source}: no valid data file found at {path}")
            continue
        all_history.append(scores_over_time(df, source))

    if not all_history:
        st.error("No data available.")
        return

    history_df = pd.concat(all_history, ignore_index=True)
    history_df = history_df.sort_values("Date")

    available_sources = sorted(history_df["Source"].dropna().unique().tolist())
    selected_sources = st.multiselect("Sources", available_sources, default=available_sources)

    filtered = history_df[history_df["Source"].isin(selected_sources)].copy()
    available_hotels = sorted(filtered["Hotel"].dropna().unique().tolist())
    selected_hotels = st.multiselect("Hotels", available_hotels, default=available_hotels[:4] if len(available_hotels) > 4 else available_hotels)
    filtered = filtered[filtered["Hotel"].isin(selected_hotels)]

    if filtered.empty:
        st.warning("No data for the selected filters.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        trend_by_source = (
            filtered.dropna(subset=["Score"])
            .groupby(["Date", "Source"], as_index=False)["Score"]
            .mean()
        )
        fig = px.line(
            trend_by_source,
            x="Date",
            y="Score",
            color="Source",
            markers=True,
            title="Average Score Trend by Source",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Hotels", int(filtered["Hotel"].nunique()))
        st.metric("Sources", int(filtered["Source"].nunique()))
        st.metric("Date Range", f"{filtered['Date'].min().date()} to {filtered['Date'].max().date()}")

    st.subheader("Hotel Trends")
    hotel_fig = px.line(
        filtered.dropna(subset=["Score"]),
        x="Date",
        y="Score",
        color="Hotel",
        line_dash="Source",
        markers=True,
        title="Score Trend by Hotel (line style = source)",
    )
    st.plotly_chart(hotel_fig, use_container_width=True)

    st.subheader("Latest Snapshot")
    latest_df = latest_snapshot(filtered)
    latest_df["Date"] = latest_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        latest_df[["Hotel", "Source", "Date", "Score"]].sort_values(["Hotel", "Source"]),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
