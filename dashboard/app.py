from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
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

SCALE_MAX = {
    "Booking": 10.0,
    "Tripadvisor": 5.0,
    "Google": 5.0,
    "Expedia": 10.0,
    "HolidayCheck": 6.0,
}


def load_source_df(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";")
    if "Hotel" not in df.columns:
        return None
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
    if hotel not in df.index:
        df.loc[hotel] = pd.NA

    if date_col not in df.columns:
        df[date_col] = pd.NA

    df.loc[hotel, date_col] = score
    df = update_average(df)
    df.to_csv(csv_path, sep=";", index_label="Hotel")


def missing_for_date(df: pd.DataFrame, date_col: str) -> list[str]:
    if date_col not in df.columns:
        return []
    series = pd.to_numeric(df[date_col], errors="coerce")
    return df.loc[series.isna(), "Hotel"].astype(str).tolist()


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
        st.markdown("**Average Score Trend by Source**")
        source_chart_df = (
            trend_by_source.pivot(index="Date", columns="Source", values="Score")
            .sort_index()
        )
        st.line_chart(source_chart_df, use_container_width=True)
    with col2:
        st.metric("Hotels", int(filtered["Hotel"].nunique()))
        st.metric("Sources", int(filtered["Source"].nunique()))
        st.metric("Date Range", f"{filtered['Date'].min().date()} to {filtered['Date'].max().date()}")

    st.subheader("Hotel Trends")
    st.caption("Each line represents a Hotel + Source series.")
    hotel_series = (
        filtered.dropna(subset=["Score"])
        .assign(Series=lambda d: d["Hotel"] + " | " + d["Source"])
        .pivot(index="Date", columns="Series", values="Score")
        .sort_index()
    )
    st.line_chart(hotel_series, use_container_width=True)

    st.subheader("Latest Snapshot")
    latest_df = latest_snapshot(filtered)
    latest_df["Date"] = latest_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        latest_df[["Hotel", "Source", "Date", "Score"]].sort_values(["Hotel", "Source"]),
        use_container_width=True,
    )

    st.subheader("Manual Missing Values")
    st.caption("Use this to fill scores when scraping failed. Changes are written directly to CSV files in data/.")

    editable_sources = [s for s in SOURCES if s in source_dfs]
    source = st.selectbox("Source", editable_sources, index=0)
    src_df = source_dfs[source]

    date_options = source_date_columns(src_df)
    if not date_options:
        st.warning(f"No date columns found for {source}.")
        return

    selected_date = st.selectbox("Date", list(reversed(date_options)), index=0)
    missing_hotels = missing_for_date(src_df, selected_date)
    hotel_options = missing_hotels if missing_hotels else sorted(src_df["Hotel"].astype(str).tolist())
    hotel = st.selectbox("Hotel", hotel_options, index=0)

    if missing_hotels:
        st.info(f"Missing for {source} on {selected_date}: {len(missing_hotels)} hotel(s)")
    else:
        st.info(f"No missing values for {source} on {selected_date}. You can still overwrite an existing value.")

    max_scale = SCALE_MAX.get(source, 10.0)
    score = st.number_input(
        "Score",
        min_value=0.0,
        max_value=max_scale,
        value=0.0,
        step=0.1,
        format="%.1f",
    )

    if st.button("Save score"):
        try:
            set_manual_score(source=source, hotel=hotel, date_col=selected_date, score=float(score))
            st.success(f"Saved {score:.1f} for {hotel} in {source} ({selected_date}).")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to save score: {exc}")


if __name__ == "__main__":
    main()
