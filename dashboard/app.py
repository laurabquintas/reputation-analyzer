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

HOTEL_LINKS: dict[str, dict[str, str]] = {
    "Booking": {
        "Ananea Castelo Suites Hotel": "https://www.booking.com/hotel/pt/castelo-suites.en-gb.html",
        "PortoBay Falésia": "https://www.booking.com/hotel/pt/porto-bay-falesia.en-gb.html",
        "Regency Salgados Hotel & Spa": "https://www.booking.com/hotel/pt/regency-salgados-amp-spa.en-gb.html",
        "NAU São Rafael Atlântico": "https://www.booking.com/hotel/pt/sao-rafael-suites-all-inclusive.en-gb.html",
        "NAU Salgados Dunas Suites": "https://www.booking.com/hotel/pt/westin-salgados-beach-resort-algarve.en-gb.html",
        "Vidamar Resort Hotel Algarve": "https://www.booking.com/hotel/pt/vidamar-algarve-hotel.en-gb.html",
    },
    "Expedia": {
        "Ananea Castelo Suites Hotel": "https://euro.expedia.net/Albufeira-Hotels-Castelo-Suites-Hotel.h111521689.Hotel-Information?pwaDialog=product-reviews",
        "PortoBay Falésia": "https://euro.expedia.net/Albufeira-Hotels-PortoBay-Falesia.h1787641.Hotel-Information?pwaDialog=product-reviews",
        "Regency Salgados Hotel & Spa": "https://euro.expedia.net/Albufeira-Hotels-Regency-Salgados-Hotel-Spa.h67650702.Hotel-Information?pwaDialog=product-reviews",
        "NAU São Rafael Atlântico": "https://euro.expedia.net/Albufeira-Hotels-Sao-Rafael-Suite-Hotel.h1210300.Hotel-Information?pwaDialogNested=PropertyDetailsReviewsBreakdownDialog",
        "Vidamar Resort Hotel Algarve": "https://euro.expedia.net/Albufeira-Hotels-VidaMar-Resort-Hotel-Algarve.h5670748.Hotel-Information?pwaDialog=product-reviews",
    },
    "HolidayCheck": {
        "Ananea Castelo Suites Hotel": "https://www.holidaycheck.de/hi/ananea-castelo-suites-algarve/069563af-47db-44a3-bdb1-3441ae3a2ac4",
        "PortoBay Falésia": "https://www.holidaycheck.de/hi/portobay-falesia/44a47534-85c4-3114-a6da-472d82e16e29",
        "Regency Salgados Hotel & Spa": "https://www.holidaycheck.de/hi/regency-salgados-hotel-spa/b0478236-7644-46b4-8fde-bd6cb1832cf8",
        "NAU São Rafael Atlântico": "https://www.holidaycheck.de/hi/nau-sao-rafael-suites-all-inclusive/739da55a-710e-3514-83f6-8e01149442a5",
        "NAU Salgados Dunas Suites": "https://www.holidaycheck.de/hi/nau-salgados-vila-das-lagoas-apartment/602ac74a-9c28-3d74-8dd9-37c47c53cd4a",
        "Vidamar Resort Hotel Algarve": "https://www.holidaycheck.de/hi/vidamar-hotel-resort-algarve/e641bc1e-59d5-37a0-832e-90e6bbb51977",
    },
    "Google": {
        "Ananea Castelo Suites Hotel": "https://maps.app.goo.gl/QsTaS8vLupyrC3hQ8",
        "PortoBay Falésia": "https://maps.app.goo.gl/DxodrUv4ub7qp89eA",
        "Regency Salgados Hotel & Spa": "https://maps.app.goo.gl/UZ6dAot3VC4eWV3U7",
        "NAU São Rafael Atlântico": "https://maps.app.goo.gl/G3Nfg49qBYQkR2xr5",
        "NAU Salgados Dunas Suites": "https://maps.app.goo.gl/CxCEgfZkiXnzAEsy9",
        "Vidamar Resort Hotel Algarve": "https://maps.app.goo.gl/etAzqPDxgnjJ2DDu7",
    },
}
ANANEA_HOTEL = "Ananea Castelo Suites Hotel"


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


def ananea_scorecard_rows(history_df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    rows = []
    for source in sources:
        src = history_df[(history_df["Source"] == source) & history_df["Score"].notna()]
        if src.empty:
            continue
        latest_date = src["Date"].max()
        latest = src[src["Date"] == latest_date]
        ananea = latest[latest["Hotel"] == ANANEA_HOTEL]
        if ananea.empty:
            continue
        ananea_score = float(ananea["Score"].iloc[0])
        peers = latest[latest["Hotel"] != ANANEA_HOTEL]["Score"].dropna()
        peers_avg = float(peers.mean()) if not peers.empty else None
        delta_vs_peers = ananea_score - peers_avg if peers_avg is not None else None
        rows.append(
            {
                "Source": source,
                "Date": latest_date,
                "Ananea Score": round(ananea_score, 2),
                "Peers Avg": round(peers_avg, 2) if peers_avg is not None else None,
                "Delta vs Peers": round(delta_vs_peers, 2) if delta_vs_peers is not None else None,
                "Peers Count": int(peers.count()),
            }
        )
    return pd.DataFrame(rows)


def ananea_vs_peers_trend(history_df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    rows = []
    for source in sources:
        src = history_df[(history_df["Source"] == source) & history_df["Score"].notna()]
        if src.empty:
            continue
        for date in sorted(src["Date"].dropna().unique()):
            point = src[src["Date"] == date]
            ananea = point[point["Hotel"] == ANANEA_HOTEL]["Score"].dropna()
            peers = point[point["Hotel"] != ANANEA_HOTEL]["Score"].dropna()
            if not ananea.empty:
                rows.append({"Date": date, "Series": f"{source} | Ananea", "Score": float(ananea.iloc[0])})
            if not peers.empty:
                rows.append({"Date": date, "Series": f"{source} | Peers Avg", "Score": float(peers.mean())})
    return pd.DataFrame(rows)


def ananea_latest_comparison(history_df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    rows = []
    for source in sources:
        src = history_df[(history_df["Source"] == source) & history_df["Score"].notna()]
        if src.empty:
            continue
        latest_date = src["Date"].max()
        latest = src[src["Date"] == latest_date]
        ananea = latest[latest["Hotel"] == ANANEA_HOTEL]
        if ananea.empty:
            continue
        ananea_score = float(ananea["Score"].iloc[0])
        for _, row in latest[latest["Hotel"] != ANANEA_HOTEL].iterrows():
            rows.append(
                {
                    "Source": source,
                    "Date": latest_date,
                    "Hotel": row["Hotel"],
                    "Hotel Score": round(float(row["Score"]), 2),
                    "Ananea Score": round(ananea_score, 2),
                    "Ananea - Hotel": round(ananea_score - float(row["Score"]), 2),
                }
            )
    return pd.DataFrame(rows)


def ananea_competitive_index(history_df: pd.DataFrame, sources: list[str]) -> dict[str, float | int]:
    """
    KPI on 0-100 scale:
    - ananea_index: average normalized Ananea score across selected sources
    - peers_index: average normalized peers score across selected sources
    - edge_pp: Ananea advantage/disadvantage in percentage points
    """
    rows = ananea_scorecard_rows(history_df, sources)
    if rows.empty:
        return {"ananea_index": float("nan"), "peers_index": float("nan"), "edge_pp": float("nan"), "sources_used": 0}

    normalized = []
    for _, row in rows.iterrows():
        source = str(row["Source"])
        scale = SCALE_MAX.get(source, 10.0)
        ananea_score = row.get("Ananea Score")
        peers_avg = row.get("Peers Avg")
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

    st.subheader("Ananea Scorecard")
    st.caption("Focused comparison for Ananea Castelo Suites Hotel across selected sources.")
    scorecard = ananea_scorecard_rows(history_df, selected_sources)
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

    if scorecard.empty:
        st.warning("No Ananea scorecard data available for the selected sources.")
    else:
        metric_cols = st.columns(max(1, len(scorecard)))
        for i, row in scorecard.reset_index(drop=True).iterrows():
            value = f"{row['Ananea Score']:.2f}/{SCALE_MAX.get(row['Source'], 10):.0f}"
            delta = None
            if pd.notna(row["Delta vs Peers"]):
                delta = f"{row['Delta vs Peers']:+.2f} vs peers"
            metric_cols[i].metric(label=f"{row['Source']} ({row['Date'].date()})", value=value, delta=delta)
        st.dataframe(scorecard.sort_values("Source"), use_container_width=True)

    st.subheader("Ananea vs Other Hotels")
    compare_latest = ananea_latest_comparison(history_df, selected_sources)
    if compare_latest.empty:
        st.warning("No latest-date comparison rows available.")
    else:
        st.dataframe(
            compare_latest.sort_values(["Source", "Ananea - Hotel"], ascending=[True, False]),
            use_container_width=True,
        )

    trend_compare = ananea_vs_peers_trend(history_df, selected_sources)
    if not trend_compare.empty:
        st.markdown("**Ananea vs Peers Trend**")
        trend_chart = trend_compare.pivot(index="Date", columns="Series", values="Score").sort_index()
        st.line_chart(trend_chart, use_container_width=True)

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
    hotel_link = HOTEL_LINKS.get(source, {}).get(hotel)

    if hotel_link:
        st.markdown(f"Hotel link: [{hotel_link}]({hotel_link})")
    else:
        st.caption("No direct hotel link configured for this source/hotel.")

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
