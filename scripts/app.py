import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from io import StringIO
import numpy as np

# ML
from sklearn.linear_model import LinearRegression
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "ticket_prices.csv"
SUMMARY_PATH = BASE_DIR / "data" / "analysis_summary.csv"

# -----------------------
# Styling
# -----------------------
def set_theme(dark: bool):
    if dark:
        st.markdown(
            """
            <style>
            body, .stApp { background-color: #0e1117 !important; color: #fafafa !important; }
            .stDataFrame thead tr th { background-color: #1e222a !important; color: #fafafa !important; }
            .stMetric { background-color: #1e222a !important; border-radius: 8px; padding: 8px; }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stMetric { background-color: #ffffff !important; border-radius: 8px; padding: 8px; }
            </style>
            """,
            unsafe_allow_html=True
        )

# -----------------------
# Core helpers
# -----------------------
def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["theatre", "seat_type"]
    avg_prices = (
        df.groupby(group_cols + ["platform"])["total_price"]
        .mean()
        .reset_index()
        .pivot_table(index=group_cols, columns="platform", values="total_price")
        .reset_index()
    )

    def cheaper_platform(row):
        bms, paytm = row.get("BMS"), row.get("Paytm")
        if pd.isna(bms) or pd.isna(paytm):
            return None
        if bms < paytm:
            return "BMS"
        elif paytm < bms:
            return "Paytm"
        return "Tie"

    def price_diff(row):
        bms, paytm = row.get("BMS"), row.get("Paytm")
        if pd.isna(bms) or pd.isna(paytm):
            return None
        return abs(bms - paytm)

    def percent_diff(row):
        bms, paytm = row.get("BMS"), row.get("Paytm")
        if pd.isna(bms) or pd.isna(paytm) or (bms == paytm):
            return 0.0
        return round(100 * (abs(bms - paytm) / min(bms, paytm)), 2)

    avg_prices["cheaper_platform"] = avg_prices.apply(cheaper_platform, axis=1)
    avg_prices["price_diff"] = avg_prices.apply(price_diff, axis=1)
    avg_prices["percent_diff"] = avg_prices.apply(percent_diff, axis=1)
    return avg_prices

def detect_anomalies(df_platform_daily, window=5, z_thresh=2.0):
    """
    Rolling z-score anomaly detection on total_price (per platform daily agg df).
    """
    s = df_platform_daily["total_price"]
    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std(ddof=0)
    z = (s - rolling_mean) / rolling_std
    df_platform_daily["zscore"] = z
    return df_platform_daily[df_platform_daily["zscore"].abs() >= z_thresh]

def train_forecast(df_daily, horizon_days=7):
    """
    df_daily: date, total_price
    Returns: forecast_df with columns [ds, yhat, yhat_lower, yhat_upper] (if Prophet) or [ds, yhat] (LR)
    """
    df_daily = df_daily.dropna(subset=["total_price"]).copy()
    if df_daily.empty or df_daily["date"].nunique() < 3:
        return None, "Not enough data to forecast."

    df_daily.rename(columns={"date": "ds", "total_price": "y"}, inplace=True)

    if HAVE_PROPHET:
        try:
            model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            model.fit(df_daily[["ds", "y"]])
            future = model.make_future_dataframe(periods=horizon_days)
            forecast = model.predict(future)
            return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], None
        except Exception as e:
            # fallback to Linear Regression
            pass

    # Fallback – Linear Regression with ordinal dates
    x = np.array([d.toordinal() for d in df_daily["ds"]]).reshape(-1, 1)
    y = df_daily["y"].values
    model = LinearRegression().fit(x, y)

    last_date = df_daily["ds"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    x_future = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    y_pred = model.predict(x_future)

    forecast = pd.DataFrame({"ds": future_dates, "yhat": y_pred})
    return forecast, None

def auto_insights(summary_f, df_f, forecast_results):
    """
    Create a natural-language insight string.
    forecast_results: dict(platform -> forecast_df or None)
    """
    lines = []
    if summary_f.empty:
        return "Not enough data to generate insights."

    counts = summary_f["cheaper_platform"].value_counts(dropna=True)
    paytm_cheaper = counts.get("Paytm", 0)
    bms_cheaper = counts.get("BMS", 0)
    tie_count = counts.get("Tie", 0)
    total_pairs = paytm_cheaper + bms_cheaper + tie_count

    lines.append(f"Across {total_pairs} theatre-seat pairs, Paytm is cheaper in {paytm_cheaper}, BMS in {bms_cheaper}, with {tie_count} ties.")

    # biggest gap
    if "price_diff" in summary_f and not summary_f["price_diff"].isna().all():
        idxmax = summary_f["price_diff"].idxmax()
        if pd.notna(idxmax):
            row = summary_f.loc[idxmax]
            lines.append(
                f"Largest average gap: ₹{row['price_diff']:.0f} for {row['theatre']} ({row['seat_type']})."
            )

    # forecast mention
    for platform, fdf in forecast_results.items():
        if fdf is None:
            continue
        try:
            last = fdf.tail(1)["yhat"].values[0]
            lines.append(f"Forecast: {platform} average ticket price expected around ₹{last:.0f} in {len(fdf)} days.")
        except Exception:
            pass

    return " ".join(lines)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"`ticket_prices.csv` not found at: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    if not SUMMARY_PATH.exists():
        summary = generate_summary(df)
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(SUMMARY_PATH, index=False)
    else:
        summary = pd.read_csv(SUMMARY_PATH)

    return df, summary

def download_csv(dataframe, filename):
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    st.download_button(
        label=f"📥 Download {filename}",
        data=csv_buffer.getvalue(),
        file_name=filename,
        mime="text/csv"
    )

# -----------------------
# App
# -----------------------
def main():
    st.set_page_config(page_title="Movie Ticket Competitor Dashboard (AI Upgraded)", layout="wide")

    # Dark mode
    dark_mode = st.sidebar.checkbox("🌙 Dark mode", value=False)
    set_theme(dark_mode)

    st.title("🎬 Movie Ticket Competitor Analysis — AI Upgraded")
    st.caption("With forecasts, anomaly alerts & auto-generated insights")

    df, _ = load_data()
    if df.empty:
        st.warning("Your ticket_prices.csv is empty.")
        st.stop()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range
    all_dates = sorted(df["date"].dt.date.unique())
    start_default, end_default = (min(all_dates), max(all_dates)) if len(all_dates) > 1 else (all_dates[0], all_dates[0])
    dr = st.sidebar.date_input("Date Range", value=[start_default, end_default],
                               min_value=min(all_dates), max_value=max(all_dates))
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start_date, end_date = dr
    else:
        start_date, end_date = dr, dr

    df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

    # Other filters
    movies = sorted(df["movie"].dropna().unique().tolist())
    cities = sorted(df["city"].dropna().unique().tolist())
    seat_types = sorted(df["seat_type"].dropna().unique().tolist())
    theatres = sorted(df["theatre"].dropna().unique().tolist())

    movie_sel = st.sidebar.multiselect("Movie(s)", movies, default=movies)
    city_sel = st.sidebar.multiselect("City(ies)", cities, default=cities)
    seat_sel = st.sidebar.multiselect("Seat type(s)", seat_types, default=seat_types)
    theatre_sel = st.sidebar.multiselect("Theatre(s)", theatres, default=theatres)

    df_f = df[
        df["movie"].isin(movie_sel)
        & df["city"].isin(city_sel)
        & df["seat_type"].isin(seat_sel)
        & df["theatre"].isin(theatre_sel)
    ].copy()

    if df_f.empty:
        st.warning("No rows match the selected filters.")
        st.stop()

    # Tabs
    tab_overview, tab_forecast, tab_insights, tab_raw = st.tabs(
        ["📊 Overview", "📈 Forecasts (ML)", "🧠 Insights & Alerts", "🗂 Raw Data & Downloads"]
    )

    # ------- Overview Tab -------
    with tab_overview:
        summary_f = generate_summary(df_f)

        counts = summary_f["cheaper_platform"].value_counts(dropna=True)
        st.info(
            f"**Cheaper Count:** Paytm - {counts.get('Paytm', 0)}, "
            f"BMS - {counts.get('BMS', 0)}, Tie - {counts.get('Tie', 0)}"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Unique Theatres", df_f["theatre"].nunique())
        bms_mean = df_f.loc[df_f["platform"] == "BMS", "total_price"].mean()
        paytm_mean = df_f.loc[df_f["platform"] == "Paytm", "total_price"].mean()
        col2.metric("Avg BMS Price", f"₹{bms_mean:.2f}" if pd.notna(bms_mean) else "—")
        col3.metric("Avg Paytm Price", f"₹{paytm_mean:.2f}" if pd.notna(paytm_mean) else "—")

        st.subheader("Price Comparison by Theatre & Seat Type (Averaged)")
        if not summary_f.empty:
            styled = summary_f.style.apply(
                lambda x: [
                    'background-color: #2e7d32; color: white' if v == "Paytm"
                    else ('background-color: #c62828; color: white' if v == "BMS" else '')
                    for v in x
                ] if x.name == 'cheaper_platform' else [''] * len(x),
                axis=0
            )
            st.dataframe(styled, use_container_width=True)
        else:
            st.warning("No rows to display in comparison table.")

        st.subheader("Cheaper Platform Share")
        counts = summary_f["cheaper_platform"].value_counts(dropna=True)
        if counts.empty:
            st.info("No data available to draw the cheaper platform pie chart.")
        else:
            fig_pie = px.pie(
                names=counts.index,
                values=counts.values,
                title="Cheaper Platform Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Price Difference by Theatre (₹)")
        if not summary_f.empty:
            summary_sorted = summary_f.sort_values("price_diff", ascending=False)
            fig_bar = px.bar(
                summary_sorted,
                x="theatre",
                y="price_diff",
                color="cheaper_platform",
                text_auto=True,
                title="Absolute Price Difference (₹) by Theatre"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Price Trend Over Time (Avg per Day)")
        if not df_f.empty:
            trend = (
                df_f.assign(date=df_f["date"].dt.date)
                   .groupby(["date", "platform"])["total_price"]
                   .mean()
                   .reset_index()
            )
            fig_line = px.line(
                trend, x="date", y="total_price",
                color="platform", markers=True,
                title="Average Total Price per Day"
            )
            st.plotly_chart(fig_line, use_container_width=True)

    # ------- Forecasts Tab -------
    with tab_forecast:
        st.subheader("Forecast average prices per platform")

        horizon = st.sidebar.slider("Forecast horizon (days)", min_value=3, max_value=30, value=7)

        trend_all = (
            df_f.assign(date=df_f["date"].dt.date)
                .groupby(["date", "platform"])["total_price"]
                .mean()
                .reset_index()
        )

        forecast_results = {}
        for platform in trend_all["platform"].unique():
            st.markdown(f"### {platform}")
            dd = trend_all[trend_all["platform"] == platform].copy()
            dd.rename(columns={"date": "date"}, inplace=True)

            forecast_df, err = train_forecast(dd[["date", "total_price"]].rename(columns={"date": "date"}), horizon_days=horizon)
            if err:
                st.warning(f"{platform}: {err}")
                forecast_results[platform] = None
                continue

            if HAVE_PROPHET and "yhat" in forecast_df.columns and "yhat_lower" in forecast_df.columns:
                # Prophet case
                hist = dd.rename(columns={"date": "ds", "total_price": "y"})
                fig = px.line(hist, x="ds", y="y", markers=True, title=f"{platform} – Historical & Forecast")
                fig.add_scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="forecast")
                fig.add_scatter(x=forecast_df["ds"], y=forecast_df["yhat_upper"], mode="lines", name="upper", line=dict(dash="dash"))
                fig.add_scatter(x=forecast_df["ds"], y=forecast_df["yhat_lower"], mode="lines", name="lower", line=dict(dash="dash"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                # LR fallback
                hist = dd.rename(columns={"date": "ds", "total_price": "y"})
                fig = px.line(hist, x="ds", y="y", markers=True, title=f"{platform} – Historical & Forecast (LR)")
                fig.add_scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="forecast")
                st.plotly_chart(fig, use_container_width=True)

            forecast_results[platform] = forecast_df

    # ------- Insights & Alerts Tab -------
    with tab_insights:
        st.subheader("Auto-generated Insights")

        # recompute summary for insights
        summary_f = generate_summary(df_f)

        # Build insights text
        insights_text = auto_insights(summary_f, df_f, forecast_results if 'forecast_results' in locals() else {})
        st.write(insights_text)

        # Anomaly detection per platform
        st.subheader("Anomaly Alerts (Rolling Z-Score ≥ 2)")
        trend_all = (
            df_f.assign(date=df_f["date"].dt.date)
                .groupby(["date", "platform"])["total_price"]
                .mean()
                .reset_index()
        )
        any_anoms = False
        for platform in trend_all["platform"].unique():
            sub = trend_all[trend_all["platform"] == platform].copy()
            anomalies = detect_anomalies(sub.sort_values("date"))
            if not anomalies.empty:
                any_anoms = True
                st.warning(f"⚠️ {platform} anomalies:")
                st.dataframe(anomalies, use_container_width=True)
        if not any_anoms:
            st.success("No significant anomalies detected based on rolling z-score.")

    # ------- Raw Data & Downloads Tab -------
    with tab_raw:
        st.subheader("Filtered Raw Data")
        st.dataframe(df_f.sort_values("date"), use_container_width=True)

        st.subheader("Download Data")
        download_csv(df_f, "filtered_ticket_prices.csv")
        summary_f = generate_summary(df_f)
        download_csv(summary_f, "analysis_summary.csv")

if __name__ == "__main__":
    main()
