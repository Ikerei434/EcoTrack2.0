# ------------------------------------------------------------
# EcoTrack – FINAL MERGED & FULLY FIXED VERSION
# ------------------------------------------------------------

import os
import io
from datetime import datetime
from typing import Dict

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="EcoTrack – Sustainable Investing", layout="wide")

st.title("🌱 EcoTrack: Investing in a Sustainable Future")
st.write("Search stocks, view sustainability scores, and build an ethical portfolio.")
st.write("This version contains ALL bug fixes including robust price-column detection.")

# ------------------------------------------------------------
# Universal price-column finder (NEW)
# ------------------------------------------------------------

def find_price_column(df: pd.DataFrame) -> str:
    """Return correct price column regardless of yfinance naming."""
    cols = list(df.columns)

    # Prefer Adj Close
    adj = [c for c in cols if "Adj Close" in c]
    if adj:
        return adj[0]

    # Fallback: any column with Close (e.g., Close NVDA)
    close = [c for c in cols if "Close" in c]
    if close:
        return sorted(close, key=len)[0]

    raise ValueError(f"No Close column found. Available: {cols}")


# ------------------------------------------------------------
# Constants & fallback data
# ------------------------------------------------------------

FALLBACK_SCORES = {
    "AAPL": 78, "TSLA": 92, "MSFT": 85, "GOOGL": 80,
    "AMZN": 70, "XOM": 35, "CVX": 30, "NEE": 83
}

FALLBACK_INDUSTRY = {
    "Technology": 75, "Automotive": 60, "Retail": 65,
    "Energy": 40, "Utilities": 55
}

GREEN_ALTERNATIVES = {
    "AAPL": ["MSFT", "NEE"],
    "AMZN": ["GOOGL", "MSFT"],
    "XOM": ["NEE", "TSLA"],
    "CVX": ["NEE", "TSLA"],
    "TSLA": ["NEE", "MSFT"]
}

CONTROVERSIAL_INDUSTRIES = {"Energy", "Tobacco", "Defense", "Coal"}

# ------------------------------------------------------------
# ESG loading helpers
# ------------------------------------------------------------

@st.cache_data
def try_autoload_esg_files() -> pd.DataFrame | None:
    candidates = [
        "esg_data.csv", "ESG_data.csv", "data/esg_data.csv",
        "/mnt/data/SP 500 ESG Risk Ratings.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                try:
                    return pd.read_csv(path, encoding="latin-1")
                except Exception:
                    pass
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if "ticker" in lc or "symbol" in lc:
            colmap[c] = "ticker"
        elif "esg" in lc and "score" in lc:
            colmap[c] = "esg_score"
        elif c.lower() == "score":
            colmap[c] = "esg_score"
        elif "carbon" in lc:
            colmap[c] = "carbon_emissions"
        elif "renew" in lc:
            colmap[c] = "renewable_usage"
        elif "water" in lc:
            colmap[c] = "water_impact"
        elif "govern" in lc:
            colmap[c] = "governance_score"
        elif "industry" in lc or "sector" in lc:
            colmap[c] = "industry"

    df = df.rename(columns=colmap)

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    return df


@st.cache_data
def load_esg_df(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file:
        try:
            return normalize_columns(pd.read_csv(uploaded_file))
        except Exception:
            uploaded_file.seek(0)
            return normalize_columns(pd.read_excel(uploaded_file))

    df = try_autoload_esg_files()
    return normalize_columns(df) if df is not None else None


def get_sustainability_score_from_df(df: pd.DataFrame | None, ticker: str):
    if df is None or "ticker" not in df.columns:
        return None
    row = df[df["ticker"] == ticker.upper()]
    if not row.empty and "esg_score" in row.columns:
        try:
            return float(row.iloc[0]["esg_score"])
        except:
            return None
    return None

# ------------------------------------------------------------
# ESG utilities
# ------------------------------------------------------------

def fake_breakdown(score: float) -> Dict[str, int]:
    cats = ["Carbon", "Energy", "Waste", "Water", "Governance"]
    vals = np.random.randint(8, 28, len(cats)).astype(float)
    vals = vals / vals.sum() * score
    vals = [int(round(v)) for v in vals]
    diff = int(score) - sum(vals)
    i = 0
    while diff:
        vals[i % len(vals)] += 1 if diff > 0 else -1
        diff -= 1 if diff > 0 else -1
        i += 1
    return dict(zip(cats, vals))


def get_sustainability_score(ticker: str) -> float:
    s = get_sustainability_score_from_df(esg_df, ticker)
    if s is not None:
        return float(s)
    return float(FALLBACK_SCORES.get(ticker.upper(), 60))


def get_breakdown(ticker: str) -> Dict[str, int]:
    if esg_df is not None:
        row = esg_df[esg_df["ticker"] == ticker]
        if not row.empty:
            out = {}
            for col, name in [
                ("carbon_emissions", "Carbon"),
                ("renewable_usage", "Energy"),
                ("water_impact", "Water"),
                ("governance_score", "Governance")
            ]:
                if col in row.columns:
                    try:
                        out[name] = float(row.iloc[0][col])
                    except:
                        pass

            if out:
                total = sum(out.values()) or 1
                score = get_sustainability_score(ticker)
                return {k: int(round(v / total * score)) for k, v in out.items()}

    return fake_breakdown(get_sustainability_score(ticker))


def get_red_flags_and_industry(ticker: str):
    flags = []
    score = get_sustainability_score(ticker)
    if score < 65:
        flags.append("⚠️ Low sustainability score")

    industry = None
    if esg_df is not None and "industry" in esg_df.columns:
        row = esg_df[esg_df["ticker"] == ticker]
        if not row.empty:
            industry = str(row.iloc[0]["industry"])

    if industry is None:
        if ticker in ("XOM", "CVX", "BP"):
            industry = "Energy"
        else:
            industry = "Technology"

    if industry in CONTROVERSIAL_INDUSTRIES:
        flags.append("🔥 Exposure to controversial or fossil-fuel industries")

    return flags, industry

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("🔎 Search for a Stock")
ticker = st.sidebar.text_input("Ticker:", value="AAPL").upper()
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload ESG CSV", type=["csv", "xlsx"])

esg_df = load_esg_df(uploaded_file)

# ------------------------------------------------------------
# Stock View
# ------------------------------------------------------------

if ticker:
    st.subheader(f"📈 Stock Data for {ticker}")
    col1, col2 = st.columns([2, 1])

    # --------------------
    # Stock Price Chart FIXED
    # --------------------
    with col1:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                st.warning("No price data found.")
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [" ".join(col).strip() for col in df.columns]

                df = df.reset_index()

                price_col = find_price_column(df)

                fig = px.line(
                    df,
                    x="Date",
                    y=price_col,
                    title=f"{ticker} Price History"
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching price data: {e}")

    # --------------------
    # ESG Panel
    # --------------------
    with col2:
        score = get_sustainability_score(ticker)
        st.metric(f"{ticker} Sustainability Score", f"{score:.2f}/100")

        flags, industry = get_red_flags_and_industry(ticker)
        if flags:
            st.markdown("### Environmental Red Flags")
            for f in flags:
                st.write("- " + f)

        industry_avg = FALLBACK_INDUSTRY.get(industry, 60)
        st.metric(
            f"{industry} Average",
            f"{industry_avg:.2f}/100",
            delta=f"{score - industry_avg:+.2f}"
        )

        alts = GREEN_ALTERNATIVES.get(ticker, [])
        if alts:
            st.info("🌿 Sustainable Alternatives: " + ", ".join(alts))

        breakdown = get_breakdown(ticker)
        br_df = pd.DataFrame({"category": breakdown.keys(), "value": breakdown.values()})
        st.plotly_chart(px.bar(br_df, x="category", y="value",
                               title="ESG Breakdown"), use_container_width=True)

        radar = go.Figure()
        cats = list(breakdown.keys())
        vals = list(breakdown.values())
        radar.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=cats + cats[:1], fill="toself"
        ))
        radar.update_layout(polar=dict(radialaxis=dict(range=[0, 100])),
                            showlegend=False)
        st.plotly_chart(radar, use_container_width=True)

        st.markdown("### 📰 Sustainability News")
        news = [
            f"{ticker} announces new renewable energy goals ({datetime.now().year}).",
            f"{ticker} expands sustainability-focused R&D.",
            f"Analysts evaluate {ticker}'s emissions reduction progress."
        ]
        for n in news:
            st.write("- " + n)

# ------------------------------------------------------------
# Portfolio Builder
# ------------------------------------------------------------

st.markdown("---")
st.subheader("📊 Build Your Sustainable Portfolio")

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = {}

portfolio = st.session_state["portfolio"]

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    pt_ticker = st.text_input("Add Ticker:").upper().strip()
with col2:
    pt_amount = st.number_input("Amount ($):", min_value=0.0, step=100.0)
with col3:
    if st.button("Add"):
        if pt_ticker and pt_amount > 0:
            portfolio[pt_ticker] = portfolio.get(pt_ticker, 0) + pt_amount
            st.success(f"Added ${pt_amount} of {pt_ticker}.")

if portfolio:
    df_port = pd.DataFrame({
        "Ticker": list(portfolio.keys()),
        "Invested": list(portfolio.values()),
        "ESG Score": [get_sustainability_score(t) for t in portfolio]
    })
    st.dataframe(df_port)

    total = sum(portfolio.values())
    weighted = sum(get_sustainability_score(t) * (amt / total)
                   for t, amt in portfolio.items())
    st.metric("🌍 Portfolio Sustainability", f"{weighted:.2f}/100")

    # -----------------------
    # Portfolio Performance FIXED
    # -----------------------
    st.write("### 📈 Portfolio Performance")
    try:
        tickers_list = list(portfolio.keys())
        data = yf.download(tickers_list, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.warning("No portfolio price data available.")
        else:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [" ".join(col).strip() for col in data.columns]

            data = data.reset_index()

            price_cols = []
            for t in tickers_list:
                matches = [c for c in data.columns if t in c and "Close" in c]
                if matches:
                    price_cols.append(sorted(matches, key=len)[0])
                else:
                    # fallback
                    fallback = [c for c in data.columns if "Close" in c]
                    if fallback:
                        price_cols.append(sorted(fallback, key=len)[0])

            df_plot = data[["Date"] + price_cols].copy()

            # rename columns for readability
            new_names = ["Date"] + [
                col.replace("Close ", "").replace("Adj Close ", "")
                for col in price_cols
            ]
            df_plot.columns = new_names

            fig2 = px.line(
                df_plot,
                x="Date",
                y=df_plot.columns[1:],
                title="Portfolio Price Performance"
            )
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Portfolio chart error: {e}")

if st.button("Clear Portfolio"):
    st.session_state["portfolio"] = {}
    st.success("Portfolio cleared.")

st.markdown("---")
st.header("🔎 Data Transparency & Privacy")

st.write("""
- ESG values shown are placeholders unless you upload a CSV.
- Uploading data makes it visible to anyone using the same instance.
""")
