import streamlit as st
import json
from pathlib import Path
from chains.recommender_chain import run_recommender
from dotenv import load_dotenv

load_dotenv()

def load_json_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text())

def main():
    st.set_page_config(page_title="Multi-Agent Finance Dashboard", layout="wide")
    st.title("🔍 Multi-Agent Finance Dashboard")

    # Sidebar: user preferences
    st.sidebar.header("User Preferences")
    risk = st.sidebar.selectbox("Risk Tolerance", ["low", "moderate", "high"], index=1)
    freq = st.sidebar.selectbox("Trade Frequency", ["weekly", "monthly", "major_events"], index=0)
    sectors = st.sidebar.multiselect("Sector Preferences", ["tech", "healthcare"], default=["tech"])
    style = st.sidebar.selectbox("Trading Style", ["value", "growth", "momentum"], index=0)
    level = st.sidebar.selectbox("Experience Level", ["beginner", "mid", "advanced"], index=1)
    scoring = st.sidebar.text_area(
        "Scoring Method",
        "sentiment_score * 0.5 + YoY_revenue_growth * 0.3 + net_profit_margin * 0.2"
    )

    if st.sidebar.button("Get Recommendations"):
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir / "data"

        # Load the two JSON fixtures from src/data
        fundamentals = load_json_file(data_dir / "fundamentals.json")
        sentiments   = load_json_file(data_dir / "sentiments.json")

        # Build user prefs from the UI
        user_prefs = {
            "risk_tolerance":   risk,
            "trade_frequency":  freq,
            "sector_preference": sectors,
            "trading_style":    style,
            "experience_level": level,
        }

        with st.spinner("Computing recommendations..."):
            recs = run_recommender(
                fundamentals=fundamentals,
                sentiments=sentiments,
                user_preferences=user_prefs,
                scoring_method=scoring
            )

        st.subheader("Top 5 Stock Picks")
        if recs:
            for r in recs:
                st.markdown(f"**{r['ticker']}** — {r['rationale']} _(score: {r['score']:.2f})_")
        else:
            st.warning("No recommendations returned. Check your inputs.")

if __name__ == "__main__":
    main()
