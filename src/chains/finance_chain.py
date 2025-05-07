import json
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from langchain.schema import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence

load_dotenv()

class FinanceAgent:
    def __init__(self, data_dir: Path):
        """
        data_dir/ contains per-ticker JSON files with keys:
          - 'profile': company profile dict or list
          - 'historical': list of OHLCV bars (date, open, high, low, close, volume)
        """
        self.data_dir = data_dir
        pitch_prompt = PromptTemplate(
            input_variables=["companyName", "sector", "ceo", "marketCap", "dividendYield"],
            template="""
        Write a one-paragraph elevator pitch for a public company with the following profile:
        - Name: {companyName}
        - Sector: {sector}
        - CEO: {ceo}
        - Market Cap: ${marketCap:.0f}
        - Dividend Yield: {dividendYield:.2%}

        Keep it concise and engaging.
        """.strip()
        )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)
        self.pitch_runner = RunnableSequence(pitch_prompt, llm)

    def _load_json(self, ticker: str) -> Dict:
        path = self.data_dir / f"{ticker}.json"
        if not path.exists():
            raise FileNotFoundError(f"Ticker data not found: {path}")
        return json.loads(path.read_text())

    def _to_df(self, raw: Dict) -> pd.DataFrame:
        data = raw.get("historical") or raw.get("history")
        if isinstance(data, dict) and "historical" in data:
            data = data["historical"]
        if not isinstance(data, list):
            raise ValueError(f"Historical data for ticker is not a list")
        df = pd.DataFrame(data)

        # === replace ambiguous .get(...) logic with explicit checks ===
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"])
        else:
            raise KeyError("No 'date' or 'timestamp' column in historical data")

        df.set_index("date", inplace=True)
        return df.sort_index()


    def elevator_pitch(self, profile: Dict) -> str:
        # Invoke the RunnableSequence to get an AIMessage or str
        raw_out = self.pitch_runner.invoke({
            "companyName":   profile.get("companyName", "N/A"),
            "sector":        profile.get("sector", "N/A"),
            "ceo":           profile.get("ceo", "N/A"),
            "marketCap":     profile.get("marketCap", 0),
            "dividendYield": profile.get("dividendYield", 0.0)
        })
        # Extract content if AIMessage
        if isinstance(raw_out, AIMessage):
            return raw_out.content.strip()
        return str(raw_out).strip()

    def compute_key_stats(self, df: pd.DataFrame) -> Dict:
        returns = df["close"].pct_change().dropna()
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        cum = (1 + returns).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        return {
            "annualized_return":     annual_return,
            "annualized_volatility": annual_vol,
            "max_drawdown":          max_dd
        }

    def compute_technicals(self, df: pd.DataFrame) -> Dict:
        tech = {}
        tech["SMA_20"] = df["close"].rolling(20).mean().iloc[-1]
        tech["EMA_20"] = df["close"].ewm(span=20).mean().iloc[-1]
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.ewm(span=14).mean() / down.ewm(span=14).mean()
        tech["RSI_14"] = (100 - (100 / (1 + rs))).iloc[-1]
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        tech["MACD_12_26"] = (ema12 - ema26).iloc[-1]
        m = df["close"].rolling(20).mean()
        s = df["close"].rolling(20).std()
        tech["BB_upper"] = (m + 2 * s).iloc[-1]
        tech["BB_lower"] = (m - 2 * s).iloc[-1]
        return tech

    def generate_signal(self, tech: Dict, price: float) -> str:
        if price < tech.get("BB_lower"):
            return "BUY"
        if price > tech.get("BB_upper"):
            return "SELL"
        return "HOLD"

    def compute_valuations(self, profile: Dict, price: float) -> Dict:
        return {
            "P/E": price / profile.get("eps", np.nan),
            "P/B": price / profile.get("bookValuePerShare", np.nan),
            "P/S": price / profile.get("revenuePerShare", np.nan)
        }

    def compare_price(self, df: pd.DataFrame, profile: Dict) -> Dict:
        hi52 = df["high"].rolling(252).max().iloc[-1]
        lo52 = df["low"].rolling(252).min().iloc[-1]
        rank_pct = df["close"].rank(pct=True).iloc[-1]
        dcf = profile.get("dcfEstimate", df["close"].iloc[-1])
        return {
            "52w_high":        hi52,
            "52w_low":         lo52,
            "range_percentile": rank_pct,
            "dcf_diff":         df["close"].iloc[-1] - dcf
        }

if __name__ == "__main__":
    # Dynamically walk back to your project root (adjust the number of parents if needed)
    project_root = Path.cwd()
    # this script lives in <project>/src/chains, so two parents up is the project root
    for _ in range(2):
        project_root = project_root.parent

    # 1) Data folder with JSONs at root:
    data_dir = project_root / "src" / "data"
    # 2) If your JSONs live in a subfolder, uncomment below:
    stock_dir = data_dir / "stock_data"

    # Glob all JSON files and extract ticker symbols
    tickers = [p.stem for p in stock_dir.glob("*.json")]

    agent = FinanceAgent(stock_dir)
    fundamentals = {}

    for ticker in tickers:
        raw = agent._load_json(ticker)
        profile = raw.get("profile") or raw
        if isinstance(profile, list):
            profile = profile[0]
        df = agent._to_df(raw)
        price = df["close"].iloc[-1]

        fundamentals[ticker] = {
            "elevator_pitch":   agent.elevator_pitch(profile),
            "key_stats":        agent.compute_key_stats(df),
            "technicals":       agent.compute_technicals(df),
            "signal":           agent.generate_signal(agent.compute_technicals(df), price),
            "valuations":       agent.compute_valuations(profile, price),
            "price_comparison": agent.compare_price(df, profile),
        }

    # Write your consolidated file back into that same folder
    out_path = data_dir / "fundamentals.json"
    out_path.write_text(json.dumps(fundamentals, indent=2))
    print(f"Wrote fundamentals for {len(fundamentals)} tickers to {out_path}")