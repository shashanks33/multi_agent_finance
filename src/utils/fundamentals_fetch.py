import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

# figure out where src/data/stock_data lives
BASE_DIR = Path(__file__).resolve().parent            # src/
STOCK_DATA_DIR = BASE_DIR / "data" / "stock_data"
STOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)

tickers = [] # Add your list of tickers here

def fetch_data(ticker):
    endpoints = {
        "profile": f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}",
        "historical": f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_API_KEY}",
        "income_statement": f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=4&apikey={FMP_API_KEY}",
        "balance_sheet": f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=4&apikey={FMP_API_KEY}",
        "cash_flow": f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?limit=4&apikey={FMP_API_KEY}",
        # "recommendation": f"https://financialmodelingprep.com/api/v4/analyst-estimates/{ticker}?apikey={FMP_API_KEY}",
        # "earnings": f"https://financialmodelingprep.com/api/v3/earnings_calendar/{ticker}?limit=4&apikey={FMP_API_KEY}"
    }

    data = {}
    for key, url in endpoints.items():
        response = requests.get(url)
        if response.ok:
            data[key] = response.json()
        else:
            data[key] = {"error": f"Failed to fetch {key}"}

    out_path = STOCK_DATA_DIR / f"{ticker}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

for ticker in tickers:
    print(f"Fetching data for {ticker}")
    fetch_data(ticker)

print("All data fetched successfully")
