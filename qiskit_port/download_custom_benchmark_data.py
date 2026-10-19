import json
from pathlib import Path

import pandas as pd
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "custom_portfolios"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2018-01-01"
END_DATE = None  # latest available trading date

UNIVERSES = {
    "custom_10_mixed": {
        "description": "Custom 10-asset mixed benchmark with technology, semiconductors, financials, airlines, gold, and bonds.",
        "tickers": ["MSFT", "AAPL", "NVDA", "AMZN", "INTC", "MU", "JPM", "AAL", "GLD", "TLT"],
    },
    "custom_12_aggressive": {
        "description": "Custom 12-asset aggressive benchmark with technology, semiconductors, turnaround/speculative names, cyclicals, financials, gold, and bonds.",
        "tickers": ["MSFT", "AAL", "NVDA", "INTC", "AAPL", "MU", "NOK", "BB", "AMZN", "JPM", "GLD", "TLT"],
    },
}


def download_close_prices(tickers):
    print("\nDownloading:", tickers)

    raw = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError(f"No data downloaded for {tickers}")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame({
            ticker: raw[ticker]["Close"]
            for ticker in tickers
            if ticker in raw.columns.get_level_values(0)
        })
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()

    return prices


def save_universe(name, info):
    prices = download_close_prices(info["tickers"])

    parquet_path = OUTPUT_DIR / f"{name}_prices.parquet.gzip"
    csv_path = OUTPUT_DIR / f"{name}_prices.csv"
    metadata_path = OUTPUT_DIR / f"{name}_metadata.json"

    prices.to_parquet(parquet_path, compression="gzip")
    prices.to_csv(csv_path)

    metadata = {
        "universe_name": name,
        "description": info["description"],
        "source": "Yahoo Finance via yfinance",
        "requested_start_date": START_DATE,
        "requested_end_date": END_DATE,
        "actual_start_date_after_cleaning": str(prices.index.min()),
        "actual_end_date_after_cleaning": str(prices.index.max()),
        "tickers_requested": info["tickers"],
        "tickers_downloaded": list(prices.columns),
        "num_rows": int(prices.shape[0]),
        "num_assets": int(prices.shape[1]),
        "auto_adjust": True,
        "parquet_file": str(parquet_path.relative_to(REPO_ROOT)),
        "csv_file": str(csv_path.relative_to(REPO_ROOT)),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\nSaved:", name)
    print("Shape:", prices.shape)
    print("Tickers:", list(prices.columns))
    print("Actual start:", prices.index.min())
    print("Actual end:", prices.index.max())
    print("Parquet:", parquet_path)
    print("CSV:", csv_path)
    print("Metadata:", metadata_path)


def main():
    for name, info in UNIVERSES.items():
        save_universe(name, info)

    print("\nDone. Custom benchmark datasets saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
