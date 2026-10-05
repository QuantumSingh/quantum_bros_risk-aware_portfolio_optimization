"""Download the custom_nonpaper_10 benchmark universe.

Ten liquid US large caps deliberately disjoint from the original paper's
15-ticker universe, so Benchmark 2 tests the models on assets the original
work never saw.

Saves raw adjusted close PRICES. Benchmarks must convert to daily returns
with pct_change() before feeding models (the original repo's
price_data.parquet.gzip already contains returns, despite its name).
"""
import json
from pathlib import Path

import pandas as pd
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "custom_portfolios"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_NAME = "custom_nonpaper_10"

DESCRIPTION = (
    "Custom 10-asset benchmark universe with zero overlap with the original "
    "paper dataset. Mega-cap tech/semis (NVDA, AMD, AMZN, META, NFLX, TSLA), "
    "defensives (COST, UNH, KO), and industrials (BA)."
)

TICKERS = ["NVDA", "AMD", "TSLA", "AMZN", "META", "NFLX", "COST", "UNH", "BA", "KO"]

# Original paper tickers that must NOT appear in this universe.
PAPER_TICKERS = {
    "AAPL", "EFA", "GLD", "IWM", "JNJ", "JPM", "LQD", "MSFT",
    "QQQ", "SPY", "TLT", "USO", "XLF", "XLV", "XOM",
}

START_DATE = "2015-01-01"
END_DATE = None  # latest available trading date


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


def main():
    overlap = set(TICKERS) & PAPER_TICKERS
    if overlap:
        raise ValueError(f"Universe overlaps with paper tickers: {sorted(overlap)}")

    prices = download_close_prices(TICKERS)

    missing = set(TICKERS) - set(prices.columns)
    if missing:
        raise RuntimeError(f"Tickers failed to download: {sorted(missing)}")

    # Keep the requested column order.
    prices = prices[TICKERS]

    parquet_path = OUTPUT_DIR / f"{UNIVERSE_NAME}_prices.parquet.gzip"
    csv_path = OUTPUT_DIR / f"{UNIVERSE_NAME}_prices.csv"
    metadata_path = OUTPUT_DIR / f"{UNIVERSE_NAME}_metadata.json"

    prices.to_parquet(parquet_path, compression="gzip")
    prices.to_csv(csv_path)

    metadata = {
        "universe_name": UNIVERSE_NAME,
        "description": DESCRIPTION,
        "source": "Yahoo Finance via yfinance",
        "content": "adjusted close prices (NOT returns; apply pct_change before modeling)",
        "requested_start_date": START_DATE,
        "requested_end_date": END_DATE,
        "actual_start_date_after_cleaning": str(prices.index.min()),
        "actual_end_date_after_cleaning": str(prices.index.max()),
        "tickers_requested": TICKERS,
        "tickers_downloaded": list(prices.columns),
        "paper_ticker_overlap": sorted(set(prices.columns) & PAPER_TICKERS),
        "num_rows": int(prices.shape[0]),
        "num_assets": int(prices.shape[1]),
        "auto_adjust": True,
        "parquet_file": str(parquet_path.relative_to(REPO_ROOT)),
        "csv_file": str(csv_path.relative_to(REPO_ROOT)),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\nSaved:", UNIVERSE_NAME)
    print("Shape:", prices.shape)
    print("Tickers:", list(prices.columns))
    print("Range:", prices.index.min(), "->", prices.index.max())
    print("Paper overlap:", metadata["paper_ticker_overlap"])
    print("Files:", parquet_path.name, csv_path.name, metadata_path.name)


if __name__ == "__main__":
    main()
