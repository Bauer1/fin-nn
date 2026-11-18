from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils import PROJECT_ROOT, load_config

RAW_DIR = PROJECT_ROOT / "data" / "raw"

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    df = df[COLUMNS]
    df.index = df.index.tz_localize(None).rename("Date")
    return df


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Empty DataFrame: check the ticker and dates")
    if df.isna().any().any():
        raise ValueError(f"There are NA:\n{df.isna().sum()}")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Dates aren't sorted or duplicated")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Not positive prices")

    max_move = df["Close"].pct_change().abs().max()
    if max_move > 0.5:
        raise ValueError(f"Suspicious price spike within a day: {max_move:.0%}")

    zero_vol = int((df["Volume"] == 0).sum())
    max_gap = df.index.to_series().diff().max()
    if zero_vol:
        print(f"Attention: lines with zero volume: {zero_vol}")
    if max_gap > pd.Timedelta(days=5):
        print(f"Attention: there is a significant gap in dates: {max_gap}")


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def get_data(
    ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    if ticker is None or start is None or end is None:
        cfg = load_config()["data"]
        ticker = ticker or cfg["ticker"]
        start = start or cfg["start"]
        end = end or cfg["end"]

    path = RAW_DIR / f"{ticker}_{start}_{end}.csv"
    if path.exists() and not force:
        return load(path)
    df = download(ticker, start, end)
    validate(df)
    save(df, path)
    return load(path)


if __name__ == "__main__":
    df = get_data()
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    print(df.dtypes)
    print("\nSplit check (June 2024):")
    print(df.loc["2024-06-05":"2024-06-12"])
