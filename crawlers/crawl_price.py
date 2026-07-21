"""
정량 데이터 수집: Yahoo Finance (yfinance) - 완전 무료, API 키 불필요
사용법: python crawl_price.py --tickers 005930.KS 000660.KS --out /opt/airflow/data/raw/price
"""
import argparse
import os
from datetime import datetime

import yfinance as yf
import pandas as pd


def crawl_price(tickers: list[str], period: str = "60d", interval: str = "1d") -> pd.DataFrame:
    frames = []
    # Always include macro tickers
    macro_tickers = ["^KS11", "^KQ11", "KRW=X", "^VIX"]
    all_tickers = list(dict.fromkeys(tickers + macro_tickers)) # remove duplicates keeping order
    for ticker in all_tickers:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            print(f"[WARN] no data for {ticker}")
            continue
        df = df.reset_index()
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            
        df["ticker"] = ticker
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True,
                         help="예: 005930.KS(삼성전자) 035420.KS(네이버) AAPL")
    parser.add_argument("--out", default="/opt/airflow/data/raw/price")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="1d")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = crawl_price(args.tickers, args.period, args.interval)

    if df.empty:
        print("[ERROR] 수집된 데이터 없음")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out, f"price_{ts}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[OK] saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
