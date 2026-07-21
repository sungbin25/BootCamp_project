"""
정성 데이터 수집: 네이버페이 증권 종목토론방

[!] 사용 전 반드시 확인하세요
  1. https://finance.naver.com/robots.txt 를 먼저 확인해 크롤링 허용 범위를 지키세요.
  2. 이 스크립트는 '포트폴리오 데모용 소량 수집'을 전제로 요청 간격을 넉넉히(2~3초) 두었습니다.
     상업적 서비스로 확장 시에는 반드시 별도로 이용약관 및 저작권 검토가 필요합니다.
  3. 과도한 요청은 IP 차단으로 이어지므로 페이지 수를 낮게 유지하세요.

사용법:
  python crawl_community.py --ticker 005930 --pages 3 --out /opt/airflow/data/raw/community
"""
import argparse
import os
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://finance.naver.com/item/board.naver"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (portfolio-demo-crawler; contact: your_email@example.com)"
}
REQUEST_INTERVAL_SEC = 2.5  # 서버 부하를 주지 않기 위한 최소 간격


def fetch_page(ticker: str, page: int) -> list[dict]:
    params = {"code": ticker, "page": page}
    resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    if resp.encoding is None or resp.encoding.lower() in ('iso-8859-1', 'latin-1'):
        resp.encoding = resp.apparent_encoding if resp.apparent_encoding else 'utf-8'

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("table.type2 tr")

    records = []
    for row in rows:
        title_tag = row.select_one("td.title a")
        date_tag = row.select_one("td:nth-of-type(1) span")
        views_tag = row.select_one("td:nth-of-type(6)")
        if not title_tag:
            continue
        records.append({
            "ticker": ticker,
            "title": title_tag.get_text(strip=True),
            "date": date_tag.get_text(strip=True) if date_tag else None,
            "views": views_tag.get_text(strip=True) if views_tag else None,
            "crawled_at": datetime.now().isoformat(),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="종목코드 6자리 (예: 005930)")
    parser.add_argument("--pages", type=int, default=3, help="수집할 페이지 수 (소량 권장)")
    parser.add_argument("--out", default="/opt/airflow/data/raw/community")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_records = []

    for page in range(1, args.pages + 1):
        try:
            records = fetch_page(args.ticker, page)
            all_records.extend(records)
            print(f"[OK] page {page}: {len(records)} rows")
        except Exception as e:
            print(f"[ERROR] page {page}: {e}")
        time.sleep(REQUEST_INTERVAL_SEC)

    if not all_records:
        print("[WARN] 수집된 게시글 없음")
        return

    df = pd.DataFrame(all_records)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out, f"community_{args.ticker}_{ts}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[OK] saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
