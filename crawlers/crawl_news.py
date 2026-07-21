"""
정성 데이터 수집: 네이버 뉴스 검색 오픈API
- 무료, developers.naver.com 에서 애플리케이션 등록 후 Client ID/Secret 발급 (5분 소요)
- 일일 호출 한도가 있으므로 Airflow에서 종목별로 나눠 호출 권장

환경변수:
  NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
사용법:
  python crawl_news.py --query "삼성전자" --ticker 005930.KS --out /opt/airflow/data/raw/news
"""
import argparse
import os
import time
from datetime import datetime

import requests
import pandas as pd

NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"


def fetch_news(query: str, client_id: str, client_secret: str, display: int = 100) -> list[dict]:
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {"query": query, "display": display, "sort": "date"}
    resp = requests.get(NAVER_NEWS_URL, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("items", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="검색어 (예: 삼성전자)")
    parser.add_argument("--ticker", required=True, help="매핑할 종목코드")
    parser.add_argument("--out", default="/opt/airflow/data/raw/news")
    args = parser.parse_args()

    client_id = os.environ["NAVER_CLIENT_ID"]
    client_secret = os.environ["NAVER_CLIENT_SECRET"]

    os.makedirs(args.out, exist_ok=True)
    items = fetch_news(args.query, client_id, client_secret)

    if not items:
        print("[WARN] 수집된 뉴스 없음")
        return

    df = pd.DataFrame(items)
    df["ticker"] = args.ticker
    df["crawled_at"] = datetime.now().isoformat()
    # 네이버 API가 HTML 태그(<b>)를 포함해서 반환하므로 정제는 Spark 단계에서 처리

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out, f"news_{args.ticker}_{ts}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[OK] saved {len(df)} rows -> {out_path}")

    time.sleep(1)  # 호출 간격 (한도 보호)


if __name__ == "__main__":
    main()
