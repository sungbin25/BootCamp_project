"""
종목명/코드 검색용 매핑 모듈.
1) 주요 고정 종목 매핑 제공
2) 6자리 종목 코드 입력 시 (086520 등) 동적 코스피(.KS)/코스닥(.KQ) 판별 및 연결
3) 전 세계/국내 모든 상장 종목(에코프로, 알테오젠, 삼천당제약 등 2,600+개) 실시간 동적 검색 및 캐싱 지원
"""

import re
import requests
import logging

logger = logging.getLogger("ticker_map")

TICKER_MAP = {
    "삼성전자": "005930.KS",
    "sk하이닉스": "000660.KS",
    "SK하이닉스": "000660.KS",
    "naver": "035420.KS",
    "NAVER": "035420.KS",
    "네이버": "035420.KS",
    "카카오": "035720.KS",
    "lg에너지솔루션": "373220.KS",
    "LG에너지솔루션": "373220.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "posco홀딩스": "005490.KS",
    "삼성바이오로직스": "207940.KS",
    "셀트리온": "068270.KS",
    "kb금융": "105560.KS",
    "신한지주": "055550.KS",
}

# 표시명 역매핑
DISPLAY_NAME = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "035420.KS": "NAVER",
    "035720.KS": "카카오",
    "373220.KS": "LG에너지솔루션",
    "005380.KS": "현대차",
    "000270.KS": "기아",
    "005490.KS": "POSCO홀딩스",
    "207940.KS": "삼성바이오로직스",
    "068270.KS": "셀트리온",
    "105560.KS": "KB금융",
    "055550.KS": "신한지주",
}


def resolve_krx_code(code_6digit: str) -> dict | None:
    """6자리 종목코드로 네이버 금융에서 코스피(.KS)/코스닥(.KQ) 여부와 정식 종목명을 조회하여 동적 매핑합니다."""
    code_clean = code_6digit.strip()
    if not re.match(r'^\d{6}$', code_clean):
        return None

    url = f"https://finance.naver.com/item/main.naver?code={code_clean}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
        if r.encoding is None or r.encoding.lower() in ('iso-8859-1', 'euc-kr'):
            r.encoding = 'euc-kr'
        text = r.text

        name_m = re.search(r'<h2>\s*<a[^>]*>([^<]+)</a>', text)
        if name_m:
            stock_name = name_m.group(1).strip()
            is_kq = 'btn_kosdaq.gif' in text
            suffix = ".KQ" if is_kq else ".KS"
            full_ticker = f"{code_clean}{suffix}"

            # 메모리 캐시 자동 등록
            TICKER_MAP[stock_name.lower()] = full_ticker
            TICKER_MAP[stock_name] = full_ticker
            DISPLAY_NAME[full_ticker] = stock_name
            return {"ticker": full_ticker, "name": stock_name}
    except Exception as e:
        logger.warning(f"Failed to resolve KRX code {code_clean}: {e}")

    # Fallback default .KS
    full_ticker = f"{code_clean}.KS"
    return {"ticker": full_ticker, "name": code_clean}


def search_online_naver(query: str) -> list[dict]:
    """네이버 금융 실시간 종목 검색 (전체 코스피/코스닥 2,600+ 종목 검색 가능)"""
    query_norm = query.strip()
    if not query_norm:
        return []

    url = f"https://finance.naver.com/search/search.naver?query={requests.utils.quote(query_norm.encode('euc-kr', 'ignore'))}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
        html = r.content.decode("euc-kr", errors="ignore")

        # 1. 단일 종목 리다이렉트 검색 (예: 알테오젠, 삼천당제약, 005930 등)
        redirect = re.search(r"location\.href='/item/main\.naver\?code=(\d+)'", html)
        if redirect:
            code = redirect.group(1)
            info = resolve_krx_code(code)
            if info:
                return [info]

        # 2. 다중 검색 결과 (예: 에코프로, HLB, 카카오 등)
        matches = re.findall(r'href="/item/main\.naver\?code=(\d+)"[^>]*>\s*([^<]+)\s*</a>', html)
        results = []
        seen = set()
        for code, name in matches:
            if code not in seen:
                seen.add(code)
                info = resolve_krx_code(code)
                if info:
                    results.append(info)
        if results:
            return results
    except Exception as e:
        logger.warning(f"Online ticker search failed for {query}: {e}")
    return []


def search_ticker(query: str) -> list[dict]:
    """
    종목명/코드 부분 일치 검색.
    1) 메모리 사전 검색
    2) 6자리 종목코드 입력 검색 (예: 086520, 086520.KQ)
    3) 국내/해외 모든 상장 종목 실시간 검색 및 자동 캐싱
    """
    query_norm = query.strip()
    if not query_norm:
        return []

    query_upper = query_norm.upper()
    query_lower = query_norm.lower()

    results = []
    seen = set()

    # 1) 이미 티커 형식으로 입력한 경우 (예: 005930.KS, 086520.KQ, AAPL)
    if query_upper in DISPLAY_NAME:
        ticker = query_upper
        results.append({"ticker": ticker, "name": DISPLAY_NAME[ticker]})
        return results

    # 2) 6자리 숫자로 종목코드를 직접 입력한 경우 (예: 086520, 196170, 000250)
    if re.match(r'^\d{6}$', query_norm):
        info = resolve_krx_code(query_norm)
        if info:
            return [info]

    # 3) 기존 사전 메모리 검색
    for name, ticker in TICKER_MAP.items():
        if query_lower in name.lower() and ticker not in seen:
            seen.add(ticker)
            results.append({"ticker": ticker, "name": DISPLAY_NAME.get(ticker, name)})

    if results:
        return results

    # 4) 미등록 종목인 경우 실시간 네이버/KRX 상장 종목 검색 및 자동 등록
    online_results = search_online_naver(query_norm)
    if online_results:
        return online_results

    # 5) 해외 주식 티커 직접 입력 Fallback (예: NVDA, AAPL, TSLA)
    if re.match(r'^[A-Za-z]{1,5}$', query_norm):
        ticker = query_upper
        name = query_upper
        TICKER_MAP[name.lower()] = ticker
        DISPLAY_NAME[ticker] = name
        return [{"ticker": ticker, "name": name}]

    return []
