"""
커뮤니티(종목토론방) 게시글 수집 및 LLM + 사전 혼합 방식 파이프라인 모듈

특징:
1. 네이버 종목토론방 50~100개 게시글 수집 및 광고·중복 정제
2. 주식 용어 사전 기반 1차 판별 (명확한 상승/하락 표현 ⚡사전 즉시 처리)
3. 애매한 게시글만 Ollama LLM 배치 2차 분류 (🤖LLM 처리)
4. 6가지 투자 의도 카테고리 분류 (Bullish, Bearish, Neutral, Question, Humor, News)
5. 최근 50~100개 게시글 비중 집계 및 LLM 종합 여론/이슈 요약
"""

import os
import glob
import json
import re
import time
import time
import logging
from collections import Counter
import pandas as pd
import requests
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from sentiment_utils import (
    pre_classify_intent_dict,
    is_spam_post,
    is_garbled_korean,
    BULLISH_EXPLICIT_WORDS,
    BEARISH_EXPLICIT_WORDS
)
from llm_service import _chat

logger = logging.getLogger("community-pipeline")

BASE_NAVER_URL = "https://finance.naver.com/item/board.naver"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def crawl_live_naver_board(ticker_code: str, target_count: int = 300, max_time_sec: float = 6.0) -> list[dict]:
    """
    네이버 종목토론방에서 제한시간(max_time_sec) 동안 최대한 많은 게시글을 수집합니다.
    (UTF-8 / EUC-KR 자동 인코딩 판별 및 깨진 한글 차단)
    """
    records = []
    clean_code = ticker_code.split(".")[0]
    pages_to_crawl = max(15, (target_count // 15) + 2)
    start_time = time.time()

    for page in range(1, pages_to_crawl + 1):
        if len(records) >= target_count or (time.time() - start_time) >= max_time_sec:
            break
        try:
            params = {"code": clean_code, "page": page}
            resp = requests.get(BASE_NAVER_URL, headers=HEADERS, params=params, timeout=3)
            if resp.status_code != 200:
                continue

            # 네이버 금융 토론방 UTF-8 / EUC-KR 인코딩 자동 처리
            if resp.encoding is None or resp.encoding.lower() in ('iso-8859-1', 'latin-1'):
                resp.encoding = resp.apparent_encoding if resp.apparent_encoding else 'utf-8'

            html_text = resp.text

            if BeautifulSoup is not None:
                soup = BeautifulSoup(html_text, "html.parser")
                rows = soup.select("table.type2 tr")

                for row in rows:
                    if len(records) >= target_count or (time.time() - start_time) >= max_time_sec:
                        break
                    title_tag = row.select_one("td.title a")
                    date_tag = row.select_one("td:nth-of-type(1) span")
                    writer_tag = row.select_one("td:nth-of-type(3)")
                    views_tag = row.select_one("td:nth-of-type(6)")

                    if not title_tag:
                        continue

                    title = title_tag.get_text(strip=True)
                    # 깨진 한글 인코딩 방지 및 스팸 필터링
                    if not title or is_spam_post(title) or is_garbled_korean(title) or not re.search(r'[가-힣A-Za-z0-9]', title):
                        continue

                    date_str = date_tag.get_text(strip=True) if date_tag else "최근"
                    writer_str = writer_tag.get_text(strip=True) if writer_tag else "익명"

                    records.append({
                        "title": title,
                        "writer": writer_str,
                        "date": date_str,
                        "views": views_tag.get_text(strip=True) if views_tag else "0"
                    })
            else:
                titles = re.findall(r'href="/item/board_read\.naver\?[^"]*"[^>]*title="([^"]+)"', html_text)
                if not titles:
                    titles = re.findall(r'class="title"[^>]*>.*?<a[^>]*title="([^"]+)"', html_text, re.DOTALL)
                if not titles:
                    titles = re.findall(r'class="title"[^>]*>.*?<a[^>]*>([^<]+)</a>', html_text, re.DOTALL)

                for t in titles:
                    if len(records) >= target_count or (time.time() - start_time) >= max_time_sec:
                        break
                    t_clean = t.strip()
                    t_clean = t_clean.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&").replace("&quot;", '"')
                    if t_clean and not is_spam_post(t_clean) and not is_garbled_korean(t_clean) and re.search(r'[가-힣A-Za-z0-9]', t_clean):
                        records.append({
                            "title": t_clean,
                            "writer": "익명",
                            "date": "최근",
                            "views": "0"
                        })

            time.sleep(0.12)
        except Exception as e:
            logger.warning(f"네이버 라이브 크롤링 실패 (page {page}): {e}")
    return records


def get_raw_community_posts(ticker: str, target_count: int = 300) -> list[dict]:
    """
    로컬 Parquet 파일 또는 네이버 실시간 크롤링으로 최대한 많은 게시글을 수집합니다.
    """
    clean_ticker = ticker.split(".")[0]
    possible_dirs = [
        "/opt/data/raw/community",
        "data/raw/community",
        "../data/raw/community",
    ]

    all_posts = []
    found_files = []

    for d in possible_dirs:
        if os.path.exists(d):
            found_files.extend(glob.glob(os.path.join(d, f"community_{clean_ticker}_*.parquet")))

    if found_files:
        found_files.sort(key=os.path.getmtime, reverse=True)
        for fpath in found_files[:3]:
            try:
                df = pd.read_parquet(fpath)
                for _, row in df.iterrows():
                    title = row.get("title", "")
                    if not title:
                        title = row.get("content", "")
                    title_str = str(title).strip()
                    if not title_str or is_spam_post(title_str) or is_garbled_korean(title_str) or not re.search(r'[가-힣A-Za-z0-9]', title_str):
                        continue
                    all_posts.append({
                        "title": title_str,
                        "writer": str(row.get("writer", "익명")) if "writer" in row else "익명",
                        "date": str(row.get("date", "최근")) if "date" in row else "최근",
                    })
            except Exception as e:
                logger.debug(f"Parquet 로드 중 예외: {e}")

    if len(all_posts) < 50:
        live_posts = crawl_live_naver_board(clean_ticker, target_count=target_count)
        all_posts.extend(live_posts)

    seen = set()
    unique_posts = []
    for p in all_posts:
        if p["title"] not in seen:
            seen.add(p["title"])
            unique_posts.append(p)

    return unique_posts[:target_count]


def classify_ambiguous_posts_with_llm(ambiguous_posts: list[dict]) -> list[str]:
    """
    사전 판별이 안 된 게시글들을 LLM(Ollama)에 배치 전달하여 6가지 카테고리로 2차 분류합니다.
    (문장에 '?'나 'ㅋ'가 있더라도 상승/하락 기대 맥락을 최우선 평가합니다.)
    """
    if not ambiguous_posts:
        return []

    titles = [p["title"] for p in ambiguous_posts]

    # 배치당 20개씩 분할하여 LLM 타임아웃 방지
    batch_size = 20
    intents = [None] * len(titles)

    system_prompt = """너는 대한민국 최고의 주식 종목토론방 "투자 의도" 분류 AI다.
제공된 게시글 제목들을 읽고 6가지 카테고리(Bullish, Bearish, Neutral, Question, Humor, News) 중 하나로 정확히 분류하라.

[핵심 분류 지침 및 우선순위]
1. 최우선 분류 (Bullish / Bearish):
   - 매수 의도, 주가 상승 기대, 양전, 드가자, 사자, 목표가, 떡상 -> Bullish
   - 매도 의도, 하락 우려, 던지고 보자, 탈출, 음전, 손절, 뺀다, 떠나라, 하락 -> Bearish
2. Question (질문/학습): 공부법, 주식 문의, 왜, 몇층인가요, 언제 오르나요 등
3. Humor (유머/밈): Pure 유머, 감자전 등
4. Neutral (관망/중립): 매수/매도 의도가 완전히 없는 무색무취 글에만 한정.

반드시 JSON 형식을 엄수하라:
{"classifications": [{"id": 0, "intent": "Bullish"}, {"id": 1, "intent": "Bearish"}]}"""

    for b_idx in range(0, len(titles), batch_size):
        batch_titles = titles[b_idx : b_idx + batch_size]
        prompt_list = [f"{i}: {t}" for i, t in enumerate(batch_titles)]
        user_prompt = f"다음 게시글 제목들의 투자 의도를 정확히 분류하라:\n" + "\n".join(prompt_list)

        try:
            resp = _chat(system_prompt, user_prompt)
            if isinstance(resp, dict):
                items = resp.get("classifications", [])
                if not items and "results" in resp:
                    items = resp["results"]
                for item in items:
                    sub_i = item.get("id")
                    intent = item.get("intent")
                    if isinstance(sub_i, int) and 0 <= sub_i < len(batch_titles):
                        if intent in ["Bullish", "Bearish", "Neutral", "Question", "Humor", "News"]:
                            intents[b_idx + sub_i] = intent
        except Exception as e:
            logger.warning(f"LLM 배치 분류 중 예외 ({e}). 맥락 보정 적용.")

    # 정밀 맥락 보정 (Context-Aware Post-Processing)
    for i, t in enumerate(titles):
        if intents[i] is None or intents[i] == "Neutral":
            t_clean = t.strip()
            # 1. Bullish intent check
            if re.search(r'\b\d+층\b|드가자|가자|양전|\+\d+%|오르|안전|가겠|반등|상승|사자|수익|상한가|양전하|양전했|줍|매수|목표가|전망|사야|무조건 오를|우상향', t_clean):
                intents[i] = "Bullish"
            # 2. Bearish intent check
            elif re.search(r'-\d+%|-\d+퍼|인버스|음전|가난|손절|폭락|떡락|팔았|하락|개잡|축하|신저가|더떨|떨어|하락하|내려|망했|지옥|나락|손실|적자|부진|악재|뒤쳐|하락시|매도|던지|떠나|탈출|아작|폭탄|거지|폭망|하락빔|끝났으면', t_clean):
                intents[i] = "Bearish"
            # 3. Question check
            elif re.search(r'공부|초보|시작할 때|과목|어떻게|무슨|질문|왜|인가요|맞나요|가나요|오긴올까|알려주', t_clean):
                intents[i] = "Question"
            # 4. News check
            elif re.search(r'뉴스|공시|기사|속보|단독|특징주|금리인상', t_clean):
                intents[i] = "News"
            # 5. Humor check
            elif re.search(r'ㅋㅋㅋㅋ|ㅎㅎㅎㅎ|ㅋㅋ|ㅎㅎ|감자전', t_clean):
                intents[i] = "Humor"
            else:
                intents[i] = "Neutral"

    return intents



_COMMUNITY_CACHE = {}
_COMMUNITY_CACHE_TTL = 1800  # 30분 캐시


def process_community_pipeline(ticker: str, target_count: int = 300) -> dict:
    """
    전체 혼합 분석 파이프라인 실행:
    크롤링/로드 -> 1차 사전 분류 -> 2차 LLM 분류 -> 비중 집계 -> LLM 종합 요약
    """
    now = time.time()
    if ticker in _COMMUNITY_CACHE:
        entry = _COMMUNITY_CACHE[ticker]
        if now - entry["time"] < _COMMUNITY_CACHE_TTL:
            return entry["data"]

    raw_posts = get_raw_community_posts(ticker, target_count=target_count)

    if not raw_posts:
        return {
            "ticker": ticker,
            "total_posts": 0,
            "intent_counts": {"Bullish": 0, "Bearish": 0, "Neutral": 0, "Question": 0, "Humor": 0, "News": 0},
            "intent_percentages": {"Bullish": 0.0, "Bearish": 0.0, "Neutral": 100.0, "Question": 0.0, "Humor": 0.0, "News": 0.0},
            "overall_stance": "Neutral",
            "weighted_sentiment": 0.0,
            "summary": "최근 수집된 종목토론방 게시글 데이터가 없습니다.",
            "bullish_reasons": [],
            "bearish_reasons": [],
            "top_keywords": [],
            "posts": []
        }

    classified_posts = []
    ambiguous_items = []

    # Stage 1: 사전 기반 1차 판별
    for post in raw_posts:
        dict_res = pre_classify_intent_dict(post["title"])
        if dict_res:
            classified_posts.append({
                **post,
                "intent": dict_res["intent"],
                "method": dict_res["method"],
                "method_tag": dict_res["method_tag"],
                "sentiment_score": dict_res["score"]
            })
        else:
            ambiguous_items.append(post)

    # Stage 2: 애매한 게시글 2차 LLM 판별
    if ambiguous_items:
        llm_intents = classify_ambiguous_posts_with_llm(ambiguous_items)
        for post, intent in zip(ambiguous_items, llm_intents):
            score_map = {"Bullish": 0.6, "Bearish": -0.6, "Neutral": 0.0, "Question": 0.0, "Humor": 0.0, "News": 0.0}
            classified_posts.append({
                **post,
                "intent": intent,
                "method": "llm",
                "method_tag": "🤖LLM",
                "sentiment_score": score_map.get(intent, 0.0)
            })

    # Stage 3: 통계 및 비중 집계
    total = len(classified_posts)
    counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0, "Question": 0, "Humor": 0, "News": 0}
    for p in classified_posts:
        c = p.get("intent", "Neutral")
        counts[c] = counts.get(c, 0) + 1

    percentages = {k: round((v / total) * 100, 1) for k, v in counts.items()}

    # 감성 점수 계산 (-1.0 ~ 1.0)
    bull_cnt = counts["Bullish"]
    bear_cnt = counts["Bearish"]
    trading_total = max(1, bull_cnt + bear_cnt + counts["Neutral"])
    weighted_sentiment = round((bull_cnt - bear_cnt) / trading_total, 2)

    # 전체 기조 결정
    if bull_cnt > bear_cnt and percentages["Bullish"] >= 30:
        overall_stance = "Bullish (상승 기대 우세)"
    elif bear_cnt > bull_cnt and percentages["Bearish"] >= 30:
        overall_stance = "Bearish (하락 우려 우세)"
    else:
        overall_stance = "Neutral (관망세 및 주가 동향 확인)"

    # 키워드 추출
    all_titles_text = " ".join([p["title"] for p in classified_posts])
    words = [w for w in re.findall(r'[가-힣]{2,}', all_titles_text) if len(w) >= 2 and w not in ["삼성", "전자", "주식", "오늘", "지금"]]
    word_counts = Counter(words).most_common(5)
    top_keywords = [w[0] for w in word_counts]

    # Stage 4: LLM 전체 투자 심리 및 핵심 이슈 종합 요약
    ticker_name = ticker
    from ticker_map import DISPLAY_NAME
    ticker_name = DISPLAY_NAME.get(ticker, ticker)

    summary_info = generate_community_executive_summary(
        ticker_name=ticker_name,
        intent_counts=counts,
        intent_pcts=percentages,
        classified_posts=classified_posts,
        overall_stance=overall_stance
    )

    res = {
        "ticker": ticker,
        "name": ticker_name,
        "total_posts": total,
        "intent_counts": counts,
        "intent_percentages": percentages,
        "overall_stance": overall_stance,
        "weighted_sentiment": weighted_sentiment,
        "summary": summary_info.get("summary", ""),
        "bullish_reasons": summary_info.get("bullish_reasons", []),
        "bearish_reasons": summary_info.get("bearish_reasons", []),
        "top_keywords": top_keywords or summary_info.get("top_keywords", []),
        "posts": classified_posts
    }
    _COMMUNITY_CACHE[ticker] = {"data": res, "time": now}
    return res


def generate_community_executive_summary(ticker_name: str, intent_counts: dict,
                                          intent_pcts: dict, classified_posts: list[dict],
                                          overall_stance: str) -> dict:
    """
    LLM을 호출하여 최근 50~100개 게시글과 분류 비중을 바탕으로 전체 심리 종합 요약을 생성합니다.
    """
    sample_titles = "\n".join([f"- [{p['method_tag']}/{p['intent']}] {p['title']}" for p in classified_posts[:25]])

    system_prompt = """너는 증권사 수석 리서치 센터장이다.
종목 토론방 50~100개 게시글의 6가지 투자 의도 분류 비중(Bullish, Bearish, Neutral, Question, Humor, News)과 주요 게시글을 종합 분석하여,
오늘의 종목 토론방 여론 분위기와 상승/하락 근거를 3~4문장의 전문적 한국어로 요약하라.

반드시 JSON 형식을 출력하라:
{
  "summary": "종합 심리 요약 문장...",
  "bullish_reasons": ["상승 기대 이유 1", "상승 기대 이유 2"],
  "bearish_reasons": ["하락/조정 우려 이유 1"],
  "top_keywords": ["키워드1", "키워드2", "키워드3"]
}"""

    user_prompt = f"""종목명: {ticker_name}
[투자 의도 비중 분포]
- Bullish (매수/상승기대): {intent_pcts.get('Bullish', 0)}% ({intent_counts.get('Bullish', 0)}건)
- Bearish (매도/하락우려): {intent_pcts.get('Bearish', 0)}% ({intent_counts.get('Bearish', 0)}건)
- Neutral (관망/중립): {intent_pcts.get('Neutral', 0)}% ({intent_counts.get('Neutral', 0)}건)
- Question (질문/문의): {intent_pcts.get('Question', 0)}% ({intent_counts.get('Question', 0)}건)
- Humor (유머/잡담): {intent_pcts.get('Humor', 0)}% ({intent_counts.get('Humor', 0)}건)
- News (뉴스/공시): {intent_pcts.get('News', 0)}% ({intent_counts.get('News', 0)}건)

[대표 수집 게시글 (사전/LLM 혼합)]
{sample_titles}
"""

    try:
        res = _chat(system_prompt, user_prompt)
        if isinstance(res, dict) and "summary" in res:
            return res
    except Exception as e:
        logger.warning(f"LLM 종합 요약 요청 중 실패 ({e}). 기본 요약 생성.")

    # 기본 Fallback 요약
    b_pct = intent_pcts.get("Bullish", 0)
    be_pct = intent_pcts.get("Bearish", 0)

    if b_pct > be_pct:
        fallback_summary = f"게시판 여론은 상승 기대(Bullish {b_pct}%)가 우세하며, 저가 매수세 및 모멘텀 호재 기대감이 형성되어 있습니다."
        b_reasons = ["매수 심리 유입 및 상승 모멘텀 기대", "저가 매수 타점 탐색 의견 증가"]
        be_reasons = ["단기 변동성 및 차익 실현 우려"]
    else:
        fallback_summary = f"게시판 여론은 하락 우려(Bearish {be_pct}%)가 상대적으로 높으며, 단기 가격 조정 및 수급 부담을 주시하고 있습니다."
        b_reasons = ["기술적 반등 유입 기대"]
        be_reasons = ["단기 매도세 압박 및 조정 리스크", "관망세 지속"]

    return {
        "summary": fallback_summary,
        "bullish_reasons": b_reasons,
        "bearish_reasons": be_reasons,
        "top_keywords": ["매수", "실적", "상승"]
    }
