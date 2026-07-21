"""
Stock Insight AI Platform - Backend API (FastAPI)

UR-01 종목 검색           -> GET  /search
UR-02~03 기간별 주가 차트  -> GET  /prices/{ticker}
UR-04~05 변곡점 탐지/마커  -> GET  /changepoints/{ticker}
UR-06~09 마커 클릭 -> 뉴스수집+LLM요약+영향도분석 -> GET /events/{ticker}/detail
UR-10~12 미래 예측 + 근거 -> GET  /predict/{ticker}
"""
"""TRACKED_STOCKS = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "035420.KS": "NAVER",
    "035720.KS": "카카오",
    "005380.KS": "현대차",
    "051910.KS": "LG화학",
    "373220.KS": "LG에너지솔루션",
}"""
from analyze_service import analyze_stock
from collections import Counter
import re
import time

from news_service import (
    fetch_news_window,
    fetch_market_news,
    summarize_article_sentiment,
    process_news_3d_pipeline
)

from sentiment_utils import COMMUNITY_POSITIVE_WORDS, COMMUNITY_NEGATIVE_WORDS

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from ticker_map import search_ticker, DISPLAY_NAME
from changepoint import detect_changepoints, analyze_changepoint_detail
#from news_service import fetch_news_window
from llm_service import summarize_news, analyze_impact, explain_prediction, summarize_community, ask_llm
from predict_service import predict as predict_price, simulate_predict
from pattern_service import find_similar_patterns
from db import init_db, get_db, PredictionLog, NewsAnalysisLog
from pydantic import BaseModel

NAME_TO_TICKER = {
    name: ticker
    for ticker, name in DISPLAY_NAME.items()
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-insight-api")

app = FastAPI(title="Stock Insight AI Platform API")

PERIOD_MAP = {
    "1d": ("5d", "5m"),
    "1w": ("1mo", "30m"),
    "1m": ("3mo", "1d"),
    "3m": ("6mo", "1d"),
    "6m": ("1y", "1d"),
    "1y": ("2y", "1d"),
    "2y": ("2y", "1d"),
    "3y": ("5y", "1wk"),
    "5y": ("10y", "1wk"),
    "all": ("max", "1mo"),
}


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("DB initialized")


# ---------------- UR-01: 종목 검색 ----------------
@app.get("/search")
def search(q: str = Query(..., min_length=1)):
    results = search_ticker(q)
    return {"query": q, "results": results}


def _fetch_price_df(ticker: str, period_label: str) -> pd.DataFrame:
    if period_label not in PERIOD_MAP:
        raise HTTPException(400, f"지원하지 않는 기간: {period_label}")

    yf_period, yf_interval = PERIOD_MAP[period_label]
    df = yf.download(ticker, period=yf_period, interval=yf_interval, progress=False)
    if df.empty:
        raise HTTPException(404, f"'{ticker}' 데이터를 찾을 수 없습니다")

    # yfinance MultiIndex 컬럼 평탄화 (reset_index 전 처리)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
    df = df.rename(columns={date_col: "date"})

    if "date" not in df.columns:
        df["date"] = df.index

    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    return df


@app.get("/prices/{ticker}")
def get_prices(ticker: str, period: str = "1m"):
    df = _fetch_price_df(ticker, period)
    records = df[["date", "Close", "Volume", "ma5", "ma20"]].copy()
    records["date"] = records["date"].astype(str)

    records = records.replace([np.nan, np.inf, -np.inf], None)
    return {
        "ticker": ticker,
        "name": DISPLAY_NAME.get(ticker, ticker),
        "period": period,
        "data": records.to_dict(orient="records"),
    }


# ---------------- UR-04, UR-05: 변곡점 탐지 + 마커 ----------------
@app.get("/changepoints/{ticker}")
def get_changepoints(ticker: str, period: str = "1y"):
    df = _fetch_price_df(ticker, period)
    points = detect_changepoints(df[["date", "Close", "Volume"]])
    return {"ticker": ticker, "period": period, "changepoints": points}


# ---------------- UR-06~09: 변곡점 상세 (기술적 원인 + Feature 표 + AI 회고 시뮬레이션 + 신뢰도) ----------------
@app.get("/events/{ticker}/detail")
def get_event_detail(ticker: str, date: str, price_change_pct: float = 0.0,
                      db: Session = Depends(get_db)):
    """
    date: 'YYYY-MM-DD' (마커 클릭한 변곡점 날짜)
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "date는 YYYY-MM-DD 형식이어야 합니다")

    df = _fetch_price_df(ticker, "1y")
    detail = analyze_changepoint_detail(df, date)
    if not detail:
        ticker_name = DISPLAY_NAME.get(ticker, ticker)
        return {
            "date": date,
            "price_change_pct": price_change_pct,
            "causes": ["✔ 가격 변동성 기준 변곡점 포착"],
            "ai_judgement": "기술적 변곡 구간",
            "sentiment": "Neutral",
            "feature_table": [],
            "ai_replay": {
                "headline": "현재 AI 모델을 당시 데이터에 적용한 회고 시뮬레이션 결과",
                "bullish_score": 50,
                "expected_pct": 0.0,
                "actual_pct": price_change_pct,
                "forecast_dates": [date],
                "forecast_prices": [0],
                "actual_prices": [0],
            },
            "confidence": {"score": 75, "stars": "★★★☆☆", "reasons": ["변동성 도달"]},
            "ai_comment": f"{ticker_name} {date} 변곡점에 대한 기술적 분석 정보입니다.",
        }

    return detail

# ---------------- 시장 뉴스 기반 이슈 종목 ----------------
#
# 동작 순서
# 1. 시장 뉴스 수집
# 2. 기사 제목/설명에서 종목명 탐색
# 3. Counter로 언급 횟수 집계
# 4. 언급량 + 주가변동률 기반 hot_score 계산
# 5. 상위 종목 반환
#
@app.get("/market/hot-stocks")
def market_hot_stocks():

    try:

        articles = fetch_market_news(display=100)

        if not articles:
            return {
                "generated_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "article_count": 0,
                "stocks": []
            }

        counter = Counter()

        # 기사별 종목 언급 탐색
        for article in articles:

            text = (
                article.get("title", "")
                + " "
                + article.get("description", "")
            )

            for company_name in NAME_TO_TICKER.keys():

                try:

                    if re.search(
                        rf"{re.escape(company_name)}",
                        text
                    ):
                        counter[company_name] += 1

                except Exception:
                    continue

        result = []

        # 상위 언급 종목 추출
        for company_name, mentions in counter.most_common(20):

            ticker = NAME_TO_TICKER.get(company_name)

            if not ticker:
                continue

            try:

                df = _fetch_price_df(
                    ticker,
                    "1m"
                )

                if len(df) >= 2:

                    prev_close = float(
                        df["Close"].iloc[-2]
                    )

                    latest_close = float(
                        df["Close"].iloc[-1]
                    )

                    change_pct = round(
                        (
                            latest_close
                            - prev_close
                        )
                        / prev_close
                        * 100,
                        2
                    )

                else:
                    change_pct = 0

            except Exception as e:

                logger.warning(
                    f"{ticker} 가격조회 실패: {e}"
                )

                change_pct = 0

            # 언급량 + 가격변동률 기반 점수
            hot_score = round(
                mentions
                + abs(change_pct) * 3,
                2
            )

            result.append({
                "ticker": ticker,
                "name": company_name,
                "mentions": mentions,
                "change_pct": change_pct,
                "hot_score": hot_score
            })

        # 최종 정렬
        result.sort(
            key=lambda x: x["hot_score"],
            reverse=True
        )

        return {
            "generated_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "article_count":
                len(articles),

            "tracked_companies":
                len(NAME_TO_TICKER),

            "stocks":
                result[:10]
        }

    except Exception as e:

        logger.exception(
            f"hot-stocks 생성 실패: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"hot-stocks 생성 실패: {str(e)}"
        )


# ---------------- AI 추천 관심 종목 ----------------

@app.get("/market/recommendations")
def market_recommendations():
    """
    AI 추천 관심 종목 API
    주요 종목의 시세 데이터 및 ML 예측 모델 / 기술적 지표(RSI, 이평선 등)를 종합 분석하여
    실시간 추천 관심 종목 목록과 사유를 반환합니다.
    """
    try:
        recommendations = []
        for ticker, name in DISPLAY_NAME.items():
            try:
                df = _fetch_price_df(ticker, "1m")
                if df is None or len(df) < 5:
                    continue

                last_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                pct_change = round(((last_close - prev_close) / prev_close) * 100, 2)

                # ML 모델 예측 실행
                pred_res = predict_price(df[["date", "Close", "Volume"]], ticker=ticker, horizon="1d")
                pred_change = pred_res.get("predicted_change_pct", 0.0)

                # 기술적 지표 계산 (RSI 14)
                deltas = df["Close"].diff()
                gains = deltas.clip(lower=0)
                losses = -1 * deltas.clip(upper=0)
                avg_gain = gains.tail(14).mean()
                avg_loss = losses.tail(14).mean()
                rs = avg_gain / avg_loss if (avg_loss and avg_loss > 0) else 1.0
                rsi = round(100 - (100 / (1 + rs)), 1)

                # 5일 / 20일 이동평균
                ma5 = df["Close"].tail(5).mean()
                ma20 = df["Close"].tail(20).mean() if len(df) >= 20 else ma5

                reason = None
                score = 0.0

                if pred_change >= 0.5:
                    conf = min(round(70 + abs(pred_change) * 3), 95)
                    reason = f"AI 모델 상승 예측 (+{pred_change:.2f}%, 신뢰도 {conf}%)"
                    score = pred_change * 3 + 10
                elif rsi <= 38:
                    reason = f"RSI 과매도 구간({rsi}) 반등 모멘텀 감지"
                    score = (40 - rsi) * 2 + 5
                elif ma5 > ma20 and last_close >= ma5:
                    reason = "단기 이동평균선(5일/20일) 골든크로스 상승 전환 국면"
                    score = 8.0 + (pct_change if pct_change > 0 else 0)
                elif pct_change >= 1.0:
                    reason = f"최근 수급 유입 및 주가 강세 흐름 (+{pct_change:.2f}%)"
                    score = pct_change * 2

                if reason:
                    c_sign = "+" if pct_change >= 0 else ""
                    recommendations.append({
                        "ticker": ticker,
                        "name": name,
                        "reason": reason,
                        "pct": f"{c_sign}{pct_change:.2f}%",
                        "predicted_pct": f"{'+' if pred_change >= 0 else ''}{pred_change:.2f}%",
                        "score": score
                    })
            except Exception as e:
                logger.warning(f"종목 추천 분석 중 오류 ({ticker}): {e}")
                continue

        # 점수 기준 내림차순 정렬 후 상위 5개 반환
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return {"recommendations": recommendations[:5]}
    except Exception as e:
        logger.exception(f"AI 추천 관심 종목 서비스 오류: {e}")
        return {"recommendations": []}


# ---------------- 이슈 종목 뉴스 ----------------

# ---------------- 이슈 종목 뉴스 ----------------

@app.get("/market/news/{ticker}")
def market_news(
    ticker: str,
    date: str | None = None
):
    clean_code = ticker.split(".")[0]
    ticker_with_ks = f"{clean_code}.KS" if not ticker.endswith(".KS") else ticker
    ticker_name = DISPLAY_NAME.get(ticker_with_ks, DISPLAY_NAME.get(ticker, clean_code))
    news_query = "네이버" if ticker in ["035420.KS", "035420"] else ticker_name

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        pipeline_res = process_news_3d_pipeline(news_query, center_date=date, window_days=3)

        return {
            "ticker": ticker,
            "name": ticker_name,
            "article_count": pipeline_res.get("total_articles", 0),
            "articles": pipeline_res.get("articles", []),
            "top_impact_articles": pipeline_res.get("top_impact_articles", []),
            "summary": pipeline_res.get("summary", ""),
            "news_sentiment": pipeline_res.get("weighted_sentiment", 0.0),
            "intent_percentages": pipeline_res.get("intent_percentages", {}),
            "overall_stance": pipeline_res.get("overall_stance", "Neutral"),
            "bullish_reasons": pipeline_res.get("bullish_reasons", []),
            "bearish_reasons": pipeline_res.get("bearish_reasons", []),
            "top_keywords": pipeline_res.get("top_keywords", []),
        }

    except Exception as e:
        logger.exception(f"{ticker} 뉴스조회 실패: {e}")
        return {
            "ticker": ticker,
            "name": ticker_name,
            "article_count": 0,
            "articles": [],
            "top_impact_articles": [],
            "summary": f"뉴스 분석 조회 중 안내: {e}",
            "news_sentiment": 0.0,
            "intent_percentages": {"Bullish": 0.0, "Bearish": 0.0, "Neutral": 100.0},
            "overall_stance": "Neutral",
            "bullish_reasons": [],
            "bearish_reasons": [],
            "top_keywords": []
        }


# ===== LLM 뉴스 감성 & 3D 영향도 분석 엔드포인트 =====
@app.get("/market/news_sentiment_analysis/{ticker}")
def news_sentiment_analysis(
    ticker: str,
    date: str | None = None
):
    """뉴스 관련도, 영향도, 감성 가중치 기반 LLM 3D 분석"""
    clean_code = ticker.split(".")[0]
    ticker_with_ks = f"{clean_code}.KS" if not ticker.endswith(".KS") else ticker
    ticker_name = DISPLAY_NAME.get(ticker_with_ks, DISPLAY_NAME.get(ticker, clean_code))
    news_query = "네이버" if ticker in ["035420.KS", "035420"] else ticker_name


    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        pipeline_res = process_news_3d_pipeline(news_query, center_date=date, window_days=3)

        intent_pcts = pipeline_res.get("intent_percentages", {})
        overall_stance = pipeline_res.get("overall_stance", "Neutral")
        summary_txt = pipeline_res.get("summary", "")
        b_reasons = pipeline_res.get("bullish_reasons", [])
        be_reasons = pipeline_res.get("bearish_reasons", [])
        keywords = pipeline_res.get("top_keywords", [])
        top_articles = pipeline_res.get("top_impact_articles", [])

        top_news_str = "\n".join([f" • [{a.get('sentiment','Neutral')}/점수:{a.get('weighted_score',0):+d}/관련도:{a.get('relevance_stars','')}/영향도:{a.get('impact_stars','')}] {a.get('title','')}" for a in top_articles[:3]])

        llm_analysis = (
            f"📰 [{ticker_name} 뉴스 영향도 & 감성 3D 종합 분석 리포트]\n\n"
            f"📌 **종합 평가**: {overall_stance}\n"
            f"📊 **가중 심리 비중**: Bullish(상승 모멘텀) {intent_pcts.get('Bullish',0)}% | Bearish(하락 리스크) {intent_pcts.get('Bearish',0)}% | Neutral(중립) {intent_pcts.get('Neutral',0)}%\n\n"
            f"💡 **AI 종합 요약**: {summary_txt}\n\n"
            f"🔥 **주가 영향력이 가장 큰 핵심 뉴스 Top 3**:\n{top_news_str}\n\n"
            f"📈 **상승 모멘텀 요인**:\n" + "\n".join([f" • {r}" for r in b_reasons]) + "\n\n"
            f"📉 **하락 리스크 요인**:\n" + "\n".join([f" • {r}" for r in be_reasons]) + "\n\n"
            f"🏷️ **주요 헤드라인 키워드**: {', '.join(keywords)}"
        )

        return {
            "ticker": ticker,
            "name": ticker_name,
            "date": date,
            "intent_percentages": intent_pcts,
            "overall_stance": overall_stance,
            "weighted_sentiment": pipeline_res.get("weighted_sentiment"),
            "top_impact_articles": top_articles,
            "llm_analysis": llm_analysis,
            "summary": summary_txt,
            "bullish_reasons": b_reasons,
            "bearish_reasons": be_reasons,
            "top_keywords": keywords,
        }

    except Exception as e:
        logger.exception(f"{ticker} 뉴스 감성 LLM 분석 실패: {e}")
        return {
            "ticker": ticker,
            "name": ticker_name,
            "date": date,
            "intent_percentages": {"Bullish": 0.0, "Bearish": 0.0, "Neutral": 100.0},
            "overall_stance": "Neutral",
            "weighted_sentiment": 0.0,
            "top_impact_articles": [],
            "llm_analysis": f"뉴스 분석 중 오류가 발생했습니다: {e}",
            "summary": "",
            "bullish_reasons": [],
            "bearish_reasons": [],
            "top_keywords": []
        }



from community_service import process_community_pipeline

# ===== LLM 커뮤니티 심리 분석 엔드포인트 =====
@app.get("/market/community_sentiment_analysis/{ticker}")
def community_sentiment_analysis(
    ticker: str,
    date: str | None = None
):
    """6가지 투자 의도 분류 및 hybrid LLM 심층 분석"""
    ticker_name = DISPLAY_NAME.get(ticker, ticker)

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        pipeline_res = process_community_pipeline(ticker, target_count=300)

        intent_pcts = pipeline_res.get("intent_percentages", {})
        overall_stance = pipeline_res.get("overall_stance", "Neutral")
        summary_txt = pipeline_res.get("summary", "")
        b_reasons = pipeline_res.get("bullish_reasons", [])
        be_reasons = pipeline_res.get("bearish_reasons", [])
        keywords = pipeline_res.get("top_keywords", [])

        llm_analysis = (
            f"💬 [{ticker_name} 커뮤니티 종합 투자 심리 분석 리포트]\n\n"
            f"📌 **종합 판단**: {overall_stance}\n"
            f"📊 **투자 의도 분포**: Bullish(매수) {intent_pcts.get('Bullish',0)}% | Bearish(매도) {intent_pcts.get('Bearish',0)}% | Neutral {intent_pcts.get('Neutral',0)}% | 질문 {intent_pcts.get('Question',0)}% | 유머 {intent_pcts.get('Humor',0)}% | 뉴스 {intent_pcts.get('News',0)}%\n\n"
            f"💡 **종합 요약**: {summary_txt}\n\n"
            f"📈 **상승 기대 근거 (Bullish)**:\n" + "\n".join([f" • {r}" for r in b_reasons]) + "\n\n"
            f"📉 **하락 우려 근거 (Bearish)**:\n" + "\n".join([f" • {r}" for r in be_reasons]) + "\n\n"
            f"🏷️ **대표 관련 키워드**: {', '.join(keywords)}"
        )

        return {
            "ticker": ticker,
            "name": ticker_name,
            "date": date,
            "intent_counts": pipeline_res.get("intent_counts"),
            "intent_percentages": intent_pcts,
            "overall_stance": overall_stance,
            "weighted_sentiment": pipeline_res.get("weighted_sentiment"),
            "llm_analysis": llm_analysis,
            "summary": summary_txt,
            "bullish_reasons": b_reasons,
            "bearish_reasons": be_reasons,
            "top_keywords": keywords,
        }

    except Exception as e:
        logger.exception(f"{ticker} 커뮤니티 심리 LLM 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

        
# ----------------UR-10~12: 미래 예측 + 시각화용 데이터 + 근거 설명 ----------------
@app.get("/predict/{ticker}")
def get_prediction(ticker: str, horizon: str = "1d", db: Session = Depends(get_db)):
    if horizon not in ("1d", "1w", "1m"):
        raise HTTPException(400, "horizon은 1d/1w/1m 중 하나여야 합니다")

    try:
        df = _fetch_price_df(ticker, "6m")
        result = predict_price(df[["date", "Close", "Volume"]], ticker=ticker, horizon=horizon)

        ticker_name = DISPLAY_NAME.get(ticker, ticker)
        explanation = explain_prediction(
            ticker_name,
            result.get("predicted_change_pct", 0.0),
            result.get("features_used", {}),
        )

        try:
            log = PredictionLog(
                ticker=ticker,
                horizon=horizon,
                base_date=result.get("base_date", ""),
                base_close=result.get("base_close", 0.0),
                predicted_close=result.get("predicted_close", 0.0),
                predicted_change_pct=result.get("predicted_change_pct", 0.0),
                reasoning_json=str(explanation.get("reasoning", [])),
            )
            db.add(log)
            db.commit()
        except Exception as db_err:
            logger.warning(f"DB prediction log write skipped ({db_err})")
            db.rollback()

        return {
            "ticker": ticker,
            "name": ticker_name,
            "horizon": horizon,
            **result,
            "explanation": explanation,
        }
    except Exception as e:
        logger.exception(f"Error in get_prediction for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"예측 분석 중 오류 발생: {e}")


class SimulationRequest(BaseModel):
    horizon: str = "1w"
    overrides: dict = {}


@app.post("/predict/{ticker}/simulate")
def simulate_prediction_endpoint(ticker: str, req: SimulationRequest):
    df = _fetch_price_df(ticker, "6m")
    if df is None or len(df) < 5:
        raise HTTPException(status_code=400, detail="시세 데이터가 부족합니다.")
    result = simulate_predict(df[["date", "Close", "Volume"]], ticker=ticker, overrides=req.overrides, horizon=req.horizon)
    return result


# ---------------- 과거 유사 패턴 탐색 ----------------

@app.get("/market/similar-patterns/{ticker}")
def get_similar_patterns(ticker: str, window_size: int = 14, predict_horizon: int = 14, top_k: int = 3):
    df = _fetch_price_df(ticker, "2y")
    if df is None or len(df) < window_size * 2:
        raise HTTPException(status_code=400, detail="과거 시세 데이터가 부족합니다.")
    res = find_similar_patterns(df[["date", "Close", "Volume"]], ticker=ticker, window_size=window_size, predict_horizon=predict_horizon, top_k=top_k)
    return res

# ---------------- 종합 AI 분석 ----------------

def get_latest_community_analysis(ticker: str) -> dict:
    try:
        pipeline_res = process_community_pipeline(ticker, target_count=300)
        pcts = pipeline_res.get("intent_percentages", {})
        return {
            "sentiment": pipeline_res.get("weighted_sentiment", 0.0),
            "positive_pct": pcts.get("Bullish", 0.0),
            "neutral_pct": pcts.get("Neutral", 0.0),
            "negative_pct": pcts.get("Bearish", 0.0),
            "intent_percentages": pcts,
            "overall_stance": pipeline_res.get("overall_stance", "Neutral")
        }
    except Exception as e:
        logger.warning(f"get_latest_community_analysis error: {e}")
        return {
            "sentiment": 0.0,
            "positive_pct": 0.0,
            "neutral_pct": 100.0,
            "negative_pct": 0.0,
            "intent_percentages": {"Bullish": 0, "Bearish": 0, "Neutral": 100, "Question": 0, "Humor": 0, "News": 0},
            "overall_stance": "Neutral"
        }


_ANALYZE_CACHE = {}
_ANALYZE_CACHE_TTL = 1800  # 30분 캐시


@app.get("/analyze/{ticker}")
def analyze_stock_endpoint(
    ticker: str,
    horizon: str = "1w"
):
    """
    UR-10 예측 결과와 UR-07~09 뉴스/커뮤니티 감성 결과를 결합하여
    종합 AI 의견 및 4단계 CoT 분석 보고서를 리턴하는 백엔드 핵심 엔드포인트
    """
    cache_key = f"{ticker}_{horizon}"
    now = time.time()
    if cache_key in _ANALYZE_CACHE:
        entry = _ANALYZE_CACHE[cache_key]
        if now - entry["time"] < _ANALYZE_CACHE_TTL:
            return entry["data"]

    # 가격 데이터
    df = _fetch_price_df(ticker, "6m")

    # 예측 결과 재사용
    prediction = predict_price(
        df[["date", "Close", "Volume"]],
        ticker=ticker,
        horizon=horizon
    )

    # 감성 점수 추출
    latest_values = (
        prediction
        .get("features_used", {})
        .get("latest_values", {})
    )

    news_sentiment = latest_values.get(
        "news_sentiment",
        0.0
    )

    comm_info = get_latest_community_analysis(ticker)
    community_sentiment = comm_info["sentiment"]

    # 뉴스 수집
    ticker_name = DISPLAY_NAME.get(
        ticker,
        ticker
    )
    
    # Use Korean name for NAVER to prevent generic English tag pollution
    news_query = "네이버" if ticker == "035420.KS" else ticker_name

    articles = fetch_news_window(
        query=news_query,
        center_date=datetime.now().strftime(
            "%Y-%m-%d"
        ),
        window_days=3
    )
    
    # Fallback to market news if company news is empty
    if not articles:
        articles = fetch_market_news()

    sentiment_info = summarize_article_sentiment(
        articles,
        datetime.now()
    )

    news_summary = summarize_news(
        ticker_name,
        articles
    )

    # AI 분석 (동적 신뢰도 가중치 종합 판정 엔진 연동)
    pattern_ret = None
    try:
        similar_patterns = find_similar_patterns(ticker)
        if isinstance(similar_patterns, list) and len(similar_patterns) > 0:
            pattern_ret = float(np.mean([p.get("future_return", 0) for p in similar_patterns]))
    except Exception:
        pattern_ret = None


    analysis = analyze_stock(
        ticker=ticker,
        prediction=prediction,
        news_summary=news_summary,
        news_sentiment=sentiment_info["news_sentiment"],
        community_sentiment=community_sentiment,
        sentiment_counts={
            "positive": sentiment_info["positive_count"],
            "neutral": sentiment_info["neutral_count"],
            "negative": sentiment_info["negative_count"],
            "weighted_sentiment": sentiment_info["weighted_sentiment"],
            "total_weight": sentiment_info["total_weight"],
        },
        community_post_count=comm_info.get("total_posts", 60),
        pattern_avg_return=pattern_ret,
        generate_report=False
    )


    # 2-step model analysis construction
    news_analysis = {
        "positive": sentiment_info["positive_count"],
        "neutral": sentiment_info["neutral_count"],
        "negative": sentiment_info["negative_count"]
    }
    
    community_analysis = {
        "positive": comm_info["positive_pct"],
        "neutral": comm_info["neutral_pct"],
        "negative": comm_info["negative_pct"]
    }

    # Generate reasons list dynamically
    reasons = []
    latest_features = prediction.get("features_used", {}).get("latest_values", {})
    
    rsi = latest_features.get("rsi14", 50)
    if rsi < 30:
        reasons.append("RSI 과매도 구간 (강한 매수 신호)")
    elif rsi > 70:
        reasons.append("RSI 과매수 구간 (단기 매도 신호)")
        
    macd_hist = latest_features.get("macd_hist", 0.0)
    if macd_hist > 0:
        reasons.append("MACD 골든크로스 발생 (상승 전환)")
    else:
        reasons.append("MACD 데드크로스 발생 (하락 전환)")
        
    news_sent = latest_features.get("news_sentiment", 0.0)
    if news_sent > 0.1:
        reasons.append("최근 뉴스 감성 긍정적 흐름 우세")
    elif news_sent < -0.1:
        reasons.append("최근 뉴스 감성 부정적 기사 우세")
        
    comm_sent = latest_features.get("community_sentiment", 0.0)
    if comm_sent > 0.1:
        reasons.append("커뮤니티 투자 심리 대폭 개선")
    elif comm_sent < -0.1:
        reasons.append("커뮤니티 내 부정 여론 확대")
        
    if not reasons:
        reasons = ["기술 지표 및 시장 감성 중립 상태"]

    direction_korean = {"UP": "상승", "DOWN": "하락", "FLAT": "횡보"}
    summary_text = f"1주일 {direction_korean.get(prediction.get('direction', 'FLAT'), '횡보')} 가능성 높음"

    res = {
        "ticker": ticker,
        "name": ticker_name,
        "summary": summary_text,
        "news_analysis": news_analysis,
        "community_analysis": community_analysis,
        "reasons": reasons,
        "prediction": {
            "predicted_close": prediction["predicted_close"],
            "predicted_change_pct": prediction["predicted_change_pct"],
            "direction": prediction.get("direction", "FLAT"),
            "direction_confidence": prediction.get("direction_confidence", 50.0),
            "expected_return_pct": prediction.get("expected_return_pct", 0.0),
            "expected_price": prediction.get("expected_price", prediction["predicted_close"])
        },
        "news_sentiment": sentiment_info["news_sentiment"],
        "news_sentiment_counts": {
            "positive": sentiment_info["positive_count"],
            "neutral": sentiment_info["neutral_count"],
            "negative": sentiment_info["negative_count"],
            "weighted_sentiment": sentiment_info["weighted_sentiment"],
            "total_weight": sentiment_info["total_weight"],
        },
        "community_sentiment": community_sentiment,
        "news_summary": news_summary,
        "analysis": analysis
    }
    _ANALYZE_CACHE[cache_key] = {"data": res, "time": now}
    return res


@app.get("/analyze/report/{ticker}")
def analyze_report_ticker(
    ticker: str,
    horizon: str = "1w"
):
    df = _fetch_price_df(ticker, "6m")

    prediction = predict_price(
        df[["date", "Close", "Volume"]],
        ticker=ticker,
        horizon=horizon
    )

    latest_values = (
        prediction
        .get("features_used", {})
        .get("latest_values", {})
    )

    news_sentiment = latest_values.get("news_sentiment", 0.0)
    community_sentiment = get_latest_community_analysis(ticker)["sentiment"]

    ticker_name = DISPLAY_NAME.get(ticker, ticker)
    news_query = "네이버" if ticker == "035420.KS" else ticker_name
    current_date = datetime.now()
    articles = fetch_news_window(
        query=news_query,
        center_date=current_date.strftime("%Y-%m-%d"),
        window_days=3
    )
    if not articles:
        articles = fetch_market_news()
    sentiment_info = summarize_article_sentiment(articles, current_date)
    news_summary = summarize_news(ticker_name, articles)

    analysis = analyze_stock(
        ticker=ticker,
        prediction=prediction,
        news_summary=news_summary,
        news_sentiment=sentiment_info["news_sentiment"],
        community_sentiment=community_sentiment,
        sentiment_counts={
            "positive": sentiment_info["positive_count"],
            "neutral": sentiment_info["neutral_count"],
            "negative": sentiment_info["negative_count"],
            "weighted_sentiment": sentiment_info["weighted_sentiment"],
            "total_weight": sentiment_info["total_weight"],
        },
        generate_report=True
    )

    return {
        "ticker": ticker,
        "name": ticker_name,
        "analysis": analysis
    }

@app.get("/market/community/{ticker}")
def get_community_posts(ticker: str):
    """
    사전 + LLM 혼합 분류 및 6가지 투자 의도 분석 커뮤니티 API
    """
    try:
        return process_community_pipeline(ticker, target_count=300)
    except Exception as e:
        logger.warning(f"Failed to process community pipeline: {e}")
        return {
            "posts": [],
            "summary": f"커뮤니티 요약 처리 실패: {e}",
            "intent_percentages": {"Bullish": 0, "Bearish": 0, "Neutral": 100, "Question": 0, "Humor": 0, "News": 0},
            "overall_stance": "Neutral",
            "weighted_sentiment": 0.0
        }


@app.get("/health")
def health():
    return {"status": "ok"}
