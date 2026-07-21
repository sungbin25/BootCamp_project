"""
과거 유사 패턴 탐색 모듈 (Historical Pattern Matching Engine)

현재 종목의 최근 N일간 지표/수익률 패턴과 가장 코사인 유사도가 높은 과거 N일 구간 TOP K를 탐색하고,
해당 과거 구간 직후 M일 동안 실제 주가가 어떻게 움직였는지(Track Record) 산출합니다.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_patterns(df: pd.DataFrame, ticker: str, window_size: int = 14, predict_horizon: int = 14, top_k: int = 3) -> dict:
    """
    df: columns = [date, Close, Volume, (optional features)]
    window_size: 비교할 패턴 일수 (기본 14일)
    predict_horizon: 패턴 이후 관찰할 미래 일수 (기본 14일)
    top_k: 상위 유사 구간 수 (기본 3개)
    """
    if df is None or len(df) < window_size * 2 + predict_horizon:
        return {
            "ticker": ticker,
            "match_count": 0,
            "summary": "과거 시세 데이터가 부족하여 유사 패턴을 탐색할 수 없습니다.",
            "current_window": {},
            "matches": []
        }

    df_calc = df.copy().sort_values("date").reset_index(drop=True)

    # 1. 지표/수익률 계산
    df_calc["return_1d"] = df_calc["Close"].pct_change().fillna(0.0)
    
    # RSI14 계산
    delta = df_calc["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df_calc["rsi14"] = (rsi.fillna(50) - 50) / 50.0  # [-1, 1] 범위로 정규화

    # MACD Hist 계산 및 정규화
    ema12 = df_calc["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_calc["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    hist_std = macd_hist.std() or 1.0
    df_calc["macd_norm"] = (macd_hist / hist_std).clip(-2.0, 2.0) / 2.0

    # Volume Z-score 계산 및 정규화
    vol_mean = df_calc["Volume"].rolling(20).mean()
    vol_std = df_calc["Volume"].rolling(20).std().replace(0, np.nan)
    vol_z = (df_calc["Volume"] - vol_mean) / vol_std
    df_calc["vol_z"] = vol_z.fillna(0.0).clip(-3.0, 3.0) / 3.0

    # 2. 피처 행렬 조합 [return_1d, rsi14, macd_norm, vol_z]
    feature_matrix = df_calc[["return_1d", "rsi14", "macd_norm", "vol_z"]].values

    # 현재 윈도우 피처 벡터 (Flatten)
    current_vec = feature_matrix[-window_size:].flatten().reshape(1, -1)

    # 현재 14일간 정규화 주가 궤적 (첫날 = 100 기준)
    curr_closes = df_calc["Close"].tail(window_size).values
    curr_norm_prices = [round((float(p) / float(curr_closes[0])) * 100.0, 2) for p in curr_closes]
    curr_dates = [str(d)[:10] for d in df_calc["date"].tail(window_size)]

    current_window_info = {
        "dates": curr_dates,
        "prices": curr_norm_prices,
        "raw_prices": [round(float(p), 2) for p in curr_closes]
    }

    # 3. 과거 전체 윈도우 스캐닝
    matches = []
    n_total = len(df_calc)
    
    max_search_idx = n_total - window_size - predict_horizon - 5

    for idx in range(20, max_search_idx):
        past_vec = feature_matrix[idx : idx + window_size].flatten().reshape(1, -1)
        sim_score = float(cosine_similarity(current_vec, past_vec)[0][0])
        
        # 코사인 유사도 0.65 이상인 구간 수집
        if sim_score >= 0.65:
            past_window_closes = df_calc["Close"].iloc[idx : idx + window_size].values
            base_p = float(past_window_closes[0]) or 1.0
            past_norm_prices = [round((float(p) / base_p) * 100.0, 2) for p in past_window_closes]
            
            # 패턴 이후 predict_horizon(14일) 동안의 실제 궤적
            future_window_closes = df_calc["Close"].iloc[idx + window_size : idx + window_size + predict_horizon].values
            future_norm_prices = [round((float(p) / base_p) * 100.0, 2) for p in future_window_closes]
            
            start_date = str(df_calc["date"].iloc[idx])[:10]
            end_date = str(df_calc["date"].iloc[idx + window_size - 1])[:10]
            
            # 14일 후 실제 변동률
            last_future_price = float(future_window_closes[-1]) if len(future_window_closes) > 0 else float(past_window_closes[-1])
            actual_return_pct = round(((last_future_price - float(past_window_closes[-1])) / float(past_window_closes[-1])) * 100.0, 2)
            
            matches.append({
                "start_date": start_date,
                "end_date": end_date,
                "similarity_pct": round(sim_score * 100.0, 1),
                "actual_return_14d": actual_return_pct,
                "matched_prices": past_norm_prices,
                "future_prices": future_norm_prices,
                "matched_dates": [str(d)[:10] for d in df_calc["date"].iloc[idx : idx + window_size]],
                "future_dates": [str(d)[:10] for d in df_calc["date"].iloc[idx + window_size : idx + window_size + predict_horizon]],
                "is_positive": bool(actual_return_pct > 0)
            })

    # 유사도 높은 순으로 정렬 후 겹치는 날짜 중복 제거
    matches.sort(key=lambda x: x["similarity_pct"], reverse=True)
    
    filtered_matches = []
    seen_dates = set()
    for m in matches:
        dt = m["start_date"]
        if not any(abs((pd.to_datetime(dt) - pd.to_datetime(s)).days) < 10 for s in seen_dates):
            seen_dates.add(dt)
            filtered_matches.append(m)
            if len(filtered_matches) >= top_k:
                break

    # 통계 요약 작성
    if filtered_matches:
        pos_count = sum(1 for m in filtered_matches if m["is_positive"])
        pos_ratio = round((pos_count / len(filtered_matches)) * 100.0, 1)
        avg_return = round(float(np.mean([m["actual_return_14d"] for m in filtered_matches])), 2)
        summary = f"현재 기술적 지표 패턴과 가장 유사한 과거 {len(filtered_matches)}개 구간을 분석한 결과, {pos_count}개 구간({pos_ratio}%)에서 이후 2주 내 평균 {avg_return:+.2f}%의 추세가 나타났습니다."
    else:
        summary = "현재 패턴과 65% 이상 일치하는 과거 구간을 찾지 못했습니다."

    return {
        "ticker": ticker,
        "match_count": len(filtered_matches),
        "summary": summary,
        "current_window": current_window_info,
        "matches": filtered_matches
    }
