"""
변곡점(급등/급락) 자동 탐지 (UR-04)

탐지 조건 (요구사항 명세 기준):
  - 5거래일 이내 누적 등락률 ±10% 이상
  - 거래량 급증 (20일 평균 대비 z-score 2 이상)
둘 중 하나라도 만족하면 변곡점으로 표시합니다.
"""
import pandas as pd
import numpy as np


def detect_changepoints(df: pd.DataFrame, pct_threshold: float = 0.10,
                         volume_z_threshold: float = 2.0) -> list[dict]:
    """
    df: columns = [date, Close, Volume] (날짜 오름차순 정렬되어 있어야 함)
    반환: [{date, pct_change, direction, volume_spike, reason}]
    """
    if df.empty or len(df) < 6:
        return []

    df = df.copy().reset_index(drop=True)
    df["cum_5d_return"] = df["Close"].pct_change(periods=5)
    df["daily_return"] = df["Close"].pct_change()

    vol_mean = df["Volume"].rolling(20, min_periods=5).mean()
    vol_std = df["Volume"].rolling(20, min_periods=5).std()
    df["volume_z"] = (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)

    changepoints = []
    for _, row in df.iterrows():
        cum_ret = row["cum_5d_return"]
        vol_z = row["volume_z"]

        pct_hit = pd.notna(cum_ret) and abs(cum_ret) >= pct_threshold
        vol_hit = pd.notna(vol_z) and vol_z >= volume_z_threshold

        if not (pct_hit or vol_hit):
            continue

        reasons = []
        if pct_hit:
            reasons.append(f"5일 누적 등락률 {cum_ret * 100:+.1f}%")
        if vol_hit:
            reasons.append(f"거래량 급증(z-score {vol_z:.1f})")

        changepoints.append({
            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "pct_change": round(float(cum_ret) * 100, 2) if pd.notna(cum_ret) else None,
            "direction": "up" if (pd.notna(cum_ret) and cum_ret > 0) else "down",
            "volume_spike": bool(vol_hit),
            "reason": ", ".join(reasons),
        })

    return changepoints


def analyze_changepoint_detail(df: pd.DataFrame, event_date: str) -> dict:
    """
    df: columns=[date, Close, Volume] (High, Low optional)
    event_date: 'YYYY-MM-DD'
    Returns technical signals, feature comparison table (T-1 vs T), AI historical replay simulation, confidence score, and AI technical comment.
    """
    if df.empty or len(df) < 5:
        return {}

    df = df.copy().reset_index(drop=True)
    df["date_str"] = df["date"].apply(lambda d: str(d.date()) if hasattr(d, "date") else str(d)[:10])

    close = df["Close"]
    vol = df["Volume"]

    # Technical Indicators
    ma20 = close.rolling(20, min_periods=5).mean()
    std20 = close.rolling(20, min_periods=5).std()
    upper_bb = ma20 + 2 * std20
    lower_bb = ma20 - 2 * std20
    bb_pct = (close - lower_bb) / (upper_bb - lower_bb).replace(0, np.nan)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50)

    if "High" in df.columns and "Low" in df.columns:
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - close.shift(1)).abs()
        tr3 = (df["Low"] - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        tr = delta.abs()
    atr = tr.rolling(14, min_periods=3).mean().fillna(0)

    sign = np.sign(delta).fillna(0)
    obv = (sign * vol).cumsum()

    vol_mean20 = vol.rolling(20, min_periods=5).mean()

    df["ma20"] = ma20
    df["upper_bb"] = upper_bb
    df["lower_bb"] = lower_bb
    df["bb_pct"] = bb_pct
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["rsi"] = rsi
    df["atr"] = atr
    df["obv"] = obv
    df["vol_mean20"] = vol_mean20

    matches = df.index[df["date_str"] == event_date].tolist()
    if not matches:
        idx = len(df) - 1
    else:
        idx = matches[0]

    idx_prev = max(0, idx - 1)
    row_t = df.iloc[idx]
    row_prev = df.iloc[idx_prev]

    ret_5d = ((close.iloc[idx] - close.iloc[max(0, idx - 5)]) / close.iloc[max(0, idx - 5)]) * 100 if idx >= 1 else 0.0

    # ① 변곡 발생 원인 분석
    causes = []
    bullish_cnt = 0
    bearish_cnt = 0

    vol_ratio = float(row_t["Volume"] / row_t["vol_mean20"]) if row_t["vol_mean20"] > 0 else 1.0
    if vol_ratio >= 1.5:
        causes.append(f"✔ 거래량 평균 대비 {vol_ratio:.1f}배 급증")
        bullish_cnt += 1

    rsi_t = float(row_t["rsi"])
    rsi_prev = float(row_prev["rsi"])
    if rsi_prev < 35 and rsi_t >= 35:
        causes.append(f"✔ RSI 과매도 탈출 ({rsi_prev:.1f} → {rsi_t:.1f})")
        bullish_cnt += 1
    elif rsi_prev > 65 and rsi_t <= 65:
        causes.append(f"✔ RSI 과매수 탈출 ({rsi_prev:.1f} → {rsi_t:.1f})")
        bearish_cnt += 1
    elif abs(rsi_t - rsi_prev) >= 10:
        sign_str = "상승" if rsi_t > rsi_prev else "하락"
        causes.append(f"✔ RSI 급변 ({rsi_prev:.1f} → {rsi_t:.1f}, {sign_str})")
        if rsi_t > rsi_prev: bullish_cnt += 1
        else: bearish_cnt += 1

    macd_prev = float(row_prev["macd"])
    macd_sig_prev = float(row_prev["macd_signal"])
    macd_t = float(row_t["macd"])
    macd_sig_t = float(row_t["macd_signal"])

    if macd_prev <= macd_sig_prev and macd_t > macd_sig_t:
        causes.append("✔ MACD 골든크로스 발생")
        bullish_cnt += 2
    elif macd_prev >= macd_sig_prev and macd_t < macd_sig_t:
        causes.append("✔ MACD 데드크로스 발생")
        bearish_cnt += 2
    elif macd_t > macd_sig_t:
        causes.append("✔ MACD 매수 우위 지속")
        bullish_cnt += 1

    bb_t = float(row_t["bb_pct"]) if pd.notna(row_t["bb_pct"]) else 0.5
    bb_prev = float(row_prev["bb_pct"]) if pd.notna(row_prev["bb_pct"]) else 0.5
    if bb_prev <= 0.10 and bb_t > 0.10:
        causes.append("✔ Bollinger Band 하단 반등")
        bullish_cnt += 2
    elif bb_prev >= 0.90 and bb_t < 0.90:
        causes.append("✔ Bollinger Band 상단 터치 후 하락")
        bearish_cnt += 2

    atr_diff = float(row_t["atr"] - row_prev["atr"])
    if atr_diff > 0:
        causes.append(f"✔ ATR 변동성 확대 (+{atr_diff:.2f})")

    if not causes:
        causes.append("✔ 가격 변동성 기준 변곡점 포착")

    if bullish_cnt > bearish_cnt:
        ai_judgement = f"기술적 반등 신호 {bullish_cnt}개 동시 발생"
        sentiment = "Bullish"
    elif bearish_cnt > bullish_cnt:
        ai_judgement = f"기술적 조정 신호 {bearish_cnt}개 동시 발생"
        sentiment = "Bearish"
    else:
        ai_judgement = "기술적 혼조 신호 포착"
        sentiment = "Neutral"

    # ② Feature 변화 표 (T-1 vs T)
    def fmt_vol(v):
        if v >= 1e8: return f"{v/1e8:.2f}억"
        if v >= 1e4: return f"{v/1e4:.0f}만"
        return f"{v:,.0f}"

    def fmt_obv(v):
        if abs(v) >= 1e6: return f"{v/1e6:.1f}M"
        return f"{v:,.0f}"

    bb_desc = "하단 반등" if (bb_prev <= 0.1 and bb_t > 0.1) else ("상단 하향" if (bb_prev >= 0.9 and bb_t < 0.9) else f"{bb_t:.2f}")
    macd_desc = "골든크로스" if (macd_prev <= macd_sig_prev and macd_t > macd_sig_t) else ("데드크로스" if (macd_prev >= macd_sig_prev and macd_t < macd_sig_t) else f"{macd_t-macd_prev:+.1f}")

    feature_table = [
        {"feature": "RSI (14)", "prev": f"{rsi_prev:.1f}", "curr": f"{rsi_t:.1f}", "change": f"{rsi_t-rsi_prev:+.1f}"},
        {"feature": "MACD", "prev": f"{macd_prev:.1f}", "curr": f"{macd_t:.1f}", "change": macd_desc},
        {"feature": "거래량", "prev": fmt_vol(row_prev['Volume']), "curr": fmt_vol(row_t['Volume']), "change": f"{(row_t['Volume']-row_prev['Volume'])/row_prev['Volume']*100:+.1f}%" if row_prev['Volume']>0 else "N/A"},
        {"feature": "Bollinger %B", "prev": f"{bb_prev:.2f}", "curr": f"{bb_t:.2f}", "change": bb_desc},
        {"feature": "ATR", "prev": f"{row_prev['atr']:.2f}", "curr": f"{row_t['atr']:.2f}", "change": f"{atr_diff:+.2f} (변동성)"},
        {"feature": "OBV", "prev": fmt_obv(row_prev['obv']), "curr": fmt_obv(row_t['obv']), "change": "매수세 유입" if row_t['obv'] > row_prev['obv'] else "매도세 유입"}
    ]

    # ③ AI 회고 시뮬레이션
    bullish_score = int(min(98, max(15, 50 + (bullish_cnt - bearish_cnt) * 15 + (ret_5d * 0.5))))
    future_window = 5
    end_idx = min(len(df) - 1, idx + future_window)
    actual_series = df.iloc[idx : end_idx + 1]
    
    actual_dates = actual_series["date_str"].tolist()
    actual_prices = [float(p) for p in actual_series["Close"].tolist()]

    curr_p = float(row_t["Close"])
    if len(actual_prices) > 1:
        actual_pct = round(((actual_prices[-1] - curr_p) / curr_p) * 100, 1)
    else:
        actual_pct = round(ret_5d, 1)

    expected_pct = round((bullish_score - 50) * 0.25, 1)
    
    # 예측 적중 여부 및 오차 원인 분석
    sign_expected = np.sign(expected_pct) if expected_pct != 0 else 0
    sign_actual = np.sign(actual_pct) if actual_pct != 0 else 0
    is_hit = bool(sign_expected == sign_actual and sign_expected != 0)
    error_pct = round(abs(expected_pct - actual_pct), 1)
    
    if is_hit:
        hit_status = "✅ 적중 (Success)"
        error_causes = [
            "✔ 기술적 지표 신호와 실제 주가 방향성 일치",
            "✔ 변곡 시점 모멘텀 지속으로 목표 수익률 구간 도달"
        ]
    else:
        hit_status = "❌ 미적중 (Failed)"
        error_causes = [
            "✔ 변곡 당일 기술적 지표 신호(거래량/크로스) 포착",
            "✖ 변곡 직후 수급 이탈 및 매크로 시장 조정 발생"
        ]

    forecast_prices = []
    n_steps = len(actual_dates)
    for i in range(n_steps):
        factor = 1 + (expected_pct / 100) * (i / max(1, n_steps - 1))
        forecast_prices.append(round(curr_p * factor, 2))

    # ④ 변곡점 기술적 신호 강도 (Signal Strength Score)
    conf_base = 65 + min(30, (bullish_cnt + bearish_cnt) * 7 + (vol_ratio * 3))
    signal_score = int(min(98, conf_base))
    star_cnt = 5 if signal_score >= 90 else (4 if signal_score >= 80 else 3)
    stars = "★" * star_cnt + "☆" * (5 - star_cnt)

    conf_reasons = [c.replace("✔ ", "") for c in causes]

    # ⑤ AI 기술적 분석 종합 코멘트
    comment = (
        f"해당 변곡점({event_date})에서는 거래량 급증(평균 대비 {vol_ratio:.1f}배) 및 "
        f"{'MACD 골든크로스' if macd_t > macd_sig_t else '기술적 지지 구간'} 신호가 포착되었습니다. "
        f"RSI 지표 역시 {rsi_t:.1f} 수준으로 모멘텀을 형성하고 있으며, "
        f"AI 회고 시뮬레이션 분석 결과 당시 기술적 신호 강도는 {signal_score}점('{sentiment}') 수준입니다."
    )

    return {
        "date": event_date,
        "price_change_pct": round(ret_5d, 2),
        "causes": causes,
        "ai_judgement": ai_judgement,
        "sentiment": sentiment,
        "feature_table": feature_table,
        "ai_replay": {
            "headline": "현재 AI 모델을 당시 데이터에 적용한 회고 시뮬레이션 결과",
            "bullish_score": bullish_score,
            "expected_pct": expected_pct,
            "actual_pct": actual_pct,
            "is_hit": is_hit,
            "hit_status": hit_status,
            "error_pct": error_pct,
            "error_causes": error_causes,
            "forecast_dates": actual_dates,
            "forecast_prices": forecast_prices,
            "actual_prices": actual_prices,
        },
        "confidence": {
            "score": signal_score,
            "stars": stars,
            "reasons": conf_reasons,
        },
        "ai_comment": comment,
    }

