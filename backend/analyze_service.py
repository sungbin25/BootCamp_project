import os
import requests

def _get_ollama_generate_endpoint():
    hosts = [
        os.environ.get("OLLAMA_HOST", "http://ollama:11434"),
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    ]
    for h in hosts:
        try:
            r = requests.get(f"{h}/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                if models:
                    pref_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
                    model_name = pref_model if pref_model in models else models[0]
                    return f"{h}/api/generate", model_name
                return f"{h}/api/generate", os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
        except Exception:
            continue
    return "http://localhost:11434/api/generate", os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")


def ask_llm(prompt: str):
    try:
        url, model_name = _get_ollama_generate_endpoint()
        r = requests.post(
            url,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        if r.status_code == 200:
            res_json = r.json()
            if "response" in res_json and res_json["response"]:
                return res_json["response"]
            elif "error" in res_json:
                print(f"[WARN] Ollama returned error: {res_json['error']}")
    except Exception as e:
        print(f"[WARN] Ollama request failed: {e}")

    return None


def _generate_rule_based_cot_report(outlook: str, normalized_score: int, predicted_change: float,
                                      reasons: list, news_sentiment: float, sentiment_counts: dict,
                                      community_sentiment: float, rsi: float, macd: float,
                                      macd_signal: float, bb_position: float, news_summary: str) -> str:
    direction_str = "상승" if predicted_change >= 0 else "하락"
    rsi_state = "과매수 영역" if rsi >= 70 else ("과매도 영역" if rsi <= 35 else "중립 지대")
    macd_state = "골든크로스(상승 전환)" if macd >= macd_signal else "데드크로스(하락 전환)"
    sentiment_str = "긍정적" if news_sentiment > 0.05 else ("부정적" if news_sentiment < -0.05 else "중립적")

    reason1 = reasons[0] if (reasons and len(reasons) > 0) else "기술적 모멘텀 및 수급 변동성 존재"
    reason2 = reasons[1] if (reasons and len(reasons) > 1) else f"RSI 지표 ({rsi:.1f}) 수준에 따른 추세 조정"

    return f"""## 1단계: 정량 지표 종합 데이터 요약
- **종합 판정**: {outlook} (AI 종합 점수: {normalized_score}/100점)
- **ML 모델 예측**: 1주일 후 **{predicted_change:+.2f}% {direction_str}** 추정
- **기술적 지표**: RSI(14) {rsi:.1f} ({rsi_state}), MACD {macd_state} (MACD: {macd:.4f}, Signal: {macd_signal:.4f}), 볼린저 위치: {bb_position:.2f}
- **감성 지표**: 뉴스 감성 {news_sentiment:+.2f} ({sentiment_str}), 커뮤니티 심리 {community_sentiment:+.2f}

## 2단계: 상승 모멘텀 vs 하락 리스크 핵심 대조
- **상승 모멘텀 요소**:
  • {reason1}
  • RSI {rsi:.1f} 지점에 따른 가격 조정 후 기술적 반등 유입 가능성
- **하락/조정 리스크 요소**:
  • MACD 지표 {macd_state}에 따른 변동성 위험
  • 뉴스 감성 지표({news_sentiment:+.2f}) 및 시장 수급 변동 리스크

## 3단계: 신호 충돌 및 세부 분석 (ML vs 기술적 지표)
- 머신러닝(ML) 예측 모델은 1주일 후 **{predicted_change:+.2f}%**의 {direction_str} 추세를 가리키고 있습니다.
- 반면, 단기 기술적 지표(MACD {macd_state}, RSI {rsi:.1f}) 및 뉴스 감성 수급은 {sentiment_str} 상태로, ML 모델 예측과의 **신호 강도 차이**가 관찰됩니다.
- 단기 수급 변화와 변곡점 뉴스 발생 여부를 지속적으로 관찰할 필요가 있습니다.

## 4단계: 최종 AI 투자 전망 & 리스크 대응 전략
- **투자의견**: **{outlook} 관망 및 분할 접근**
- **대응 전략**: 단기 지표({macd_state})의 안정적 회복을 확인한 후 접근하는 것을 권장하며, 목표가 및 손절가를 철저히 준수하는 위험 관리가 필요합니다.

*(※ 본 리포트는 AI 통계 모델 및 4단계 Chain-of-Thought 알고리즘에 의해 생성된 투자 참고용 자료입니다.)*"""


def analyze_stock(
    ticker: str,
    prediction: dict,
    news_summary: str,
    news_sentiment: float,
    community_sentiment: float,
    sentiment_counts: dict | None = None,
    community_post_count: int = 60,
    pattern_avg_return: float | None = None,
    generate_report: bool = True
):
    latest = prediction.get("features_used", {}).get("latest_values", {})
    predicted_change = float(prediction.get("predicted_change_pct", 0))
    rsi = float(latest.get("rsi14", 50))
    macd = float(latest.get("macd", 0))
    macd_signal = float(latest.get("macd_signal", 0))
    bb_position = float(latest.get("bb_position", 0.5))

    reasons = []

    model_info = prediction.get("model_info", {})
    r2_score = float(model_info.get("r2", 0.0))

    # -------------------------------------------------------------
    # 1단계. 동적 가중치 산출 (신뢰도 기반)
    # -------------------------------------------------------------
    if r2_score >= 0.50:
        w_ml = 0.35
    elif r2_score >= 0.30:
        w_ml = 0.30
    elif r2_score >= 0.10:
        w_ml = 0.20
    elif r2_score >= 0.0:
        w_ml = 0.10
    else:  # R² < 0.0 (낮은 신뢰도)
        w_ml = 0.05
        reasons.append(f"⚠️ ML 모델 R² 성능 저조(R²={r2_score:+.4f})로 ML 반영 비중을 5%로 축소 조정했습니다.")

    extra = 0.35 - w_ml
    w_news = 0.25 + (extra * 0.50)
    w_tech = 0.20 + (extra * 0.30)
    w_comm = 0.10 + (extra * 0.20)
    w_pattern = 0.10

    # -------------------------------------------------------------
    # 2단계. 각 요소별 개별 정규화 점수 (-1.0 ~ +1.0)
    # -------------------------------------------------------------
    # (1) ML 예측 점수
    if predicted_change >= 10.0:
        s_ml = 1.0
        reasons.append(f"ML 모델이 1주일 후 강력한 상승(+{predicted_change:.2f}%)을 예측합니다.")
    elif predicted_change >= 5.0:
        s_ml = 0.75
        reasons.append(f"ML 모델이 1주일 후 상승(+{predicted_change:.2f}%)을 예측합니다.")
    elif predicted_change >= 1.0:
        s_ml = 0.40
        reasons.append(f"ML 모델이 1주일 후 소폭 상승(+{predicted_change:.2f}%)을 예측합니다.")
    elif predicted_change <= -10.0:
        s_ml = -1.0
        reasons.append(f"ML 모델이 1주일 후 강력한 하락({predicted_change:.2f}%)을 예측합니다.")
    elif predicted_change <= -5.0:
        s_ml = -0.75
        reasons.append(f"ML 모델이 1주일 후 하락({predicted_change:.2f}%)을 예측합니다.")
    elif predicted_change <= -1.0:
        s_ml = -0.40
        reasons.append(f"ML 모델이 1주일 후 소폭 하락({predicted_change:.2f}%)을 예측합니다.")
    else:
        s_ml = 0.0

    # (2) 뉴스 영향도 점수
    s_news = min(1.0, max(-1.0, float(news_sentiment)))
    if s_news >= 0.4:
        reasons.append(f"3D 뉴스 가중 평가 결과 상승 모멘텀(지표: {s_news:+.2f})이 우세합니다.")
    elif s_news <= -0.4:
        reasons.append(f"3D 뉴스 가중 평가 결과 하락 압력(지표: {s_news:+.2f})이 우세합니다.")

    # (3) 기술적 지표 점수 (MACD 40%, RSI 25%, BB 20%, Trend 15%)
    s_macd = 1.0 if macd >= macd_signal else -1.0
    if macd >= macd_signal:
        reasons.append("MACD 지표가 골든크로스(상승 전환) 상태입니다.")
    else:
        reasons.append("MACD 지표가 데드크로스(하락 전환) 상태입니다.")

    if rsi >= 75:
        s_rsi = -1.0
        reasons.append(f"RSI({rsi:.1f})가 과매수 영역으로 단기 조정 가능성이 높습니다.")
    elif rsi <= 35:
        s_rsi = 0.8
        reasons.append(f"RSI({rsi:.1f})가 과매도 지점으로 단기 반등 모멘텀이 존재합니다.")
    else:
        s_rsi = 0.0

    if bb_position > 0.9:
        s_bb = -1.0
        reasons.append("볼린저밴드 상단 근접으로 저항 압력이 존재합니다.")
    elif bb_position < 0.1:
        s_bb = 1.0
        reasons.append("볼린저밴드 하단 근접으로 지지력이 형성되어 있습니다.")
    else:
        s_bb = 0.0

    s_trend = 1.0 if predicted_change > 0 else -1.0
    s_tech = (0.40 * s_macd) + (0.25 * s_rsi) + (0.20 * s_bb) + (0.15 * s_trend)

    # (4) 커뮤니티 심리 점수 (샘플 수 신뢰도 승수 반영)
    if community_post_count >= 100:
        comm_mult = 1.0
    elif community_post_count >= 50:
        comm_mult = 0.8
    elif community_post_count >= 20:
        comm_mult = 0.6
    else:
        comm_mult = 0.3

    s_comm = min(1.0, max(-1.0, float(community_sentiment))) * comm_mult
    if s_comm >= 0.2:
        reasons.append(f"종목 토론방 투자 의도 분석 결과 매수 심리({s_comm:+.2f})가 유입되고 있습니다.")
    elif s_comm <= -0.2:
        reasons.append(f"종목 토론방 투자 의도 분석 결과 매도/우려 심리({s_comm:+.2f})가 우세합니다.")

    # (5) 과거 유사패턴 점수
    if pattern_avg_return is not None:
        s_pattern = min(1.0, max(-1.0, float(pattern_avg_return) / 10.0))
        if s_pattern > 0.2:
            reasons.append(f"과거 유사 차트 패턴의 평균 수익률({pattern_avg_return:+.2f}%)이 긍정적입니다.")
        elif s_pattern < -0.2:
            reasons.append(f"과거 유사 차트 패턴의 평균 수익률({pattern_avg_return:+.2f}%)이 부정적입니다.")
    else:
        s_pattern = 0.0

    # -------------------------------------------------------------
    # 3단계. 최종 총점 산출 (-100 ~ +100) 및 5단계 판정 매핑
    # -------------------------------------------------------------
    total_score = (
        (w_ml * s_ml) +
        (w_news * s_news) +
        (w_tech * s_tech) +
        (w_comm * s_comm) +
        (w_pattern * s_pattern)
    )

    total_scaled_score = round(total_score * 100.0, 1)
    normalized_score = min(100, max(0, int(round(50.0 + (total_scaled_score / 2.0)))))

    if total_scaled_score >= 40.0:
        outlook = "Strong Bullish"
    elif total_scaled_score >= 15.0:
        outlook = "Bullish"
    elif total_scaled_score >= -14.0:
        outlook = "Neutral"
    elif total_scaled_score >= -39.0:
        outlook = "Bearish"
    else:
        outlook = "Strong Bearish"


    llm_report = ""
    if generate_report:

        prompt = f"""
당신은 대한민국 대표 증권사의 수석 투자 전략가(Head Analyst)입니다.
아래 금융 데이터 및 기술적/감성 지표를 바탕으로 4단계 사고 체계(Chain-of-Thought)에 맞춰 논리적이고 입체적인 AI 투자 리포트를 작성하세요.

[필수 사고 단계 (Chain-of-Thought)]
1단계: 정량 지표 데이터 종합 요약 (주가 예측, RSI, MACD, 뉴스 감성)
2단계: 상승 모멘텀 요인 vs 하락/조정 리스크 요소 정밀 대조
3단계: ML 예측 신호와 단기 기술적 신호 간 충돌/상충 지점 해행 분석
4단계: 최종 투자 결론 및 대응 전략 가이드 제시

응답 형식 (반드시 아래 Markdown 헤더 구조를 유지할 것):

## 1단계: 정량 지표 종합 데이터 요약
...

## 2단계: 상승 모멘텀 vs 하락 리스크 핵심 대조
...

## 3단계: 신호 충돌 및 세부 분석 (ML vs 기술적 지표)
...

## 4단계: 최종 AI 투자 전망 & 리스크 대응 전략
...

[분석 대상 데이터]
- 규칙 기반 종합 판정: {outlook}
- ML 모델 1주일 후 예측 변동률: {predicted_change:+.2f}%
- 주요 산출 근거:
{chr(10).join(reasons)}

- 뉴스 감성 지표: {news_sentiment:+.2f} (긍정 {sentiment_counts.get('positive', 0) if sentiment_counts else 0}건 / 중립 {sentiment_counts.get('neutral', 0) if sentiment_counts else 0}건 / 부정 {sentiment_counts.get('negative', 0) if sentiment_counts else 0}건)
- 커뮤니티 투자 심리 지표: {community_sentiment:+.2f}
- 기술적 지표: RSI14 = {rsi:.2f}, MACD = {macd:.4f}, Signal = {macd_signal:.4f}, 볼린저 밴드 위치 = {bb_position:.2f}
- 주요 이슈 뉴스 요약: {news_summary}
"""

        try:
            llm_report = ask_llm(prompt)
        except Exception:
            llm_report = None

        if not llm_report or "LLM 분석 실패" in str(llm_report):
            llm_report = _generate_rule_based_cot_report(
                outlook=outlook,
                normalized_score=normalized_score,
                predicted_change=predicted_change,
                reasons=reasons,
                news_sentiment=news_sentiment,
                sentiment_counts=sentiment_counts or {},
                community_sentiment=community_sentiment,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                bb_position=bb_position,
                news_summary=news_summary
            )

        if llm_report:
            llm_report = llm_report.replace("综合分析", "").replace("以下是", "").replace("基于", "")

    # 신호 충돌 감지 (Signal Conflict - R² >= 0.0 인 유효 모델일 때만 적용)
    is_conflict = False
    conflict_msg = None

    is_low_r2 = bool(r2_score < 0.0)
    if not is_low_r2:

        if predicted_change > 0.5 and (macd < macd_signal or news_sentiment < -0.05):
            is_conflict = True
            conflict_msg = f"🚨 **[신호 충돌 감지]**: ML 모델은 상승(+{predicted_change:.2f}%)을 예측했으나, MACD 데드크로스 및 부정적 뉴스 감성({news_sentiment:+.2f}) 등 단기 기술적 지표는 하락/조정 압력을 가리키고 있습니다. 신중한 관망을 권장합니다."
        elif predicted_change < -0.5 and (macd > macd_signal or news_sentiment > 0.05 or rsi <= 35):
            is_conflict = True
            conflict_msg = f"🚨 **[신호 충돌 감지]**: ML 모델은 하락({predicted_change:.2f}%)을 예측했으나, RSI 과매도({rsi:.1f}) 및 긍정적 뉴스 감성({news_sentiment:+.2f}) 등 단기 반등 기술적 모멘텀이 포착되었습니다."

    return {
        "outlook": outlook,
        "score": normalized_score,
        "total_scaled_score": total_scaled_score,
        "reasons": reasons,
        "llm_report": llm_report,
        "is_conflict": is_conflict,
        "conflict_msg": conflict_msg,
        "weights_breakdown": {
            "ml_weight": round(w_ml * 100, 1),
            "news_weight": round(w_news * 100, 1),
            "tech_weight": round(w_tech * 100, 1),
            "comm_weight": round(w_comm * 100, 1),
            "pattern_weight": round(w_pattern * 100, 1),
        }
    }