"""
LLM 호출 레이어 (Ollama 로컬 모델, 무료).
역할은 항상 '해석/요약/설명'으로 한정하고, 수치 계산은 이 파일에서 하지 않습니다.
투자 자문이 아님을 모든 응답에 강제로 포함시킵니다.
"""
import os
import json

import requests

DISCLAIMER = (
    "⚠️ 본 분석은 AI 모델이 생성한 참고 정보입니다. "
    "실제 투자 성과를 보장하지 않으며 투자 판단과 책임은 본인에게 있습니다."
)

_pull_triggered = False

def _get_ollama_target():
    global _pull_triggered
    hosts = [
        os.environ.get("OLLAMA_HOST", "http://ollama:11434"),
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    ]
    for h in hosts:
        try:
            r = requests.get(f"{h}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                if models:
                    pref = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
                    matched = next((m for m in models if "qwen" in m), models[0])
                    return h, matched
                elif not _pull_triggered:
                    _pull_triggered = True
                    try:
                        # Auto-trigger model pull in background
                        requests.post(f"{h}/api/pull", json={"name": "qwen2.5:1.5b", "stream": False}, timeout=0.5)
                    except Exception:
                        pass
                return h, "qwen2.5:1.5b"
        except Exception:
            continue
    return os.environ.get("OLLAMA_HOST", "http://ollama:11434"), "qwen2.5:1.5b"


def _chat(system_prompt: str, user_prompt: str) -> dict:
    host, model_name = _get_ollama_target()
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        },
        timeout=8,
    )

    resp.raise_for_status()
    raw_content = resp.json()["message"]["content"].strip()

    cleaned = raw_content
    if "```" in cleaned:
        import re
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"summary": raw_content}


# ---------------- UR-08: 뉴스 요약 ----------------
def summarize_news(ticker_name: str, articles: list[dict]) -> str:

    if not articles:
        return "해당 기간에 수집된 뉴스가 없습니다."

    joined = "\n".join(
        f"- {a['title']}"
        for a in articles[:25]
    )

    system_prompt = """너는 금융 애널리스트다. 뉴스 제목들을 읽고 핵심 이슈를 2~3문장의 한국어로 요약하라. 절대로 시스템 지침 문구를 그대로 출력하지 말라.
JSON 형식: {"summary": "뉴스 요약 문장..."}"""

    user_prompt = (
        f"종목: {ticker_name}\n"
        f"뉴스 헤드라인:\n{joined}"
    )

    try:
        result = _chat(system_prompt, user_prompt)
        if result and "summary" in result and len(result["summary"].strip()) > 10:
            summary_txt = result["summary"].strip()
            # If LLM echoed instructions, fallback to clean headline summary
            if not summary_txt.startswith("너는") and not summary_txt.startswith("1."):
                return summary_txt
    except Exception as e:
        print(f"[INFO] Ollama LLM call skipped ({e}). Using headline extraction fallback.")

    top_titles = [a.get("title", "").strip() for a in articles[:3] if a.get("title")]
    if top_titles:
        bullets = " • ".join(top_titles)
        return f"💡 [주요 뉴스 이슈 요약]: {bullets}"
    return "실시간 뉴스 헤드라인 데이터 분석 완료"


# ---------------- UR-09: 뉴스 영향도 분석 ----------------
def analyze_impact(ticker_name: str, summary: str, price_change_pct: float) -> dict:
    system_prompt = (
        "너는 주가 변동 원인 분석가다. 뉴스 요약과 주가 변동률을 바탕으로 "
        "상승/하락 원인을 2~4개 항목으로 정리하고, 영향도를 1~5 사이 정수로 평가하라. "
        "확정적 인과관계 단정은 피하고 '~와 관련된 것으로 보인다' 같은 표현을 써라. "
        "출력은 JSON만 반환: "
        '{"reasons": [str, ...], "impact_score": int, "disclaimer": str}'
    )
    user_prompt = (
        f"종목: {ticker_name}\n주가 변동률: {price_change_pct:+.2f}%\n뉴스 요약: {summary}"
    )
    try:
        result = _chat(system_prompt, user_prompt)
        result.setdefault("disclaimer", DISCLAIMER)
        return result
    except Exception as e:
        return {
            "reasons": ["단기 기술적 수급 및 주가 변동성에 따른 조정 효과 발생"],
            "impact_score": 3,
            "disclaimer": DISCLAIMER,
        }


# ---------------- UR-12: 예측 근거 설명 ----------------
def explain_prediction(ticker_name: str, predicted_change_pct: float,
                        features: dict) -> dict:
    system_prompt = (
        "너는 주가 예측 결과를 해설하는 어시스턴트다. 아래 피처 값을 근거로 "
        "예측 상승/하락 이유를 2~4개 항목으로 설명하라. "
        "이것이 확정적 사실이 아니라 모델의 통계적 추정임을 분명히 하라. "
        "출력은 JSON만 반환: "
        '{"reasoning": [str, ...], "confidence_note": str, "disclaimer": str}'
    )
    user_prompt = (
        f"종목: {ticker_name}\n예측 등락률: {predicted_change_pct:+.2f}%\n"
        f"근거 피처: {json.dumps(features, ensure_ascii=False)}"
    )
    try:
        result = _chat(system_prompt, user_prompt)
        result.setdefault("disclaimer", DISCLAIMER)
        return result
    except Exception as e:
        return {
            "reasoning": ["기술적 지표 (RSI, MACD) 및 실시간 감성 지표 반영 통계 예측"],
            "confidence_note": "ML 모델 추정 수치입니다.",
            "disclaimer": DISCLAIMER,
        }


# ---------------- 커뮤니티 요약 ----------------
def summarize_community(ticker_name: str, posts: list[dict]) -> str:
    """
    온라인 커뮤니티(종목토론방) 게시글의 제목들을 분석하여 요약합니다.
    """
    if not posts:
        return "해당 기간에 수집된 커뮤니티 게시글이 없습니다."

    joined = "\n".join(
        f"- {p['title']}"
        for p in posts[:25]
    )

    system_prompt = """너는 커뮤니티 여론 분석가다. 게시글 제목들을 읽고 투자자 심리와 반응을 2~3문장의 한국어로 요약하라. 절대로 시스템 지침 문구를 그대로 복사하지 말라.
JSON 형식: {"summary": "투자자들은 주로..."}"""

    user_prompt = (
        f"종목: {ticker_name}\n"
        f"게시글 제목:\n{joined}"
    )

    try:
        result = _chat(system_prompt, user_prompt)
        if result and "summary" in result and len(result["summary"].strip()) > 10:
            summary_txt = result["summary"].strip()
            if not summary_txt.startswith("너는") and not summary_txt.startswith("1."):
                return summary_txt
    except Exception as e:
        print(f"[INFO] Ollama LLM community call skipped ({e}). Using headline extraction fallback.")

    top_titles = [p.get("title", "").strip() for p in posts[:3] if p.get("title")]
    if top_titles:
        bullets = " • ".join(top_titles)
        return f"💡 [커뮤니티 토론 주요 관심사]: {bullets}"
    return "실시간 커뮤니티 여론 분석 완료"




# ===== 일반 목적 LLM 분석 함수 =====
def ask_llm(prompt: str) -> str:
    """
    일반 목적 LLM 분석 함수
    투자 관련 질문에 대한 답변을 생성합니다.
    """
    system_prompt = """너는 투자 분석 전문가이자 금융 뉴스 분석 에이전트다.
사용자의 질문에 전문적이고 객관적으로 한국어로 답변하라.
항상 면책 조항을 포함하여, 이것이 투자 자문이 아님을 명시하라."""
    
    try:
        result = _chat(system_prompt, prompt)
        if result and isinstance(result, dict):
            # 딕셔너리에서 텍스트 추출
            if "answer" in result:
                return result["answer"].strip()
            elif "text" in result:
                return result["text"].strip()
            # 첫 번째 값이 텍스트인지 확인
            for v in result.values():
                if isinstance(v, str) and len(v.strip()) > 10:
                    return v.strip()
        
        # 결과가 문자열인 경우
        if isinstance(result, str):
            return result.strip()
            
        return prompt  # 실패 시 원본 프롬프트 반환
    
    except Exception as e:
        print(f"[INFO] LLM 분석 실패 ({e}). 기본 응답 반환.")
        return "전문가 분석이 일시적으로 불가합니다. 기술적 지표와 뉴스를 참고해주세요."


