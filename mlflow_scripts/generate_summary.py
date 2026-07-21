"""
MLflow 최신 모델의 예측 결과 + Spark 피처 데이터를 바탕으로
LLM(Ollama 로컬 모델)이 자연어 요약/설명을 생성해 저장합니다.

중요 설계 원칙:
  - LLM은 수치를 계산하지 않는다 (예측/피처 중요도는 이미 ML 단계에서 계산 완료)
  - LLM의 역할은 "해석 + 설명 + 대화"로 한정
  - 투자 자문이 아님을 명시하는 디스클레이머를 항상 포함
"""
import os
import json
from datetime import datetime

import requests
import mlflow
import mlflow.sklearn
import pandas as pd

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
MODEL_NAME = "stock_direction_classifier"
FEATURE_PATH = "/opt/airflow/data/processed/features.parquet"
OUT_PATH = "/opt/airflow/data/processed/llm_summary.json"

SYSTEM_PROMPT = """너는 투자 정보 해설 어시스턴트다. 아래 규칙을 반드시 지켜라.
1. 특정 종목의 매수/매도를 직접적으로 권유하지 않는다.
2. 모든 판단은 "~할 가능성이 있다", "~한 경향이 관찰된다" 같은 확률적/관찰적 표현만 사용한다.
3. 응답 마지막에 반드시 "본 내용은 투자 참고용 정보이며, 투자 판단과 책임은 본인에게 있습니다."를 포함한다.
4. 제공된 피처 중요도와 감성 점수를 근거로 반드시 함께 설명한다.
5. 출력은 반드시 JSON만 반환한다. 다른 텍스트를 포함하지 마라.
JSON 형식: {"summary": str, "key_factors": [str], "confidence_note": str, "disclaimer": str}
"""


def get_latest_predictions() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_PATH)
    latest_date = df["date"].max()
    return df[df["date"] == latest_date]


def call_llm(ticker: str, row: pd.Series, importance: dict) -> dict:
    user_prompt = f"""
종목: {ticker}
최근 종가 대비 등락률: {row['prev_close_pct_change']:.4f}
평균 감성점수(최근 뉴스): {row['avg_sentiment']:.4f}
언급량: {row['mention_count']}
5일 이평 대비 괴리율: {row['ma5_gap']:.4f}
주요 피처 중요도: {json.dumps(importance, ensure_ascii=False)}

위 데이터를 바탕으로 해당 종목의 최근 동향을 해석해줘.
"""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return json.loads(content)


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 최신 등록 모델에서 피처 중요도 아티팩트 로드
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production"])
    importance = {}
    if latest_version:
        run_id = latest_version[0].run_id
        local_path = client.download_artifacts(run_id, "feature_importance.csv")
        imp_df = pd.read_csv(local_path)
        importance = dict(zip(imp_df["feature"], imp_df["importance"].round(4)))

    latest_df = get_latest_predictions()
    results = {}
    for ticker, group in latest_df.groupby("ticker"):
        row = group.iloc[0]
        try:
            summary = call_llm(ticker, row, importance)
        except Exception as e:
            summary = {
                "summary": f"LLM 요약 생성 실패: {e}",
                "key_factors": [],
                "confidence_note": "",
                "disclaimer": "본 내용은 투자 참고용 정보이며, 투자 판단과 책임은 본인에게 있습니다.",
            }
        results[ticker] = summary

    output = {"generated_at": datetime.now().isoformat(), "results": results}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] LLM summary saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
