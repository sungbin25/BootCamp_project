# Stock Insight AI Platform (포트폴리오 데모)

종목을 검색하면 주가 차트, 변곡점(급등락) 자동 탐지, 관련 뉴스 LLM 요약, 영향도 분석,
미래 주가 예측까지 한 화면에서 제공하는 AI 기반 주식 분석 플랫폼입니다.

> ⚠️ 본 프로젝트는 학습/포트폴리오 목적이며, 실제 투자 자문 서비스가 아닙니다.
> 크롤링 대상 사이트의 이용약관과 robots.txt를 사전에 확인하세요.

## 두 개의 레이어로 구성됩니다

1. **실시간 서빙 레이어 (UR-01~UR-12 요구사항 담당)**
   `backend`(FastAPI) + `mysql` + `streamlit`. 사용자가 종목을 검색하면
   그 자리에서 즉시 가격 조회, 변곡점 탐지, 뉴스 수집·LLM 요약, 예측을 수행합니다.
   **이 레이어만으로도 데모가 완결됩니다.**

2. **배치 파이프라인 레이어 (MLOps 역량 시연용, 선택)**
   `airflow` + `spark` + `mlflow` + `minio`. 대량 데이터를 정기적으로 수집·정제하고
   모델을 재학습하는 파이프라인입니다. 실시간 서빙과는 독립적으로 동작합니다.

## 아키텍처

```
[사용자] → Streamlit(8501) → FastAPI backend(8000)
                                   ├─ yfinance          (UR-02,03 가격 조회)
                                   ├─ changepoint.py     (UR-04,05 변곡점 탐지)
                                   ├─ news_service.py    (UR-07 뉴스 수집, 네이버 API)
                                   ├─ llm_service.py     (UR-08,09,12 LLM 요약/분석/설명, Ollama)
                                   ├─ predict_service.py (UR-10,11 예측)
                                   └─ MySQL               (조회/예측 로그 저장)

[배치, 선택] Airflow → Spark → MLflow → MinIO   (대량 수집/재학습용, README 하단 참고)
```


## 사전 준비물

1. **Docker Desktop** (또는 Docker Engine + Compose v2) 설치
   - 최소 사양 권장: RAM 8GB 이상, 디스크 여유 10GB 이상
2. **네이버 오픈API 키** 발급 (무료)
   - https://developers.naver.com/apps/#/register 에서 애플리케이션 등록 → "검색" API 사용 설정
3. `.env.example`을 복사해 `.env` 생성 후 발급받은 키 입력

```bash
cp .env.example .env
# .env 파일 열어서 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 채우기
```

## 실행 순서

### 1) 전체 스택 기동
```bash
docker compose up -d
```
- 첫 실행 시 이미지 다운로드 + Airflow 커스텀 이미지 빌드(JDK 설치 포함)로 **10~15분** 정도 걸릴 수 있습니다.
- `-d`는 백그라운드 실행 옵션입니다.

> ℹ️ Spark는 `bitnami/spark` 무료 태그가 2025년 하반기부터 중단되어 Apache 공식 이미지(`apache/spark:3.5.3-python3`)를 사용합니다.
> Airflow는 `spark-submit`(pyspark) 실행에 JDK가 필요해서 공식 이미지를 그대로 쓰지 않고 `airflow_image/Dockerfile`로 직접 빌드합니다.

### 2) 각 서비스 접속 확인

| 서비스 | URL | 계정 |
|---|---|---|
| **Streamlit (메인 화면)** | http://localhost:8501 | - |
| **FastAPI backend (Swagger 문서)** | http://localhost:8000/docs | - |
| MySQL | localhost:3306 | stockapp / stockapp |
| Airflow (배치, 선택) | http://localhost:8081 | admin / admin |
| MLflow (배치, 선택) | http://localhost:5000 | - |
| Spark Master UI (배치, 선택) | http://localhost:8080 | - |
| MinIO 콘솔 (배치, 선택) | http://localhost:9001 | minioadmin / minioadmin |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | - |

### 3) [선택, 배치 파이프라인용] MinIO에 MLflow 아티팩트용 버킷 생성
메인 데모(종목 검색/차트/예측)만 보실 거라면 이 단계는 건너뛰어도 됩니다.
Airflow 배치 파이프라인까지 시연하실 경우에만, MinIO 콘솔(9001)에서
`mlflow-artifacts` 버킷을 하나 생성하세요.

### 4) Ollama 모델 다운로드 (최초 1회)
```bash
docker exec -it stock-llm-pipeline-ollama-1 ollama pull qwen2.5:7b
```
GPU가 없다면 더 가벼운 모델(`qwen2.5:1.5b`, `llama3.2:3b`)로 대체해도 됩니다.

### 5) 메인 데모 사용 (UR-01~UR-12)
1. http://localhost:8501 접속
2. 종목 검색창에 "삼성전자" 입력 → 검색결과에서 선택 → "이 종목 선택" 클릭
3. 상단 기간 버튼(1일~전체)으로 조회 기간 변경
4. 차트에 표시된 ★ 변곡점을 클릭(또는 드롭다운 선택) → 뉴스 수집·LLM 요약·영향도 분석 확인
5. "예측 시점"에서 1일/1주일/1개월 선택 → 차트에 빨간 점선으로 예측치 표시, 하단에 LLM 예측 근거 확인

> FastAPI 문서(http://localhost:8000/docs)에서 각 엔드포인트를 개별적으로 테스트해볼 수도 있습니다.

### 6) [선택] Airflow 배치 파이프라인 실행
1. http://localhost:8081 접속 (admin/admin)
2. `stock_pipeline_dag` 토글 ON → 우측 ▶(Trigger DAG)로 수동 1회 실행
3. MLflow(5000) → Experiments에서 학습 로그, Model Registry에서 버전 확인
4. Grafana(3000) → Airflow 메타DB / MySQL(조회·예측 로그) 연동된 대시보드로 실행 이력 조회

## 디렉토리 구조
```
.
├── docker-compose.yml          # 전체 스택 정의
├── init-multi-db.sh             # Postgres 내 airflow/mlflow DB 자동 생성 (배치용)
├── backend/                     # ★ 실시간 서빙 API (UR-01~UR-12 핵심)
│   ├── main.py                  # FastAPI 엔드포인트 전체
│   ├── ticker_map.py            # UR-01 종목 검색
│   ├── changepoint.py           # UR-04,05 변곡점 탐지
│   ├── news_service.py          # UR-07 뉴스 수집 (네이버 API)
│   ├── llm_service.py           # UR-08,09,12 LLM 요약/영향도/예측근거
│   ├── predict_service.py       # UR-10,11 예측 모델
│   ├── db.py                    # MySQL 로그 저장
│   └── Dockerfile
├── streamlit_app/
│   ├── app.py                   # ★ 메인 화면 UI (UR-01~UR-12 전체)
│   └── Dockerfile.streamlit
├── dags/
│   └── stock_pipeline_dag.py    # [배치] Airflow 메인 DAG
├── crawlers/                    # [배치] 대량 수집용 크롤러
├── spark_jobs/                  # [배치] Spark 정제
├── mlflow_scripts/              # [배치] 모델 재학습
└── monitoring/                  # Grafana/Prometheus
```

## 요구사항(UR) ↔ 코드 매핑

| 요구사항 | 구현 위치 |
|---|---|
| UR-01 종목 검색 | `backend/ticker_map.py`, `GET /search` |
| UR-02 기간 선택 | `streamlit_app/app.py` 상단 버튼, `backend/main.py`의 `PERIOD_MAP` |
| UR-03 주가 차트 | `GET /prices/{ticker}`, Streamlit plotly 차트 |
| UR-04 변곡점 탐지 | `backend/changepoint.py` |
| UR-05 이벤트 마커 | Streamlit 차트의 별(★) 마커 |
| UR-06 마커 클릭 | `streamlit-plotly-events` (미설치 시 드롭다운 대체) |
| UR-07 뉴스 자동 수집 | `backend/news_service.py` |
| UR-08 LLM 뉴스 요약 | `backend/llm_service.py::summarize_news` |
| UR-09 뉴스 영향도 분석 | `backend/llm_service.py::analyze_impact` |
| UR-10 미래 주가 예측 | `backend/predict_service.py` |
| UR-11 예측 시각화(점선) | Streamlit 차트의 빨간 점선 |
| UR-12 예측 근거 설명 | `backend/llm_service.py::explain_prediction` |

## 다음 단계로 확장하고 싶다면
- 감성분석을 간이 사전(`spark_jobs/clean_and_join.py`의 `POSITIVE_WORDS`) 대신
  **KR-FinBert** 등 사전학습 모델로 교체
- `check_retrain_needed`에 모델 드리프트(정확도 하락) 감지 로직 추가
- MLflow Model Registry의 Staging → Production 승격 프로세스 도입
- Grafana에 커스텀 대시보드 JSON 추가 (DAG 성공률, 평균 실행시간 패널)

## 트러블슈팅
- **Streamlit에서 "API 호출 실패"가 뜸**: `docker compose logs backend`로 원인 확인. 대부분
  MySQL 헬스체크가 끝나기 전에 backend가 뜨려다 실패한 경우이므로 `docker compose restart backend`
- **뉴스 요약/영향도/예측근거가 비어있음**: Ollama 모델을 아직 pull하지 않았을 가능성이 높습니다.
  `docker exec -it <ollama_container> ollama list`로 모델이 있는지 확인
- **"수집된 뉴스가 없습니다"만 나옴**: `.env`의 `NAVER_CLIENT_ID/SECRET` 미설정 또는 오류.
  `docker compose logs backend`에서 네이버 API 응답 상태코드 확인
- **변곡점이 하나도 안 뜸**: 짧은 기간(1일/1주일)에서는 조건(5일 누적 ±10%)을 만족하기 어렵습니다.
  1개월 이상 기간에서 확인하세요
- **`bitnami/spark:3.5: not found` 에러**: 최신 파일을 받으셨다면 이미 `apache/spark:3.5.3-python3`로 교체되어 있습니다.
  혹시 예전 zip을 쓰고 계시다면 `docker-compose.yml`의 spark-master/spark-worker 이미지를 이 태그로 바꿔주세요.
- **Airflow 컨테이너가 계속 재시작됨**: `docker compose logs airflow-init`으로 DB 마이그레이션 로그 확인
- **Spark job이 데이터를 못 찾음**: `data/` 볼륨 마운트 경로가 Airflow 컨테이너(`/opt/airflow/data`)와
  Spark 컨테이너(`/opt/data`)에서 다르므로, DAG의 BashOperator 명령어 경로를 컨테이너 기준으로 맞췄는지 확인
- **네이버 API 401 에러**: `.env`의 Client ID/Secret이 정확한지, 애플리케이션에 "검색" API가 활성화되어 있는지 확인
