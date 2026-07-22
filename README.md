# Stock Insight AI Platform (Portfolio Project)

AI 기반 주식 종합 분석 및 시계열 머신러닝 예측 플랫폼입니다.

본 프로젝트는 단순한 주가 예측이나 텍스트 요약에 그치지 않고, **정량 데이터와 정성 데이터를 융합한 머신러닝 기반 투자 의사결정 지원 시스템**입니다. 
사용자는 종목 검색 시 주가 차트, 변곡점(급등락) 자동 탐지, 뉴스·커뮤니티 여론 수집 및 LLM 종합 감성 분석, **RandomForest & SHAP 기반 주가 방향 예측 및 연속적 R² 신뢰도 캘리브레이션**, 그리고 **Historical Replay를 통한 과거 예측 성능 실증 검증**까지 하나의 종합 대시보드에서 온디맨드(On-demand)로 확인할 수 있습니다.

> ⚠️ **Disclaimer**: 본 프로젝트는 교육 및 포트폴리오 목적으로 제작되었으며, 실제 투자 자문이나 금융 권유 서비스를 제공하지 않습니다.

---

## 📸 Platform Overview (시연 스크린샷)

<p align="center">
  <img src="docs/images/image01.png" width="95%" alt="Main Dashboard Screenshot" /><br>
  <em>[메인 홈 화면 : 주식 목록 및 실시간 브리핑]</em>
</p>

<p align="center">
  <img src="docs/images/image02.png" width="47%" alt="Historical Replay Screenshot" />
  <img src="docs/images/image03.png" width="47%" alt="Community Analysis Screenshot" /><br>
  <em>[좌: 주가 그래프 및 변곡점 탐지 | 우: 커뮤니티 감성 분석]</em>
</p>

<p align="center">
  <img src="docs/images/image04.png" width="95%" alt="News Analysis Screenshot" /><br>
  <em>[과거 유사 패턴 매칭 및 뉴스 영향도 분석]</em>
</p>

<p align="center">
  <img src="docs/images/image05.png" width="47%" alt="Prediction Screenshot" />
  <img src="docs/images/image06.png" width="47%" alt="SHAP Analysis Screenshot" /><br>
  <em>[좌: 뉴스 상세 요약 | 우: SHAP 기반 피처 기여도 시각화]</em>
</p>

---

## 🌟 주요 기능 (Key Features)

- **🔍 On-demand 종목 검색 & 차트 조회**: KOSPI / KOSDAQ 전 종목 검색 및 기간별(1일~전체) 동적 주가/거래량 Plotly 차트 표출
- **⚡ 변곡점(급등락) 자동 탐지**: 5일 누적 등락률 ±10% 이상 및 거래량 급증(z-score ≥ 2.0) 기반 기술적 변곡점 자동 포착 및 이벤트 마커 시각화
- **📊 43개 정량·정성 Feature Engineering**: 주가/거래량, 기술적 지표, 거시경제, 뉴스/커뮤니티 감성 지표 통합
- **🌲 RandomForest & Ensemble Voting**: 비선형 파라미터 학습 및 100개 Decision Tree Consensus 기반 방향성 예측
- **📐 연속적 R² 신뢰도 캘리브레이션**: 회고 평가 R² 점수와 Tree Consensus를 연동한 연속적 신뢰도 보정 수식으로 예측 불확실성 조율
- **🔍 Explainable AI (SHAP 피처 해설)**: SHAP(SHapley Additive exPlanations) 값 기반 상승/하락 기여 지표 수치화 및 Plotly 시각화
- **📈 Historical Replay (회고 오차 실증 검증)**: 과거 특정 변곡 시점으로 돌아가 당시 데이터 기반 AI 예측 재현 및 실제 주가 추이와 대조 오차 분석
- **📰 뉴스 3차원 영향도 & 💬 커뮤니티 6대 인텐트 분석**: 네이버 뉴스/종토방 온디맨드 수집 및 로컬 LLM(Ollama Qwen2.5) 기반 핵심 요약 및 감성 분류
- **💡 LLM 기반 단계적 투자 분석 보고서**: 기술 지표 + 감성 분석 + ML 예측을 종합 결합한 최종 투자 의견(매수/관망/매도) 생성

---

## 🧠 Machine Learning Pipeline

### 📊 머신러닝 예측 파이프라인 개요

본 프로젝트는 단순한 주가 예측이 아닌, **정량 데이터와 정성 데이터를 융합한 머신러닝 기반 투자 의사결정 지원 시스템**입니다.

예측 과정은 다음과 같은 체계적인 7단계 파이프라인으로 수행됩니다.

```text
               원시 데이터 수집 (OHLCV, KOSPI/KOSDAQ, 환율, VIX, 뉴스, 커뮤니티)
                                        │
                                        ▼
                               Feature Engineering
                                (총 43개 Feature 생성)
                                        │
                                        ▼
                             RandomForest 모델 학습
                               (1일 / 1주 / 1개월 예측)
                                        │
                                        ▼
                              Tree Ensemble Voting
                          (100개 Decision Tree Consensus)
                                        │
                                        ▼
                             R² 기반 신뢰도 보정
                       (Continuous R² Calibration Factor)
                                        │
                                        ▼
                             SHAP Explainable AI
                        (주요 지표별 상승/하락 기여도 산출)
                                        │
                                        ▼
                             최종 AI 투자 의견 생성
                    (가격 + 방향성 + 최종 신뢰도 + 근거 보고서)
```

---

### 📊 Feature Engineering (총 43개 정량·정성 Feature)

모델 성능 향상을 위해 **정량 데이터(주가·지표·매크로)와 정성 데이터(뉴스·커뮤니티 감성)를 결합한 총 43개의 Feature**를 다차원 생성합니다.

#### 1. 주가·거래량 Feature (17개)
* OHLCV (시가, 고가, 저가, 종가, 거래량)
* 1일 / 5일 / 20일 수익률
* 이동평균 이격도 (MA5 / MA20 / MA60 Disparity)
* 거래량 비율 및 변화율
* 주가 변동성 (Volatility)
* Lag Feature (시계열 지연 지표)

#### 2. 기술적 지표 (8개)
* RSI (상대강도지수, 14일)
* MACD & MACD Signal & MACD Histogram
* Bollinger Bands (%B, Upper/Lower Band Gap)
* ATR (Average True Range, 14일)
* Volume Z-Score (거래량 표준편차 점수)

#### 3. 시장 & 거시경제 Feature (9개)
* KOSPI 수익률 & 이격도
* KOSDAQ 수익률
* 원/달러 환율 (USD/KRW) 변동률 & 이격도
* VIX Index (시장 변동성 지수)
* 시장 전체 거래대금 변화율

#### 4. 뉴스·커뮤니티 감성 Feature (9개)
* 뉴스 긍부정 감성 점수 (LLM / FinBERT)
* 뉴스 영향도 지수 (3일 / 7일 누적)
* 커뮤니티 투자심리 지수 (네이버 종토방 수집)
* Bullish / Bearish 비율 및 여론 강도

---

### 🌲 RandomForest 선택 이유

본 프로젝트에서 회귀/분류 핵심 예측 모델로 **RandomForest**를 채택한 이유는 다음과 같습니다.

* **비선형 패턴 학습**: 복잡하고 비선형적인 금융 시계열 데이터의 파라미터를 효과적으로 학습
* **과적합(Overfitting) 방지**: 여러 Decision Tree의 앙상블로 노이즈가 많은 주가 데이터에서 뛰어난 일반화 성능 보임
* **Feature Importance 제공**: 변수 중요도 평가 및 시각화 용이
* **안정성**: 데이터 스케일 변화에 견고하며 금융 타임시리즈 예측에서 우수한 안정성 검증

본 모델은 **수치 예측(Regression)**을 수행하여 향후 **1일, 1주일(5일), 1개월(20일)** 단위 주가와 등락 방향을 다기간으로 예측합니다.

---

### 🎯 Ensemble Voting (Tree Consensus)

단순히 최종 평균 예측값만 사용하는 것이 아니라, **RandomForest 내부 100개 Decision Tree들의 예측 방향성을 비교하여 Tree Voting Consensus**를 산출합니다.

```text
             ┌──────────────────────────────────────────┐
             │       100개의 Decision Tree 예측        │
             └────────────────────┬─────────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
            ▲ 상승 : 81개                    ▼ 하락 : 19개
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                     Tree Consensus = 81%
```

> **Tree Consensus**가 높을수록 모델 내부의 개별 나무들이 일관되게 같은 방향(상승/하락)을 지지한다는 의미이므로, 예측의 방향성 확실성이 높아집니다.

---

### 📈 Continuous Confidence Calibration (연속적 R² 신뢰도 보정)

RandomForest의 Voting 결과만으로는 실제 과거 검증 성과가 반영되지 않으므로, **과거 검증 결과($R^2$ 점수)를 연동한 보정 수식**으로 최종 신뢰도를 조율합니다.

```text
Tree Consensus (Voting 부호 일치율)
        │
        ▼
Historical R² (회고 평가 성과 연동)
        │
        ▼
Continuous Calibration
        │
        ▼
Final Direction Confidence (최종 신뢰도)
```

#### 📐 캘리브레이션 수식
$$\text{Tree Consensus} = \frac{\text{방향 일치 Tree 개수}}{\text{전체 Tree 개수}} \times 100\%$$

$$\text{Calibration Factor}(R^2) = \max\left(0.4, \min\left(1.0, \frac{R^2 + 1.0}{2.0}\right)\right)$$

$$\text{Final Direction Confidence} = \max\left(50.0\%, \min\left(99.0\%, \text{Tree Consensus} \times \text{Calibration Factor}(R^2)\right)\right)$$

* **예시 시나리오**:
  $$\text{Tree Consensus } 92\% \quad \xrightarrow{\quad R^2 = 0.63 \text{ 성과 연동} \quad} \quad \text{최종 신뢰도 } 81\%$$
  과거 성과가 저조한 구간에서는 신뢰도가 과대평가되지 않도록 안정적으로 감쇄 조정됩니다.

---

### 🔍 Explainable AI (SHAP 피처 기여도 해설)

AI의 예측 결과뿐만 아니라 **"왜 그러한 판단을 내렸는지"**에 대한 정량적 근거를 설명 가능하도록 SHAP(SHapley Additive exPlanations)을 적용합니다.

각 Feature가 예측 주가 상승(+) 또는 하락(-)에 기여한 영향도를 개별 계산합니다.

| Feature | 변수 유형 | 기여 영향도 | 해설 |
| :--- | :--- | :--- | :--- |
| **뉴스 감성** | 정성 지표 | **+15%** | 호재 뉴스 비율 및 LLM 감성 점수 긍정적 기여 |
| **RSI (14)** | 기술적 지표 | **+12%** | 과매도 구간脱出에 따른 기술적 반등 신호 |
| **거래량 Z-Score**| 주가/거래량 | **+9%** | 평균 대비 거래량 급증에 따른 상승 동력 확보 |
| **MACD Hist** | 기술적 지표 | **+8%** | 매수 골든크로스 시그널 형성 |
| **커뮤니티 심리** | 정성 지표 | **-5%** | 단기 과열에 따른 개인 투자자 음봉 우려 |

이를 **Plotly 대화형 바 차트**로 시각화하여 사용자가 AI의 판단 근거를 직관적으로 검증할 수 있습니다.

---

### 📈 Historical Replay (회고 시뮬레이션 및 오차 실증 분석)

대부분의 주가 예측 프로젝트가 미래 예측값만 제시하고 검증이 불가능한 한계를 극복하기 위해, **과거 특정 변곡 시점으로 돌아가 당시 데이터만으로 AI 판단을 재현하는 Historical Replay**를 제공합니다.

```text
과거 변곡 데이터 선택 ──> AI 예측 재현 ──> 실제 5일 주가 대조 ──> 적중 여부(✅/❌) & 오차 원인 실증 분석
```

```text
📈 AI Historical Replay (회고 시뮬레이션 및 오차 검증 시연)

[ Replay 종합 점수 ]   [ 당시 AI 예상 5일 ]   [ 실제 발생 5일 ]   [ 예측 적중 여부 ]
   96 / 100점               ▲ +11.5%             ▼ -7.5%          ❌ 미적중 (오차: 19.0%p)

🧐 AI 회고 및 오차 원인 실증 분석
 - ✔ 변곡 당일 기술적 지표 신호(거래량/크로스) 포착 성공
 - ✖ 변곡 직후 외국인·기관 수급 이탈 및 매크로 시장 조정 발생
```

---

### ⭐ 프로젝트 차별성 비교

| 구분 | 기존 일반 주가 예측 프로젝트 | 본 Stock Insight AI 플랫폼 |
| :--- | :--- | :--- |
| **예측 범위** | 단순 가격 수치 출력 | **가격 + 5일/20일 방향성 + 최종 신뢰도 점수** |
| **데이터 활용** | 주가/기술적 지표 위주 | **정량(주가·지표·매크로) + 정성(뉴스·커뮤니티) 43개 융합** |
| **예측 근거** | 블랙박스 AI (결과만 제시) | **SHAP 기반 지표별 상승/하락 기여도 정량 해설** |
| **신뢰도 평가** | 모델 고정 정확도 제시 | **Tree Voting Consensus + R² Calibration 수식 보정** |
| **검증 체계** | 과거 테스트 미제공 | **Historical Replay 시뮬레이션을 통한 과거 오차 실증** |
| **설명 가능성** | 블랙박스 (Black-box) | **Explainable AI (XAI) + LLM 종합 보고서 융합** |

---

## 🛠 Tech Stack

| 구분 | 기술 스택 | 설명 |
| :--- | :--- | :--- |
| **Frontend** | Streamlit, Plotly, HTML5/CSS3 | 대화형 프론트엔드 대시보드 및 동적 차트 시각화 |
| **Backend** | FastAPI, Python 3.11, Pydantic, SQLAlchemy | On-demand RESTful API 서빙 및 캐싱 |
| **AI / ML** | RandomForest (Scikit-Learn), SHAP, Ollama (Qwen2.5:7b) | 43개 Feature 기반 회귀/방향 예측, SHAP 피처 해설, LLM 감성/요약 |
| **Database** | MySQL 8.0 | 사용자 조회 및 예측 이력 저장 |
| **MLOps** | Airflow, Spark 3.5, MLflow, MinIO | Airflow 데이터 수집, Spark Feature Engineering, MLflow 레지스트리, MinIO Artifact |
| **Monitoring** | Prometheus, Grafana | 배치 DAG 성공률 및 시스템 메트릭 모니터링 |
| **Infrastructure**| Docker, Docker Compose | 전체 11개 컨테이너 서비스 오케스트레이션 |

---

## 🏗 System Architecture & Internal Pipeline

### 1. 시스템 아키텍처 (Architecture)

```text
                                [ 사용자 ]
                                    │
                                    ▼
                         Streamlit Frontend (8501)
                                    │
                                    ▼
                         FastAPI Backend (8000)
                                    │
  ┌───────────────┬─────────────────┼─────────────────┬────────────────┐
  │               │                 │                 │                │
  ▼               ▼                 ▼                 ▼                ▼
yfinance      changepoint      news_service      community_service   predict_service
(주가/지표)  (변곡점 탐지)     (뉴스 수집)       (토론방 크롤링)     (RF/SHAP 예측)
                                    │                 │                │
                                    └────────┬────────┘                │
                                             ▼                         │
                                      llm_service                      │
                                   (Ollama Qwen 2.5)                   │
                                             │                         │
  ───────────────────────────────────────────┴─────────────────────────┴────────────
  [MLOps 파이프라인] Airflow (8081) ──> Spark ──> MLflow (5000) ──> MinIO (9001) / MySQL
```

### 2. 처리 흐름 (Data Flow)

```text
사용자 종목 검색 요청 (On-demand)
    │
    ▼
주가 데이터 수집 (yfinance API) & 기술적 지표 산출
    │
    ▼
변곡점 자동 탐지 (5일 누적 등락률 ±10% / 거래량 Z-Score ≥ 2.0)
    │
    ├─────────────────────────────┐
    ▼                             ▼
뉴스 수집 (네이버 API)         커뮤니티 수집 (네이버 증권 토론방)
    │                             │
    └──────────────┬──────────────┘
                   ▼
       LLM 요약 및 6대 인텐트 감성 분류 (Ollama)
                   │
                   ▼
      기술적 + 감성 43개 Feature 생성
                   │
                   ▼
   RandomForest 회귀 예측 & Tree Consensus 계산
                   │
                   ▼
   연속적 R² 캘리브레이션 적용 (Final Confidence 산출)
                   │
                   ▼
   SHAP 피처 기여도 해설 생성
                   │
                   ▼
   Historical Replay 시뮬레이션 & 오차 실증 분석
                   │
                   ▼
   LLM 기반 단계적 투자 분석 보고서 출력
```

---

## 🤖 AI & LLM 분석 파이프라인

### 📰 뉴스 분석 (News Pipeline)
- 네이버 뉴스 API를 통해 직근 3일간의 종목 뉴스 온디맨드 수집
- LLM을 통해 핵심 헤드라인 3줄 요약, 긍정/부정/중립 감성 점수 산출 및 3차원 주가 영향도 평가

### 💬 커뮤니티 분석 (Community Pipeline)
- 종목별 커뮤니티(네이버 증권 종목토론방) 최신 게시글 100~300건 온디맨드 수집
- 1차 사전 분류 + 2차 LLM 6대 인텐트 분류(`Bullish`, `Bearish`, `Neutral`, `Question`, `News`, `Humor`)를 거쳐 여론 비중 집계 및 핵심 이슈 요약 생성

### 💡 LLM 기반 단계적 투자 분석 보고서 (Investment Decision Report)
- 뉴스 감성, 커뮤니티 여론, 기술적 지표(MACD, RSI, BB), 머신러닝 주가 방향 예측 결과를 종합 가중 평가하여 **매수(BUY)**, **관망(HOLD)**, **매도(SELL)** 단계적 리포트 생성

---

## 📁 디렉토리 구조 (Directory Structure)

```text
stock-llm-pipeline/
├── docker-compose.yml              # 전체 통합 11개 컨테이너 스택 정의
├── .env.example                    # 환경변수 템플릿 (네이버 API 키 등)
├── backend/                        # ★ FastAPI 백엔드 API (On-demand 서빙)
│   ├── main.py                     # FastAPI 엔드포인트 라우팅 및 캐싱
│   ├── predict_service.py          # 머신러닝 예측, SHAP 및 R² 캘리브레이션
│   ├── news_service.py             # 네이버 뉴스 수집 & LLM 영향도 분석
│   ├── community_service.py        # 네이버 증권 토론방 수집 & 6대 인텐트 분류
│   ├── llm_service.py              # Ollama LLM 연동 헬퍼
│   ├── changepoint.py              # 변곡점 탐지 및 Historical Replay
│   ├── analyze_service.py          # LLM 기반 단계적 투자 분석 보고서 생성 Engine
│   ├── ticker_map.py               # 종목 코드 매핑
│   └── db.py                       # MySQL 데이터베이스 연동
├── streamlit_app/
│   ├── app.py                      # ★ 메인 대시보드 UI (Streamlit & Plotly)
│   └── Dockerfile.streamlit
├── dags/
│   └── stock_pipeline_dag.py       # [MLOps] Airflow 정기 데이터 수집 & 재학습 DAG
├── crawlers/                       # [MLOps] 배치 수집용 크롤러
├── spark_jobs/                     # [MLOps] Spark Feature Engineering 작업
├── mlflow_scripts/                 # [MLOps] MLflow 모델 재학습 및 레지스트리 관리
└── monitoring/                     # Prometheus & Grafana 대시보드 설정
```

---

## 🚀 시작하기 (Quick Start)

### 1. 사전 준비
- Docker Engine & Docker Compose v2 (권장 RAM 8GB 이상)
- 네이버 개발자 센터 (https://developers.naver.com) 검색 API 키 발급

### 2. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일 내 NAVER_CLIENT_ID 및 NAVER_CLIENT_SECRET 입력
```

### 3. 전체 스택 실행
```bash
docker compose up -d
```

### 4. Ollama LLM 모델 다운로드 (최초 1회)
```bash
docker exec -it stock-llm-pipeline-ollama-1 ollama pull qwen2.5:7b
```

### 5. 서비스 접속 정보
- **Streamlit 메인 UI**: `http://localhost:8501`
- **FastAPI Docs**: `http://localhost:8000/docs`
- **MLflow Dashboard**: `http://localhost:5000`
- **Airflow DAG Manager**: `http://localhost:8081` (admin / admin)
- **MinIO Console**: `http://localhost:9001` (minioadmin / minioadmin)
- **Grafana Monitoring**: `http://localhost:3000` (admin / admin)

---

## 📋 요구사항 (UR) ↔ 코드 매핑표

| 요구사항 ID | 요구사항 명칭 | 구현 파일 위치 |
| :--- | :--- | :--- |
| **UR-01** | 종목 검색 | `backend/ticker_map.py`, `GET /search` |
| **UR-02/03** | 주가 차트 및 기간 조회 | `streamlit_app/app.py`, `GET /prices/{ticker}` |
| **UR-04/05** | 변곡점 자동 탐지 & 마커 | `backend/changepoint.py`, `detect_changepoints` |
| **UR-07** | 뉴스 수집 | `backend/news_service.py` |
| **UR-08/09** | LLM 뉴스 요약 & 영향도 | `backend/news_service.py`, `backend/llm_service.py` |
| **UR-10** | 머신러닝 주가 방향 예측 | `backend/predict_service.py` |
| **UR-11** | 예측 점선 시각화 | `streamlit_app/app.py` (Plotly overlay) |
| **UR-12** | 예측 근거 & SHAP 분해 시각화 | `backend/predict_service.py`, `streamlit_app/app.py` |
| **UR-13** | 커뮤니티 6대 인텐트 수집/요약 | `backend/community_service.py` |
| **UR-14** | LLM 단계적 투자 분석 보고서 | `backend/analyze_service.py` |
| **UR-15** | Historical Replay 오차 실증 검증 | `backend/changepoint.py` |
