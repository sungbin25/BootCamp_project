"""
Stock Insight AI Platform - Toss Securities Style UI Frontend (Streamlit)

Features:
- Main Page: Search bar, popular stocks TOP10, major news, AI recommended stocks.
- Detail Page:
  - Upper: name, price, change %, cap, volume.
  - Left (70%): Plotly Chart + 점선 AI Prediction, 2-step (Classifier + Regressor) Multi-horizon predictions card.
  - Right (30%): 🤖 AI 종합 분석 (outlook, score, reasons, LLM report).
  - Middle: Related news & News AI Analysis.
  - Bottom: Live crawled community reaction posts & Community AI Analysis.
"""
import os
import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import datetime

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
DISCLAIMER = "⚠️ 본 서비스는 포트폴리오 데모이며, 모든 정보는 투자 참고용입니다. 투자 판단과 책임은 본인에게 있습니다."

TRACKED_STOCKS = [
    {"ticker": "005930.KS", "name": "삼성전자"},
    {"ticker": "000660.KS", "name": "SK하이닉스"},
    {"ticker": "035420.KS", "name": "NAVER"},
    {"ticker": "035720.KS", "name": "카카오"},
    {"ticker": "005380.KS", "name": "현대차"},
    {"ticker": "051910.KS", "name": "LG화학"},
    {"ticker": "373220.KS", "name": "LG에너지솔루션"},
]

PERIOD_OPTIONS = [
    ("1주일", "1w"), ("1개월", "1m"), ("3개월", "3m"), ("6개월", "6m"), ("1년", "1y"), ("전체", "all"),
]

st.set_page_config(page_title="Stock Insight AI Platform", layout="wide", initial_sidebar_state="collapsed")

# Custom Toss-Style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .stock-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    .stock-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.05em;
    }
    .prediction-box {
        background: linear-gradient(135deg, #eff6ff, #f5f3ff);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 20px;
        margin-top: 15px;
    }
    .news-card {
        border-bottom: 1px solid #f3f4f6;
        padding: 12px 0;
    }
    .news-title {
        font-weight: 600;
        font-size: 1rem;
        color: #1f2937;
        text-decoration: none;
    }
    .news-title:hover {
        color: #3b82f6;
    }
    .community-post {
        background-color: #f9fafb;
        color: #1f2937;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

def api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API 호출 실패 ({path}): {e}")
        return None


def api_post(path: str, json_data: dict | None = None) -> dict | None:
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json=json_data, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API POST 호출 실패 ({path}): {e}")
        return None

# ---------------- Session State Initializers ----------------
if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "ticker_name" not in st.session_state:
    st.session_state.ticker_name = None
if "period" not in st.session_state:
    st.session_state.period = "1m"
if "selected_event_date" not in st.session_state:
    st.session_state.selected_event_date = None

# TOP 10 live price cache
if "top_stocks_cache" not in st.session_state:
    st.session_state.top_stocks_cache = None

# Header Logo area
logo_col, back_col = st.columns([8, 2])
with logo_col:
    st.markdown('<div class="main-title">📊 Stock Insight AI Platform</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{DISCLAIMER}</div>', unsafe_allow_html=True)
with back_col:
    if st.session_state.ticker is not None:
        if st.button("↩️ 홈으로 이동", use_container_width=True):
            st.session_state.ticker = None
            st.session_state.ticker_name = None
            st.session_state.selected_event_date = None
            st.rerun()

# =========================================================================
# HOME SCREEN
# =========================================================================
if st.session_state.ticker is None:
    # 🔍 Search bar
    query = st.text_input("🔍 종목 검색", placeholder="삼성전자, SK하이닉스, NAVER, 카카오 등 종목명을 입력하세요")
    
    if query:
        result = api_get("/search", {"q": query})
        if result and result["results"]:
            options = {f"{r['name']} ({r['ticker']})": r for r in result["results"]}
            picked = st.selectbox("검색 결과", list(options.keys()))
            if st.button("🎯 이 종목 선택"):
                chosen = options[picked]
                st.session_state.ticker = chosen["ticker"]
                st.session_state.ticker_name = chosen["name"]
                st.session_state.selected_event_date = None
                st.rerun()
        elif result:
            st.warning("검색 결과가 없습니다. (삼성전자, SK하이닉스, NAVER, 카카오, 현대차 등 데모 가능)")

    st.markdown("---")
    
    left_col, right_col = st.columns([6, 4])
    
    with left_col:
        st.markdown("### 📈 오늘의 인기 종목 TOP10")
        
        # Load prices for top stocks
        if st.session_state.top_stocks_cache is None:
            with st.spinner("인기 종목 실시간 시세 로드 중..."):
                top_stocks = []
                for stock in TRACKED_STOCKS:
                    p_data = api_get(f"/prices/{stock['ticker']}", {"period": "1m"})
                    if p_data and p_data["data"]:
                        df_p = pd.DataFrame(p_data["data"])
                        if len(df_p) >= 2:
                            last_close = float(df_p["Close"].iloc[-1])
                            prev_close = float(df_p["Close"].iloc[-2])
                            pct_change = ((last_close - prev_close) / prev_close) * 100
                        else:
                            last_close = float(df_p["Close"].iloc[-1])
                            pct_change = 0.0
                        top_stocks.append({
                            "ticker": stock["ticker"],
                            "name": stock["name"],
                            "price": last_close,
                            "change": pct_change
                        })
                    else:
                        top_stocks.append({
                            "ticker": stock["ticker"],
                            "name": stock["name"],
                            "price": 0.0,
                            "change": 0.0
                        })
                st.session_state.top_stocks_cache = top_stocks
        
        top_stocks = st.session_state.top_stocks_cache
        for i, stock in enumerate(top_stocks):
            c_color = "red" if stock["change"] >= 0 else "blue"
            c_sign = "+" if stock["change"] >= 0 else ""
            
            col_name, col_price, col_btn = st.columns([4, 4, 2])
            col_name.markdown(f"**{i+1}. {stock['name']}** <small style='color:grey;'>{stock['ticker']}</small>", unsafe_allow_html=True)
            col_price.markdown(f"<span style='color:{c_color}; font-weight:600;'>{stock['price']:,.0f}원 ({c_sign}{stock['change']:.2f}%)</span>", unsafe_allow_html=True)
            if col_btn.button("상세", key=f"btn_stock_{stock['ticker']}", use_container_width=True):
                st.session_state.ticker = stock["ticker"]
                st.session_state.ticker_name = stock["name"]
                st.session_state.selected_event_date = None
                st.rerun()

    with right_col:
        st.markdown("### 📰 오늘의 주요 뉴스 & HOT 종목")
        hot_news = api_get("/market/hot-stocks")
        if hot_news and hot_news.get("stocks"):
            for stock in hot_news["stocks"][:5]:
                c_pct = stock.get("change_pct", 0.0)
                c_sign = "+" if c_pct >= 0 else ""
                c_color = "#ef4444" if c_pct >= 0 else "#3b82f6"
                st.markdown(f"""
                <div class="news-card">
                    <span style="background-color:#eff6ff; color:#3b82f6; font-size:11px; font-weight:600; padding:2px 6px; border-radius:4px; margin-right:8px;">HOT</span>
                    <b>{stock['name']}</b> <small style="color:{c_color}; font-weight:600;">({c_sign}{c_pct:.2f}%)</small> 에 시장 이목 집중
                    <br><small style="color:grey;">뉴스 언급 {stock.get('mentions', 0)}회 • 관심도 점수 {stock.get('hot_score', 0):.1f}점</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("실시간 시장 뉴스 언급 데이터를 수집 중입니다.")
            
        st.markdown("### 🔥 AI 추천 관심 종목")
        rec_data = api_get("/market/recommendations")
        if rec_data and rec_data.get("recommendations"):
            for rec in rec_data["recommendations"]:
                st.markdown(f"""
                <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 12px; margin-bottom: 10px;">
                    <span style="color:#16a34a; font-weight:700; font-size:14px;">📈 {rec['name']} ({rec['pct']})</span>
                    <br><small style="color:#15803d;">{rec['reason']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("실시간 AI 추천 종목을 분석 중입니다.")
            
    st.stop()

# =========================================================================
# STOCK DETAIL PAGE (Toss Style)
# =========================================================================
ticker = st.session_state.ticker
ticker_name = st.session_state.ticker_name
period = st.session_state.period

# Fetch basic data
price_result = api_get(f"/prices/{ticker}", {"period": period})
cp_result = api_get(f"/changepoints/{ticker}", {"period": period if period != "1d" else "1m"})

if not price_result or not price_result.get("data"):
    st.warning("주가 데이터를 불러오지 못했습니다.")
    st.stop()

price_df = pd.DataFrame(price_result["data"])
price_df["date"] = pd.to_datetime(price_df["date"])
changepoints = cp_result["changepoints"] if cp_result else []

# Compute current stats
last_row = price_df.iloc[-1]
prev_row = price_df.iloc[-2] if len(price_df) > 1 else last_row
curr_price = float(last_row["Close"])
prev_price = float(prev_row["Close"])
price_change = curr_price - prev_price
price_change_pct = (price_change / prev_price) * 100
volume = int(last_row["Volume"])

color = "green" if price_change >= 0 else "blue"
sign = "+" if price_change >= 0 else ""

# ----------------- 1. Upper Area (Header) -----------------
st.markdown(f"""
<div style="background-color: #f9fafb; padding: 20px; border-radius: 16px; margin-bottom: 20px; border: 1px solid #e5e7eb;">
    <div style="font-size: 1.2rem; color: #6b7280; font-weight: 500;">{ticker_name} <small style='color:grey;'>{ticker}</small></div>
    <div style="display: flex; align-items: baseline; gap: 15px; margin-top: 5px;">
        <span class="metric-value">{curr_price:,.0f}원</span>
        <span style="color: {color}; font-size: 1.3rem; font-weight: 600;">{sign}{price_change:,.0f}원 ({sign}{price_change_pct:.2f}%)</span>
    </div>
    <div style="display: flex; gap: 40px; margin-top: 15px; font-size: 0.9rem; color: #4b5563;">
        <span>거래량 <b>{volume:,.0f}주</b></span>
        <span>기준일자 <b>{last_row['date'].strftime('%Y-%m-%d')}</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Period Selector Bar
p_cols = st.columns(len(PERIOD_OPTIONS))
for idx, (label, code) in enumerate(PERIOD_OPTIONS):
    if p_cols[idx].button(label, use_container_width=True, key=f"btn_p_{code}",
                          type="primary" if period == code else "secondary"):
        st.session_state.period = code
        st.session_state.selected_event_date = None
        st.rerun()

# ----------------- Load AI prediction data ahead of time -----------------
# 2-step model calls
pred_1d = api_get(f"/predict/{ticker}", {"horizon": "1d"}) or {}
pred_1w = api_get(f"/predict/{ticker}", {"horizon": "1w"}) or {}
pred_1m = api_get(f"/predict/{ticker}", {"horizon": "1m"}) or {}
analysis = api_get(f"/analyze/{ticker}", {"horizon": "1w"}) or {}
comm_posts = api_get(f"/market/community/{ticker}") or {}

# ----------------- 2. Columns (Left 70% | Right 30%) -----------------
left_col, right_col = st.columns([7, 3])

with left_col:
    # A. Price Chart + AI Prediction
    fig = go.Figure()
    
    # Solid price line
    fig.add_trace(go.Scatter(
        x=price_df["date"], y=price_df["Close"], mode="lines",
        name="실제 가격", line=dict(color="#2563eb", width=2.5),
    ))
    
    # Changepoints (Stars)
    if changepoints:
        cp_df = pd.DataFrame(changepoints)
        cp_df["date"] = pd.to_datetime(cp_df["date"], utc=True).dt.tz_localize(None)
        price_df["date"] = pd.to_datetime(price_df["date"], utc=True).dt.tz_localize(None)
        
        merged_cp = cp_df.merge(price_df[["date", "Close"]], on="date", how="left").dropna(subset=["Close"])
        fig.add_trace(go.Scatter(
            x=merged_cp["date"], y=merged_cp["Close"], mode="markers",
            name="변곡점", marker=dict(size=14, symbol="star",
                                     color=["#ef4444" if d == "up" else "#3b82f6" for d in merged_cp["direction"]]),
            text=merged_cp["reason"], hovertemplate="%{x}<br>%{text}<extra></extra>",
        ))

    # Dotted Prediction line (1-week future path)
    if pred_1w and "predicted_close" in pred_1w:
        last_date = price_df["date"].iloc[-1]
        future_date = last_date + pd.Timedelta(days=7)
        fig.add_trace(go.Scatter(
            x=[last_date, future_date],
            y=[price_df["Close"].iloc[-1], pred_1w["predicted_close"]],
            mode="lines+markers", name="AI 예측(점선)",
            line=dict(color="#f43f5e", width=2, dash="dash"),
            marker=dict(size=10, symbol="diamond"),
        ))

    fig.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10),
                      hovermode="x unified", xaxis_title="날짜", yaxis_title="가격(원)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 🎯 변곡점 선택 (AI 기술적 분석 & 회고 시뮬레이션)
    if changepoints:
        options = {f"{c['date']} ({c['pct_change']:+.1f}%, {c['reason']})": c for c in changepoints}
        picked_event = st.selectbox("🎯 관심 변곡점을 선택하면 AI 기술적 원인 및 회고 시뮬레이션을 분석합니다", list(options.keys()), key=f"sb_cp_{ticker}")
        if st.button("🔍 이 시점 AI 기술적 회고 분석 보기", key=f"btn_cp_{ticker}"):
            st.session_state.selected_event_date = options[picked_event]["date"]

        # ----------------- AI 변곡점 심층 기술적 분석 & 회고 시뮬레이션 결과 -----------------
        if st.session_state.selected_event_date:
            event_date = st.session_state.selected_event_date
            matched = next((c for c in changepoints if c["date"] == event_date), None)
            price_change = matched["pct_change"] if matched else 0.0

            st.markdown("---")
            st.subheader(f"🔍 {event_date} AI 변곡점 심층 기술적 분석 & 회고 시뮬레이션")
            
            with st.spinner("AI가 시계열 데이터 및 기술적 지표를 심층 분석 중입니다..."):
                detail = api_get(f"/events/{ticker}/detail",
                                  {"date": event_date, "price_change_pct": price_change})

            if detail:
                # 📊 1. 변곡점 요약 카드
                st.info(f"📊 **변곡점 요약 ({event_date})**: 5일 등락률 `{detail.get('price_change_pct', 0.0):+.1f}%` | AI 진단: **{detail.get('ai_judgement', '기술적 변곡')}**")

                c1, c2 = st.columns([6, 4])
                
                with c1:
                    # ① AI 변곡 원인 분석
                    st.markdown("### 🔍 AI 변곡 원인 분석")
                    for cause in detail.get("causes", []):
                        st.markdown(f"- **{cause}**")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # ② Feature 변화 표 (T-1 vs T)
                    st.markdown("### 📋 Feature 변화 비교표 (전일 T-1 vs 당일 T)")
                    ft = detail.get("feature_table", [])
                    if ft:
                        ft_df = pd.DataFrame(ft)
                        ft_df.columns = ["Feature", "전일 (T-1)", "당일 (T)", "변화 / 신호"]
                        st.table(ft_df)
                        
                with c2:
                    # ⭐ 변곡점 기술적 신호 강도 (Signal Strength)
                    conf = detail.get("confidence", {})
                    st.markdown("### 📊 기술적 신호 강도")
                    st.markdown(f"""
                    <div style="background: #f8fafc; border-radius: 12px; padding: 16px; border: 1px solid #cbd5e1; text-align: center; margin-bottom: 15px;">
                        <div style="font-size: 1.8rem; font-weight: 800; color: #1e3a8a;">{conf.get('stars', '★★★★★')}</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #2563eb; margin: 4px 0;">신호 강도 {conf.get('score', 90)} / 100점</div>
                        <div style="font-size: 0.85rem; color: #64748b;">주요 기술적 지표 복합 포착 점수 (지표 70% + 수급 30%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 🤖 AI 기술적 분석 코멘트
                    st.markdown("### 🤖 AI 기술적 분석 코멘트")
                    st.info(detail.get("ai_comment", "해당 변곡점에 대한 AI 기술적 분석 코멘트입니다."))

                st.markdown("<br>", unsafe_allow_html=True)

                ai_replay = detail.get("ai_replay", {})
                st.markdown("### 📈 AI Historical Replay (회고 시뮬레이션 및 오차 검증)")
                st.caption("💡 *당시 데이터를 현재 trained AI 모델에 입력하여 도출한 회고 예측과 실제 발생 주가 추이를 실증 대조합니다.*")

                score_val = ai_replay.get('bullish_score', 50)
                exp_pct = ai_replay.get('expected_pct', 0.0)
                act_pct = ai_replay.get('actual_pct', 0.0)
                is_hit = ai_replay.get('is_hit', False)
                err_val = ai_replay.get('error_pct', 0.0)
                sentiment_label = detail.get('sentiment', 'Neutral')

                exp_col = "#22c55e" if exp_pct >= 0 else "#3b82f6"
                exp_arrow = "▲" if exp_pct >= 0 else "▼"
                act_col = "#22c55e" if act_pct >= 0 else "#3b82f6"
                act_arrow = "▲" if act_pct >= 0 else "▼"
                
                hit_badge = "✅ 적중" if is_hit else "❌ 미적중"
                hit_bg = "#f0fdf4" if is_hit else "#fff5f5"
                hit_border = "#bbf7d0" if is_hit else "#fecaca"
                hit_color = "#16a34a" if is_hit else "#dc2626"

                st.markdown(f"""
                <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; margin-bottom: 15px;">
                    <div style="flex: 1; min-width: 130px; background: #f8fafc; border-radius: 12px; padding: 12px 16px; border: 1px solid #e2e8f0; text-align: center;">
                        <div style="font-size: 0.82rem; color: #64748b; font-weight: 600;">Replay 종합 점수</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #1e293b; margin: 4px 0;">{score_val} / 100점</div>
                        <div style="font-size: 0.78rem; color: #64748b; font-weight: 600;">{sentiment_label}</div>
                    </div>
                    <div style="flex: 1; min-width: 130px; background: #f8fafc; border-radius: 12px; padding: 12px 16px; border: 1px solid #e2e8f0; text-align: center;">
                        <div style="font-size: 0.82rem; color: #64748b; font-weight: 600;">당시 AI 예상 5일</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: {exp_col}; margin: 4px 0;">{exp_arrow} {exp_pct:+.1f}%</div>
                    </div>
                    <div style="flex: 1; min-width: 130px; background: #f8fafc; border-radius: 12px; padding: 12px 16px; border: 1px solid #e2e8f0; text-align: center;">
                        <div style="font-size: 0.82rem; color: #64748b; font-weight: 600;">실제 발생 5일</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: {act_col}; margin: 4px 0;">{act_arrow} {act_pct:+.1f}%</div>
                    </div>
                    <div style="flex: 1; min-width: 130px; background: {hit_bg}; border-radius: 12px; padding: 12px 16px; border: 1px solid {hit_border}; text-align: center;">
                        <div style="font-size: 0.82rem; color: #64748b; font-weight: 600;">예측 적중 여부</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: {hit_color}; margin: 4px 0; white-space: nowrap;">{hit_badge}</div>
                        <div style="font-size: 0.78rem; color: {hit_color}; font-weight: 600;">오차: {err_val:.1f}%p</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 회고 오차 원인 분석 카드
                err_causes = ai_replay.get("error_causes", [])
                if err_causes:
                    st.markdown("##### 🧐 AI 회고 및 오차 원인 실증 분석")
                    for cause in err_causes:
                        st.markdown(f"- {cause}")

                # Plotly 비교 차트 오버레이
                dates = ai_replay.get("forecast_dates", [])
                f_prices = ai_replay.get("forecast_prices", [])
                a_prices = ai_replay.get("actual_prices", [])

                if dates and f_prices and a_prices:
                    fig_rep = go.Figure()
                    fig_rep.add_trace(go.Scatter(
                        x=dates, y=a_prices, mode="lines+markers", name="실제 주가 추이",
                        line=dict(color="#2563eb", width=3), marker=dict(size=8)
                    ))
                    fig_rep.add_trace(go.Scatter(
                        x=dates, y=f_prices, mode="lines+markers", name="AI 회고 예상 경로 (점선)",
                        line=dict(color="#f43f5e", width=2, dash="dash"), marker=dict(size=8, symbol="diamond")
                    ))
                    fig_rep.update_layout(
                        title=dict(text=f"[{event_date}] 변곡점 이후 AI 회고 예상 경로 vs 실제 주가 추이 대조", font=dict(size=14)),
                        height=320, margin=dict(t=40, b=10, l=10, r=10),
                        hovermode="x unified", xaxis_title="날짜", yaxis_title="주가 (원)"
                    )
                    st.plotly_chart(fig_rep, use_container_width=True)
            st.markdown("---")

    # B. AI Prediction Info Card (Toss Style)
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.15rem; font-weight:700; color:#1e3a8a;'>🔮 AI 2단계 예측 정보 패널</span>", unsafe_allow_html=True)
    
    c_pred1, c_pred2, c_pred3 = st.columns(3)
    if pred_1d and "predicted_change_pct" in pred_1d:
        p1_val = pred_1d['predicted_change_pct']
        p1_col = "#22c55e" if p1_val >= 0 else "#3b82f6"
        p1_arrow = "▲" if p1_val >= 0 else "▼"
        p1_sign = "+" if p1_val >= 0 else ""
        ci1 = pred_1d.get("confidence_interval", {})
        ci1_txt = f"<br><small style='color:grey;'>95% CI: [{ci1.get('lower_pct', 0.0):+.1f}% ~ {ci1.get('upper_pct', 0.0):+.1f}%]</small>" if ci1 else ""
        c_pred1.markdown(f"""
        <div style="text-align: center; background: #f8fafc; border-radius: 12px; padding: 12px; border: 1px solid #e2e8f0; min-height: 110px;">
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">1일 후 예상 수익률</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {p1_col}; margin: 4px 0;">{p1_arrow} {p1_sign}{p1_val:.2f}%</div>
            <div style="font-size: 1rem; color: #1e293b; font-weight: 600;">{pred_1d['predicted_close']:,.0f}원</div>
            {ci1_txt}
        </div>
        """, unsafe_allow_html=True)
        
    if pred_1w and "predicted_change_pct" in pred_1w:
        pw_val = pred_1w['predicted_change_pct']
        pw_col = "#22c55e" if pw_val >= 0 else "#3b82f6"
        pw_arrow = "▲" if pw_val >= 0 else "▼"
        pw_sign = "+" if pw_val >= 0 else ""
        ciw = pred_1w.get("confidence_interval", {})
        ciw_txt = f"<br><small style='color:grey;'>95% CI: [{ciw.get('lower_pct', 0.0):+.1f}% ~ {ciw.get('upper_pct', 0.0):+.1f}%]</small>" if ciw else ""
        c_pred2.markdown(f"""
        <div style="text-align: center; background: #f8fafc; border-radius: 12px; padding: 12px; border: 1px solid #e2e8f0; min-height: 110px;">
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">1주일 후 예상 수익률</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {pw_col}; margin: 4px 0;">{pw_arrow} {pw_sign}{pw_val:.2f}%</div>
            <div style="font-size: 1rem; color: #1e293b; font-weight: 600;">{pred_1w['predicted_close']:,.0f}원</div>
            {ciw_txt}
        </div>
        """, unsafe_allow_html=True)
        
    if pred_1m and "predicted_change_pct" in pred_1m:
        pm_val = pred_1m['predicted_change_pct']
        pm_col = "#22c55e" if pm_val >= 0 else "#3b82f6"
        pm_arrow = "▲" if pm_val >= 0 else "▼"
        pm_sign = "+" if pm_val >= 0 else ""
        cim = pred_1m.get("confidence_interval", {})
        cim_txt = f"<br><small style='color:grey;'>95% CI: [{cim.get('lower_pct', 0.0):+.1f}% ~ {cim.get('upper_pct', 0.0):+.1f}%]</small>" if cim else ""
        c_pred3.markdown(f"""
        <div style="text-align: center; background: #f8fafc; border-radius: 12px; padding: 12px; border: 1px solid #e2e8f0; min-height: 110px;">
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">1개월 후 예상 수익률</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {pm_col}; margin: 4px 0;">{pm_arrow} {pm_sign}{pm_val:.2f}%</div>
            <div style="font-size: 1rem; color: #1e293b; font-weight: 600;">{pred_1m['predicted_close']:,.0f}원</div>
            {cim_txt}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # ----------------- Low Confidence Warning Banner -----------------
    if pred_1w and pred_1w.get("confidence_warning_msg"):
        st.warning(pred_1w["confidence_warning_msg"])
    c_meta1, c_meta2, c_meta3 = st.columns(3)
    
    direction_display = {"UP": "상승 📈", "DOWN": "하락 📉", "FLAT": "횡보 ➡️"}
    dir_k = pred_1w.get("direction", "FLAT")
    conf = pred_1w.get("direction_confidence", 50.0)
    
    c_meta1.markdown(f"**방향 예측 판정**  \n<span style='font-size:1.3rem; font-weight:600;'>{direction_display.get(dir_k, dir_k)} ({conf:.1f}%)</span>", unsafe_allow_html=True)
    
    m_info = pred_1w.get("model_info", {})
    m_ver = m_info.get("version", "v4")
    m_r2 = float(m_info.get("r2", 0.0))
    
    if m_r2 < 0:
        r2_badge = f"<span style='font-size:1.2rem; font-weight:600; color:#ef4444;'>낮음 ({m_r2:+.4f})</span>"
    elif m_r2 < 0.3:
        r2_badge = f"<span style='font-size:1.2rem; font-weight:600; color:#f59e0b;'>보통 ({m_r2:+.4f})</span>"
    else:
        r2_badge = f"<span style='font-size:1.2rem; font-weight:600; color:#10b981;'>우수 ({m_r2:+.4f})</span>"

    c_meta2.markdown(f"**AI 모델 버전**  \n<span style='font-size:1.2rem; font-weight:600;'>{m_ver}</span>", unsafe_allow_html=True)
    c_meta3.markdown(f"**모델 평가 품질 (R²)**  \n{r2_badge}", unsafe_allow_html=True)

    # ----------------- SHAP Feature Contribution Chart & Table -----------------
    st.markdown("<h5 style='margin-top: 18px;'>🎯 AI 예측 근거 (SHAP 피처 기여도 분해)</h5>", unsafe_allow_html=True)
    st.caption("AI 모델이 상승/하락 예측 판단 시 가장 결정적 영향을 미친 주요 요인 분석")

    contributions = (
        pred_1w.get("shap_contributions")
        or pred_1w.get("features_used", {}).get("feature_contributions")
        or pred_1w.get("features_used", {}).get("feature_importance")
        or {}
    ) if pred_1w else {}

    display_names = {
        "return_1d": "일일 수익률",
        "return_5d": "5일 누적 수익률",
        "return_20d": "20일 누적 수익률",
        "rsi14": "RSI(14) 지표",
        "macd": "MACD 수치",
        "macd_signal": "MACD Signal",
        "macd_hist": "MACD 오실레이터",
        "bb_position": "볼린저 밴드 위치",
        "volume_zscore": "거래량 Z-Score",
        "volume_ratio": "거래량 비율",
        "volatility_5": "5일 변동성",
        "volatility_20": "20일 변동성",
        "news_sentiment": "뉴스 감성 지표",
        "news_sentiment_3d": "3일 뉴스 감성 평균",
        "community_sentiment": "커뮤니티 심리 지표",
        "usdkrw_return_1d": "원/달러 환율 변동",
        "usdkrw_return_5d": "원/달러 환율 5일 변동",
        "kospi_return_1d": "코스피 지수 변동",
        "kospi_return_5d": "코스피 5일 변동",
        "vix_return_1d": "VIX 변동성 지수",
        "prev_close_pct_change": "전일 대비 변동률",
        "ma5_gap": "5일 이동평균 이격도",
        "ma20_gap": "20일 이동평균 이격도",
        "ma60_gap": "60일 이동평균 이격도",
    }

    if contributions:
        # Sort contributions by absolute magnitude top 7
        sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:7]
        
        # 1. 📈 기여도 바 차트 (Bar Chart)
        y_labels = [display_names.get(k, k) for k, v in reversed(sorted_items)]
        x_vals = [v for k, v in reversed(sorted_items)]
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in x_vals]

        import plotly.graph_objects as go
        fig_shap = go.Figure(go.Bar(
            x=x_vals,
            y=y_labels,
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.2f}%p" for v in x_vals],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>예측 기여도: %{x:+.2f}%p<extra></extra>"
        ))
        fig_shap.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="예측 수익률 기여도 (%p)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # 2. 📊 피처 기여도 테이블 (Factor, Contribution, Direction Table)
        st.markdown("##### 📊 피처 기여도 상세 데이터 테이블")
        table_rows = []
        for k, v in sorted_items:
            fname = display_names.get(k, k)
            direction_label = "🟢 상승 유인" if v > 0 else ("🔴 하락 압력" if v < 0 else "⚪ 중립")
            table_rows.append({
                "주요 요인": fname,
                "수익률 기여도 (%p)": f"{v:+.2f}%p",
                "영향 방향": direction_label
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("피처 기여도 데이터를 분석하는 중입니다.")

    # ----------------- Step 4: What-If Simulator Expander -----------------
    with st.expander("🎛️ What-If 가상 시뮬레이터 (조건 변경 시 예측 변화 테스트)"):
        st.caption("주요 지표(뉴스 감성, 환율, RSI 등)의 가상 조건을 조절할 때 AI 모델 예측이 어떻게 변하는지 테스트합니다.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sim_news = st.slider("뉴스 감성 지표", min_value=-1.0, max_value=1.0, value=0.2, step=0.1, key=f"sim_news_{ticker}")
            sim_rsi = st.slider("RSI 지표 (14일)", min_value=10.0, max_value=90.0, value=45.0, step=1.0, key=f"sim_rsi_{ticker}")
        with col_s2:
            sim_usd = st.slider("원/달러 환율 1일 변동률(%)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1, key=f"sim_usd_{ticker}")
            sim_vol = st.slider("거래량 비율 (평균 대비)", min_value=0.5, max_value=3.0, value=1.0, step=0.1, key=f"sim_vol_{ticker}")

        if st.button("🚀 시뮬레이션 예측 실행", key=f"btn_sim_{ticker}", use_container_width=True):
            overrides = {
                "news_sentiment": sim_news,
                "rsi14": sim_rsi,
                "usdkrw_return_1d": sim_usd / 100.0,
                "volume_ratio": sim_vol
            }
            with st.spinner("가상 조건 시뮬레이션 재계산 중..."):
                sim_res = api_post(f"/predict/{ticker}/simulate", {"horizon": "1w", "overrides": overrides})
            
            if sim_res and "sim_predicted_change_pct" in sim_res:
                base_p = sim_res["base_predicted_change_pct"]
                sim_p = sim_res["sim_predicted_change_pct"]
                diff_p = sim_res["diff_pct"]
                
                diff_col = "#22c55e" if diff_p >= 0 else "#ef4444"
                diff_sign = "+" if diff_p >= 0 else ""
                
                st.markdown(f"""
                <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 12px; margin-top: 10px;">
                    <div style="font-weight: 700; color: #1e293b; font-size: 1rem;">📊 시뮬레이션 예측 결과</div>
                    <div style="margin-top: 6px; font-size: 0.95rem;">
                        • 기존 1주일 예측: <b>{base_p:+.2f}%</b><br>
                        • 시뮬레이션 예측: <b>{sim_p:+.2f}%</b> ({sim_res.get('sim_predicted_close', 0):,.0f}원)<br>
                        • 변동 효과: <span style="color: {diff_col}; font-weight: 700;">{diff_sign}{diff_p:.2f}%p</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("시뮬레이션 실행에 실패했습니다.")

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # 🤖 AI 종합 분석
    st.markdown("#### 🤖 AI 종합 전망 (신뢰도 가중 복합 산출)")
    if analysis and "analysis" in analysis:
        ai_res = analysis["analysis"]

        outlook_raw = ai_res.get("outlook", "Neutral")
        score_val = ai_res.get("score", 50)
        w_b = ai_res.get("weights_breakdown", {})

        # Outlook mapping
        if outlook_raw == "Strong Bullish":
            outlook_kr, outlook_bg, outlook_fg = "🚀 강력 상승 (Strong Bullish)", "#d1fae5", "#065f46"
        elif outlook_raw == "Bullish":
            outlook_kr, outlook_bg, outlook_fg = "📈 상승 우세 (Bullish)", "#ecfdf5", "#047857"
        elif outlook_raw == "Bearish":
            outlook_kr, outlook_bg, outlook_fg = "📉 하락 우세 (Bearish)", "#fff7ed", "#c2410c"
        elif outlook_raw == "Strong Bearish":
            outlook_kr, outlook_bg, outlook_fg = "🔴 강력 하락 (Strong Bearish)", "#fee2e2", "#991b1b"
        else:
            outlook_kr, outlook_bg, outlook_fg = "⏸️ 관망 / 횡보 (Neutral)", "#eff6ff", "#1d4ed8"

        st.markdown(f"""
        <div style="background-color: {outlook_bg}; color: {outlook_fg}; border: 1.5px solid {outlook_fg}; border-radius: 12px; padding: 14px 16px; margin-bottom: 15px;">
            <div style="font-size: 0.82rem; font-weight: 600; text-transform: uppercase;">AI 종합 투자의견 판정</div>
            <div style="font-size: 1.45rem; font-weight: 800; margin-top: 2px;">{outlook_kr}</div>
            <div style="font-size: 0.95rem; font-weight: 700; margin-top: 4px;">AI 산출 점수: <span style="font-size: 1.3rem;">{score_val}</span> / 100점</div>
        </div>
        """, unsafe_allow_html=True)

        # Signal Conflict Notice Banner
        if ai_res.get("is_conflict") and ai_res.get("conflict_msg"):
            st.warning(ai_res["conflict_msg"])

        # Dynamic Weights Breakdown
        if w_b:
            with st.expander("📊 신뢰도 기반 동적 가중치 비율 보기"):
                st.markdown(f"- **ML 모델 예측**: `{w_b.get('ml_weight', 0)}%` (R² 신뢰도 반영)")
                st.markdown(f"- **뉴스 3D 영향도**: `{w_b.get('news_weight', 0)}%` (관련도·파급력 반영)")
                st.markdown(f"- **기술적 지표 (MACD/RSI/볼린저)**: `{w_b.get('tech_weight', 0)}%`")
                st.markdown(f"- **커뮤니티 투자 의도**: `{w_b.get('comm_weight', 0)}%`")
                st.markdown(f"- **과거 유사 차트 패턴**: `{w_b.get('pattern_weight', 0)}%`")

        # Reasons
        st.markdown("##### 💡 주요 산출 근거")
        for reason in ai_res.get("reasons", []):
            st.info(reason)

        # AI Investment Report Button
        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        if st.button("📄 AI 4단계 심층 리포트 생성", key=f"btn_ai_report_{ticker}", use_container_width=True):
            with st.spinner("AI 수석 전략가가 4단계 분석 리포트를 작성 중입니다..."):
                report_data = api_get(f"/analyze/report/{ticker}", {"horizon": "1w"})
            if report_data and report_data.get("analysis", {}).get("llm_report"):
                st.markdown("---")
                st.markdown("##### 📄 AI 투자 리포트")
                st.markdown(report_data["analysis"]["llm_report"])
            else:
                st.error("리포트 작성에 실패했습니다.")
    else:
        st.info("종합 분석 데이터를 로드할 수 없습니다.")


st.markdown("---")

# ----------------- Historical Pattern Matching Section -----------------
st.markdown("### 📈 과거 유사 패턴 매칭 & 예측 트랙 레코드 (AI 예측 근거)")
st.caption("현재 종목의 기술적/감성지표 14일 패턴과 가장 유사했던 과거 3개 구간 및 이후 실제 주가 궤적을 비교 분석합니다.")

pattern_data = api_get(f"/market/similar-patterns/{ticker}")

if pattern_data and pattern_data.get("matches"):
    st.info(pattern_data["summary"])
    
    curr_w = pattern_data.get("current_window", {})
    matches = pattern_data.get("matches", [])
    
    import plotly.graph_objects as go
    fig_pattern = go.Figure()
    
    # Current 14 days curve
    if curr_w and "prices" in curr_w:
        x_curr = list(range(-len(curr_w["prices"]) + 1, 1))
        fig_pattern.add_trace(go.Scatter(
            x=x_curr,
            y=curr_w["prices"],
            mode='lines+markers',
            name='현재 14일 추이',
            hovertemplate="<b>[현재 14일 추이]</b><br>경과: Day %{x}<br>기준 지수: %{y:.2f}<extra></extra>",
            line=dict(color='#2563eb', width=3.5),
            marker=dict(size=6)
        ))
        
    palette = ["#16a34a", "#d97706", "#9333ea"]
    
    for idx, m in enumerate(matches):
        c_color = palette[idx % len(palette)]
        x_matched = list(range(-len(m["matched_prices"]) + 1, 1))
        
        # Matched past 14 days curve
        fig_pattern.add_trace(go.Scatter(
            x=x_matched,
            y=m["matched_prices"],
            mode='lines',
            name=f'과거 {idx+1}위 ({m["similarity_pct"]}% 유사)',
            hovertemplate=f"<b>[과거 {idx+1}위 매칭 ({m['similarity_pct']}% 유사)]</b><br>기간: {m['start_date']} ~ {m['end_date']}<br>경과: Day %{{x}}<br>기준 지수: %{{y:.2f}}<extra></extra>",
            line=dict(color=c_color, width=2, dash='dash')
        ))
        
        # Outcome next 14 days curve
        x_future = list(range(0, len(m["future_prices"])))
        c_sign = "+" if m["actual_return_14d"] >= 0 else ""
        fig_pattern.add_trace(go.Scatter(
            x=x_future,
            y=m["future_prices"],
            mode='lines+markers',
            name=f'➔ 과거 {idx+1}위 결과 ({c_sign}{m["actual_return_14d"]}%)',
            hovertemplate=f"<b>[과거 {idx+1}위 결과 ({c_sign}{m['actual_return_14d']}%)]</b><br>경과: Day +%{{x}}<br>기준 지수: %{{y:.2f}}<extra></extra>",
            line=dict(color=c_color, width=2.5),
            marker=dict(size=5)
        ))

    # Add reference line at Day 0
    fig_pattern.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#64748b", annotation_text="현재 시점 (Day 0)", annotation_position="top left")

    fig_pattern.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="경과 일수 (Day 0 = 현재 / 과거 패턴 매칭 분기점)",
        yaxis_title="기준 주가 지수 (Day -13 = 100 기준)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="sans-serif",
            bordercolor="#cbd5e1"
        ),
        hovermode="closest",
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9', dtick=2),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    
    st.plotly_chart(fig_pattern, use_container_width=True)
    
    # Render TOP K match cards
    m_cols = st.columns(len(matches))
    for idx, (m, col) in enumerate(zip(matches, m_cols)):
        ret_val = m["actual_return_14d"]
        ret_color = "#16a34a" if ret_val >= 0 else "#dc2626"
        ret_icon = "🟢" if ret_val >= 0 else "🔴"
        ret_sign = "+" if ret_val >= 0 else ""
        
        col.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">과거 매칭 {idx+1}위 ({m['similarity_pct']}% 유사)</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #334155; margin: 4px 0;">{m['start_date']} ~ {m['end_date']}</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: {ret_color};">{ret_icon} 2주 후 {ret_sign}{ret_val:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("과거 유사 패턴 데이터를 수집 및 비교 분석 중입니다.")

st.markdown("---")

# ----------------- 3. Middle Area (News 3D Impact & AI Analysis) -----------------
st.markdown("### 📰 뉴스 영향도 & 투자 심리 AI 분석 (3차원 평가 파이프라인)")
news_col, news_ai_col = st.columns([6, 4])

with news_col:
    selected_date = price_df["date"].iloc[-1].strftime("%Y-%m-%d")
    news_data = api_get(f"/market/news/{ticker}", {"date": selected_date})

    top_arts = news_data.get("top_impact_articles", []) if news_data else []
    all_arts = news_data.get("articles", []) if news_data else []
    display_arts = top_arts[:3] if top_arts else all_arts[:3]

    if news_data and display_arts:
        # 1. 주가 영향력이 가장 높은 핵심 뉴스 Top 3 표출
        st.markdown("##### 🔥 주가 영향력이 가장 큰 핵심 뉴스 (관련도 × 영향도 가중 평가)")
        for art in display_arts:
            sent = art.get("sentiment", "Neutral")
            sent_bg = "#d1fae5" if sent == "Bullish" else ("#fee2e2" if sent == "Bearish" else "#f3f4f6")
            sent_fg = "#065f46" if sent == "Bullish" else ("#991b1b" if sent == "Bearish" else "#374151")
            sent_icon = "🚀 Bullish" if sent == "Bullish" else ("📉 Bearish" if sent == "Bearish" else "⏸️ Neutral")
            score_val = art.get("weighted_score", 0)

            st.markdown(f"""
            <div style="padding: 10px 12px; margin-bottom: 8px; border-radius: 8px; border: 1px solid #e5e7eb; background-color: #fafafa;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div>
                        <span style="background-color: {sent_bg}; color: {sent_fg}; padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; font-weight: 700;">{sent_icon}</span>
                        <span style="background-color: #3b82f6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 4px;">점수 {score_val:+d}점</span>
                    </div>
                    <small style="color: #4b5563;">관련도 <b style="color:#d97706;">{art.get('relevance_stars','')}</b> | 영향도 <b style="color:#dc2626;">{art.get('impact_stars','')}</b></small>
                </div>
                <div style="font-weight: 600; font-size: 0.93rem; margin-top: 4px;">
                    <a href="{art['link']}" target="_blank" style="text-decoration: none; color: #1e40af;">{art['title']}</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. 일반 뉴스 기사 목록
        with st.expander("📋 전체 수집 뉴스 목록 보기"):
            for art in (all_arts or display_arts)[:7]:
                st.markdown(f"- [{art['title']}]({art['link']}) <small style='color:grey;'>({art.get('pub_date','')})</small>", unsafe_allow_html=True)

        # 3. AI 이슈 요약 표기
        if news_data.get("summary"):
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            st.info(f"💡 **[AI 뉴스 흐름 종합 요약]**\n\n{news_data['summary']}")
    else:
        st.info("최근 관련 뉴스 기사가 없습니다.")


with news_ai_col:
    if news_data and "intent_percentages" in news_data:
        n_pcts = news_data["intent_percentages"]
        b_pct = float(n_pcts.get("Bullish", 0.0))
        be_pct = float(n_pcts.get("Bearish", 0.0))
        neu_pct = float(n_pcts.get("Neutral", 0.0))
        n_stance = news_data.get("overall_stance", "Neutral")

        st.markdown("##### 📊 3차원 가중 뉴스 투자 심리 비중")
        st.markdown(f"📌 **뉴스 종합 판단**: <span style='font-size:1.05rem; font-weight:700; color:#2563eb;'>{n_stance}</span>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:0.85rem; font-weight:600; color:#065f46;'>🚀 Bullish (상승 모멘텀 가중치) : <b>{b_pct:.1f}%</b></div>", unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, b_pct / 100.0)))

        st.markdown(f"<div style='font-size:0.85rem; font-weight:600; color:#991b1b;'>📉 Bearish (하락 리스크 가중치) : <b>{be_pct:.1f}%</b></div>", unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, be_pct / 100.0)))

        st.markdown(f"<div style='font-size:0.85rem; font-weight:600; color:#374151;'>⏸️ Neutral (중립/관망 가중치) : <b>{neu_pct:.1f}%</b></div>", unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, neu_pct / 100.0)))

        w_sent = news_data.get("news_sentiment", 0.0)
        st.markdown(f"**실시간 가중 감성 지표 :** <span style='font-size:1.15rem; font-weight:700; color:#10b981;'>{w_sent:+.2f}</span>", unsafe_allow_html=True)

        if news_data.get("bullish_reasons") or news_data.get("bearish_reasons"):
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            if news_data.get("bullish_reasons"):
                st.markdown("**🚀 주요 상승 호재 요인:**")
                for r in news_data["bullish_reasons"]:
                    st.markdown(f"- {r}")
            if news_data.get("bearish_reasons"):
                st.markdown("**📉 주요 하락 악재 요인:**")
                for r in news_data["bearish_reasons"]:
                    st.markdown(f"- {r}")

        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        if st.button("🤖 LLM 뉴스 영향도 3D 심층 분석", key=f"btn_news_llm_{ticker}", use_container_width=True):
            with st.spinner("AI 모델이 관련도·영향도 가중 뉴스를 심층 분석 중입니다..."):
                news_llm_res = api_get(f"/market/news_sentiment_analysis/{ticker}")
            if news_llm_res and news_llm_res.get("llm_analysis"):
                st.info(news_llm_res['llm_analysis'])
            else:
                st.error("LLM 뉴스 분석을 불러올 수 없습니다.")


st.markdown("---")

# ----------------- 4. Bottom Area (Community Intent & AI Hybrid Pipeline) -----------------
st.markdown("### 💬 커뮤니티 반응 & 투자 의도 AI 분석 (사전 + LLM 혼합 파이프라인)")
comm_col, comm_ai_col = st.columns([6, 4])

# Intent styling helper mapping
INTENT_STYLES = {
    "Bullish": {"bg": "#d1fae5", "fg": "#065f46", "label": "🚀 Bullish (매수)", "icon": "🟢"},
    "Bearish": {"bg": "#fee2e2", "fg": "#991b1b", "label": "📉 Bearish (매도)", "icon": "🔴"},
    "Neutral": {"bg": "#f3f4f6", "fg": "#374151", "label": "⏸️ Neutral (관망)", "icon": "⚪"},
    "Question": {"bg": "#dbeafe", "fg": "#1e40af", "label": "❓ Question (질문)", "icon": "🔵"},
    "Humor": {"bg": "#f3e8ff", "fg": "#6b21a8", "label": "🤡 Humor (유머/밈)", "icon": "🟣"},
    "News": {"bg": "#fef3c7", "fg": "#92400e", "label": "📰 News (공시/기사)", "icon": "🟡"},
}

with comm_col:
    if comm_posts and comm_posts.get("posts"):
        all_posts = comm_posts["posts"]
        total_posts_cnt = comm_posts.get('total_posts', len(all_posts))

        st.markdown(f"##### 📋 최근 토론방 게시글 목록 (총 {total_posts_cnt}개 게시글 분석)")
        
        # 드롭다운 필터 선택기
        filter_opt = st.selectbox(
            "🔍 투자 의도 카테고리 드롭다운 필터",
            ["전체 보기", "🚀 Bullish (매수)", "📉 Bearish (매도)", "⏸️ Neutral (관망)", "❓ Question (질문)", "🤡 Humor (유머)", "📰 News (공시/기사)"],
            key=f"sb_comm_filter_{ticker}"
        )

        # 필터링 적용
        if filter_opt.startswith("🚀"):
            filtered_posts = [p for p in all_posts if p.get("intent") == "Bullish"]
        elif filter_opt.startswith("📉"):
            filtered_posts = [p for p in all_posts if p.get("intent") == "Bearish"]
        elif filter_opt.startswith("⏸️"):
            filtered_posts = [p for p in all_posts if p.get("intent") == "Neutral"]
        elif filter_opt.startswith("❓"):
            filtered_posts = [p for p in all_posts if p.get("intent") == "Question"]
        elif filter_opt.startswith("🤡"):
            filtered_posts = [p for p in all_posts if p.get("intent") == "Humor"]
        elif filter_opt.startswith("📰"):
            filtered_posts = [p for p in all_posts if p.get("intent") == "News"]
        else:
            filtered_posts = all_posts

        def render_post_card(post):
            intent = post.get("intent", "Neutral")
            style = INTENT_STYLES.get(intent, INTENT_STYLES["Neutral"])
            method_tag = post.get("method_tag", "⚡사전")
            method_bg = "#eff6ff" if "LLM" in method_tag else "#fdf2f8"
            method_fg = "#1e40af" if "LLM" in method_tag else "#831843"
            
            st.markdown(f"""
            <div class="community-post" style="padding: 8px 10px; margin-bottom: 6px; border-radius: 8px; border: 1px solid #e5e7eb; background: #ffffff;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                    <div>
                        <span style="background-color: {method_bg}; color: {method_fg}; padding: 2px 6px; border-radius: 4px; font-size: 0.73rem; font-weight: 600;">{method_tag}</span>
                        <span style="background-color: {style['bg']}; color: {style['fg']}; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; margin-left: 4px;">{style['label']}</span>
                    </div>
                    <small style="color: #6b7280; font-size: 0.75rem;">{post.get('writer', '익명')} | {str(post.get('date', ''))[:16]}</small>
                </div>
                <div style="font-weight: 600; font-size: 0.88rem; color: #1f2937;">
                    {post.get('title', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 15개 이상인 경우 드롭다운으로 표기
        if len(filtered_posts) >= 15:
            # 개별 게시글 바로 선택 드롭다운
            post_options = [f"[{p.get('method_tag','⚡사전')}/{p.get('intent','Neutral')}] {p.get('title','')}" for p in filtered_posts]
            selected_single = st.selectbox(
                "📌 개별 게시글 드롭다운 선택 보기",
                ["-- 전체 목록 접기/펼치기 선택 --"] + post_options,
                key=f"sb_single_post_{ticker}"
            )
            
            if selected_single != "-- 전체 목록 접기/펼치기 선택 --":
                s_idx = post_options.index(selected_single)
                st.markdown("##### 🔍 선택한 게시글")
                render_post_card(filtered_posts[s_idx])
                st.markdown("<br>", unsafe_allow_html=True)

            with st.expander(f"📥 수집 게시글 드롭다운 목록 (총 {len(filtered_posts)}건 / 전체 {total_posts_cnt}건 - 클릭하여 펼치기)", expanded=False):
                st.markdown('<div style="max-height: 480px; overflow-y: auto; padding-right: 6px;">', unsafe_allow_html=True)
                for post in filtered_posts:
                    render_post_card(post)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # 15개 미만인 경우 화면에 직접 출력
            for post in filtered_posts:
                render_post_card(post)

        # 2. AI 커뮤니티 종합 요약 하단 카드
        if comm_posts.get("summary"):
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            st.info(f"💡 **[AI 종목토론방 분위기 요약]**\n\n{comm_posts['summary']}")
    else:
        st.info("최근 수집된 커뮤니티 토론 게시글 데이터가 없습니다.")

with comm_ai_col:
    intent_pcts = comm_posts.get("intent_percentages") if comm_posts else None
    if not intent_pcts and analysis and "community_analysis" in analysis:
        c_an = analysis["community_analysis"]
        if "intent_percentages" in c_an:
            intent_pcts = c_an["intent_percentages"]
        else:
            intent_pcts = {
                "Bullish": c_an.get("positive", 0.0),
                "Bearish": c_an.get("negative", 0.0),
                "Neutral": c_an.get("neutral", 0.0),
                "Question": 0.0,
                "Humor": 0.0,
                "News": 0.0
            }

    if intent_pcts:
        overall_stance = comm_posts.get("overall_stance", "Neutral") if comm_posts else "Neutral"
        st.markdown(f"##### 📊 투자 의도 6단계 비중 분포")
        st.markdown(f"📌 **종합 분위기**: <span style='font-size:1.05rem; font-weight:700; color:#2563eb;'>{overall_stance}</span>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

        # Render 6 intent category progress bars
        for cat_key in ["Bullish", "Bearish", "Neutral", "Question", "Humor", "News"]:
            pct = float(intent_pcts.get(cat_key, 0.0))
            style = INTENT_STYLES[cat_key]
            st.markdown(f"<div style='font-size:0.85rem; font-weight:600; color:#374151;'>{style['label']} : <b>{pct:.1f}%</b></div>", unsafe_allow_html=True)
            st.progress(min(1.0, max(0.0, pct / 100.0)))

        w_sent = comm_posts.get("weighted_sentiment", 0.0) if comm_posts else analysis.get("community_sentiment", 0.0)
        st.markdown(f"**실시간 수치화 감성 지표 :** <span style='font-size:1.15rem; font-weight:700; color:#10b981;'>{w_sent:+.2f}</span>", unsafe_allow_html=True)
        
        # Bullish / Bearish Reasons breakdown if available
        if comm_posts and (comm_posts.get("bullish_reasons") or comm_posts.get("bearish_reasons")):
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            if comm_posts.get("bullish_reasons"):
                st.markdown("**🚀 주요 상승 기대 요인:**")
                for r in comm_posts["bullish_reasons"]:
                    st.markdown(f"- {r}")
            if comm_posts.get("bearish_reasons"):
                st.markdown("**📉 주요 하락 우려 요인:**")
                for r in comm_posts["bearish_reasons"]:
                    st.markdown(f"- {r}")

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        if st.button("🤖 LLM 커뮤니티 심리 심층 분석", key=f"btn_comm_llm_{ticker}", use_container_width=True):
            with st.spinner("AI 모델이 커뮤니티 여론 데이터를 심층 분석 중입니다..."):
                comm_llm_res = api_get(f"/market/community_sentiment_analysis/{ticker}")
            if comm_llm_res and comm_llm_res.get("llm_analysis"):
                st.info(comm_llm_res['llm_analysis'])
            else:
                st.error("LLM 커뮤니티 분석을 불러올 수 없습니다.")
