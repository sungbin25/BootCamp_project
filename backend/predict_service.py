"""
미래 주가 예측 (UR-10)

MLflow Registry 및 Runs에 저장된 최신 모델을 로드하여 예측을 수행합니다.
저장된 모델이 없거나 로드에 실패할 경우, 즉석에서 Random Forest 모델을 학습하여 예측하는 Failover 메커니즘을 제공합니다.
"""
import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"


def _is_mlflow_available(uri: str) -> bool:
    try:
        r = requests.get(f"{uri}/api/2.0/mlflow/experiments/search", timeout=0.3)
        return r.status_code in (200, 400, 404)
    except Exception:
        return False

HORIZON_DAYS = {"1d": 1, "1w": 5, "1m": 21}

OLD_FEATURE_COLS = ["prev_close_pct_change", "ma5_gap", "ma20_gap", "volume_zscore"]

NEW_FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "ma5_gap",
    "ma20_gap",
    "ma60_gap",
    "volatility_5",
    "volatility_20",
    "volume_ratio",
    "volume_change",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "bb_upper_gap",
    "bb_lower_gap",
    "atr14",
    "volume_zscore",
    "close_lag1",
    "close_lag2",
    "close_lag3",
    "close_lag5",
    "volume_lag1",
    "volume_lag2",
    "volume_lag3",
    "news_sentiment",
    "community_sentiment",
    "news_sentiment_3d",
    "news_sentiment_7d",
    "community_sentiment_3d",
    "community_sentiment_7d",
    "kospi_return_1d",
    "kospi_return_5d",
    "kospi_ma20_gap",
    "kosdaq_return_1d",
    "kosdaq_return_5d",
    "usdkrw_return_1d",
    "usdkrw_return_5d",
    "usdkrw_ma20_gap",
    "vix_return_1d",
    "market_turnover_change",
]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MultiIndex 컬럼 평탄화 (yfinance 대응)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 날짜(date) 컬럼 확보 및 정렬
    if "date" not in df.columns:
        if "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        else:
            df = df.reset_index()
            if "Date" in df.columns:
                df.rename(columns={"Date": "date"}, inplace=True)
            elif "date" not in df.columns:
                df.rename(columns={df.columns[0]: "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 컬럼 대소문자 정규화 및 필수 컬럼(Close, High, Low, Open, Volume) 안전 처리
    cols_lower = {str(c).lower(): c for c in df.columns}
    if "close" in cols_lower and cols_lower["close"] != "Close":
        df.rename(columns={cols_lower["close"]: "Close"}, inplace=True)
    if "volume" in cols_lower and cols_lower["volume"] != "Volume":
        df.rename(columns={cols_lower["volume"]: "Volume"}, inplace=True)

    if "High" not in df.columns:
        if "high" in cols_lower:
            df["High"] = df[cols_lower["high"]]
        else:
            df["High"] = df["Close"]
    if "Low" not in df.columns:
        if "low" in cols_lower:
            df["Low"] = df[cols_lower["low"]]
        else:
            df["Low"] = df["Close"]
    if "Open" not in df.columns:
        if "open" in cols_lower:
            df["Open"] = df[cols_lower["open"]]
        else:
            df["Open"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 1.0

    # 1차원 Series로 강제 (단일 종목 DataFrame 수신 대응)
    for col in ["Close", "Volume", "High", "Low", "Open"]:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    
    # 이동평균 및 보조 열
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma60"] = df["Close"].rolling(60).mean()
    
    # 1. 舊 피처 세트 (하위 호환성 유지)
    df["prev_close_pct_change"] = df["Close"].pct_change()
    df["ma5_gap"] = (df["Close"] - df["ma5"]) / df["ma5"]
    df["ma20_gap"] = (df["Close"] - df["ma20"]) / df["ma20"]
    
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    
    # 2. 新 피처 세트
    df["return_1d"] = df["prev_close_pct_change"]
    df["return_5d"] = df["Close"].pct_change(periods=5)
    df["return_20d"] = df["Close"].pct_change(periods=20)
    df["ma60_gap"] = (df["Close"] - df["ma60"]) / df["ma60"]
    
    df["volatility_5"] = df["return_1d"].rolling(5).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    
    df["volume_ratio"] = df["Volume"] / vol_mean.replace(0, np.nan)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    df["volume_change"] = df["Volume"].pct_change().fillna(0.0)
    
    # RSI14 계산
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df["rsi14"] = rsi.fillna(50)

    # MACD 계산
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Position 및 Upper/Lower Gaps 계산
    std20 = df["Close"].rolling(20).std()
    upper_band = df["ma20"] + 2 * std20
    lower_band = df["ma20"] - 2 * std20
    band_width = upper_band - lower_band
    df["bb_position"] = (df["Close"] - lower_band) / band_width.replace(0, np.nan)
    df["bb_position"] = df["bb_position"].fillna(0.5)
    df["bb_upper_gap"] = ((upper_band - df["Close"]) / df["Close"]).fillna(0.0)
    df["bb_lower_gap"] = ((df["Close"] - lower_band) / df["Close"]).fillna(0.0)

    # ATR (Average True Range)
    high_low = df["High"] - df["Low"]
    high_close_prev = (df["High"] - df["Close"].shift(1)).abs()
    low_close_prev = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr14"] = (tr.rolling(14).mean() / df["Close"]).fillna(0.0)

    # Volume Z-Score 계산
    volume_std = df["Volume"].rolling(20).std()
    df["volume_zscore"] = (df["Volume"] - vol_mean) / volume_std.replace(0, np.nan)
    df["volume_zscore"] = df["volume_zscore"].fillna(0.0)

    # Lag Features (과거 1~5일 가격·거래량 시차 피처)
    df["close_lag1"] = df["return_1d"].shift(1).fillna(0.0)
    df["close_lag2"] = df["return_1d"].shift(2).fillna(0.0)
    df["close_lag3"] = df["return_1d"].shift(3).fillna(0.0)
    df["close_lag5"] = df["return_1d"].shift(5).fillna(0.0)

    df["volume_lag1"] = df["volume_change"].shift(1).fillna(0.0)
    df["volume_lag2"] = df["volume_change"].shift(2).fillna(0.0)
    df["volume_lag3"] = df["volume_change"].shift(3).fillna(0.0)
    
    # 5일 평활화 미래 수익률 예측 (타겟 변수)
    future_avg = (
        df["Close"].shift(-1)
        + df["Close"].shift(-2)
        + df["Close"].shift(-3)
        + df["Close"].shift(-4)
        + df["Close"].shift(-5)
    ) / 5.0
    df["next_pct_change"] = (future_avg - df["Close"]) / df["Close"]
    return df


_MACRO_CACHE = None
_MACRO_CACHE_TIME = 0.0
_MODEL_CACHE = {}
_MODEL_CACHE_TTL = 3600  # 1시간 모델 캐시


def _get_cached_model(ticker: str):
    if ticker in _MODEL_CACHE:
        entry = _MODEL_CACHE[ticker]
        if time.time() - entry["time"] < _MODEL_CACHE_TTL:
            return entry
    return None


def _set_cached_model(ticker: str, model_reg, model_clf, method, model_info, features_to_use):
    _MODEL_CACHE[ticker] = {
        "model_reg": model_reg,
        "model_clf": model_clf,
        "method": method,
        "model_info": model_info,
        "features_to_use": features_to_use,
        "time": time.time(),
    }


def get_macro_features(period: str = "3mo") -> pd.DataFrame:
    global _MACRO_CACHE, _MACRO_CACHE_TIME
    now = time.time()
    if _MACRO_CACHE is not None and isinstance(_MACRO_CACHE, pd.DataFrame) and len(_MACRO_CACHE) > 0 and (now - _MACRO_CACHE_TIME) < 3600:
        return _MACRO_CACHE.copy()

    print(f"[INFO] Downloading Macro economic data for period {period}...")
    try:
        kospi = yf.download("^KS11", period=period, auto_adjust=True, progress=False)
        kosdaq = yf.download("^KQ11", period=period, auto_adjust=True, progress=False)
        usdkrw = yf.download("KRW=X", period=period, auto_adjust=True, progress=False)
        vix = yf.download("^VIX", period=period, auto_adjust=True, progress=False)
        
        for df_macro in [kospi, kosdaq, usdkrw, vix]:
            if isinstance(df_macro.columns, pd.MultiIndex):
                df_macro.columns = df_macro.columns.get_level_values(0)
            df_macro.reset_index(inplace=True)
            df_macro.rename(columns={"Date": "date"}, inplace=True)
            df_macro["date"] = pd.to_datetime(df_macro["date"])
            df_macro.sort_values("date", inplace=True)
            
        # KOSPI features
        kospi["kospi_return_1d"] = kospi["Close"].pct_change()
        kospi["kospi_return_5d"] = kospi["Close"].pct_change(periods=5)
        kospi_ma20 = kospi["Close"].rolling(20).mean()
        kospi["kospi_ma20_gap"] = (kospi["Close"] - kospi_ma20) / kospi_ma20
        kospi_amount = kospi["Close"] * kospi["Volume"]
        kospi["market_turnover_change"] = kospi_amount.pct_change()
        
        kospi_features = kospi[[
            "date", "kospi_return_1d", "kospi_return_5d", "kospi_ma20_gap", "market_turnover_change"
        ]]
        
        # KOSDAQ features
        kosdaq["kosdaq_return_1d"] = kosdaq["Close"].pct_change()
        kosdaq["kosdaq_return_5d"] = kosdaq["Close"].pct_change(periods=5)
        
        kosdaq_features = kosdaq[["date", "kosdaq_return_1d", "kosdaq_return_5d"]]
        
        # USD/KRW features
        usdkrw["usdkrw_return_1d"] = usdkrw["Close"].pct_change()
        usdkrw["usdkrw_return_5d"] = usdkrw["Close"].pct_change(periods=5)
        usdkrw_ma20 = usdkrw["Close"].rolling(20).mean()
        usdkrw["usdkrw_ma20_gap"] = (usdkrw["Close"] - usdkrw_ma20) / usdkrw_ma20
        
        usdkrw_features = usdkrw[["date", "usdkrw_return_1d", "usdkrw_return_5d", "usdkrw_ma20_gap"]]
        
        # VIX features
        vix["vix_return_1d"] = vix["Close"].pct_change()
        
        vix_features = vix[["date", "vix_return_1d"]]
        
        # Merge all
        macro_df = pd.merge(kospi_features, kosdaq_features, on="date", how="outer")
        macro_df = pd.merge(macro_df, usdkrw_features, on="date", how="outer")
        macro_df = pd.merge(macro_df, vix_features, on="date", how="outer")
        macro_df.sort_values("date", inplace=True)
        
        macro_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        macro_df.ffill(inplace=True)
        macro_df.bfill(inplace=True)
        _MACRO_CACHE = macro_df.copy()
        _MACRO_CACHE_TIME = now
        return macro_df
    except Exception as e:
        print(f"[WARN] Failed to fetch macro features: {e}")
        return pd.DataFrame()
    
FEATURE_DISPLAY_NAMES = {
    "return_1d": "1일 주가 수익률",
    "return_5d": "5일 주가 수익률",
    "return_20d": "20일 주가 수익률",
    "ma5_gap": "5일 이동평균선 이격도",
    "ma20_gap": "20일 이동평균선 이격도",
    "ma60_gap": "60일 이동평균선 이격도",
    "volatility_5": "5일 주가 변동성",
    "volatility_20": "20일 주가 변동성",
    "volume_ratio": "거래량 평균 대비 비율",
    "rsi14": "RSI (14일 상대강도지수)",
    "news_sentiment": "실시간 뉴스 감성 지표",
    "community_sentiment": "커뮤니티 투자 심리 지표",
    "macd": "MACD 지표",
    "macd_signal": "MACD 시그널",
    "macd_hist": "MACD 히스토그램",
    "bb_position": "볼린저 밴드 위치 비율",
    "volume_zscore": "거래량 급증도 (Z-score)",
    "news_sentiment_3d": "3일 평균 뉴스 감성",
    "news_sentiment_7d": "7일 평균 뉴스 감성",
    "community_sentiment_3d": "3일 커뮤니티 심리",
    "community_sentiment_7d": "7일 커뮤니티 심리",
    "kospi_return_1d": "코스피 1일 수익률",
    "kospi_return_5d": "코스피 5일 수익률",
    "kospi_ma20_gap": "코스피 20일 이평 이격도",
    "kosdaq_return_1d": "코스닥 1일 수익률",
    "kosdaq_return_5d": "코스닥 5일 수익률",
    "usdkrw_return_1d": "원/달러 환율 1일 변동률",
    "usdkrw_return_5d": "원/달러 환율 5일 변동률",
    "usdkrw_ma20_gap": "환율 20일 이평 이격도",
    "vix_return_1d": "VIX 지수 1일 변동률",
    "market_turnover_change": "시장 거래대금 변동률",
    "prev_close_pct_change": "전일 대비 변동률",
}


def _calculate_shap_contributions(model_reg, X_latest: pd.DataFrame, features_to_use: list, predicted_change: float = 0.0) -> dict:
    """
    SHAP (TreeExplainer) 및 피처 방향성을 고려하여 예측 결과에 대한 피처 기여도(%p) 산출
    """
    contributions = {}
    row_vals = X_latest.iloc[0].to_dict() if len(X_latest) > 0 else {}
    
    try:
        import shap
        explainer = shap.TreeExplainer(model_reg)
        shap_vals = explainer.shap_values(X_latest, check_additivity=False)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[0]

        for fname, val in zip(features_to_use, shap_vals):
            c_val = float(val) * 100.0
            contributions[fname] = round(c_val, 3)
        return contributions
    except Exception as e:
        print(f"[WARN] SHAP TreeExplainer 미지원/에러 (방향성 Fallback 적용): {e}")
        
    if hasattr(model_reg, "feature_importances_"):
        # 방향성 피처 맵핑 (단순 절대값 수치를 상승/하락 영향에 맞게 부호 부여)
        for fname, imp in zip(features_to_use, model_reg.feature_importances_):
            val_raw = row_vals.get(fname, 0.0)
            val = float(val_raw) if val_raw is not None and not pd.isna(val_raw) else 0.0
            importance_pct = float(imp) * 10.0
            
            # 피처 상태값 및 주가 방향에 따른 부호(+/-) 부여
            if fname in ["return_1d", "return_5d", "return_20d", "prev_close_pct_change", "news_sentiment", "community_sentiment", "macd", "macd_hist"]:
                sign = 1.0 if val >= 0 else -1.0
            elif fname in ["rsi14"]:
                sign = 1.0 if val >= 50.0 else -1.0
            elif fname in ["volume_zscore", "volume_ratio"]:
                ret1 = float(row_vals.get("return_1d", 0.0) or 0.0)
                sign = -1.0 if ret1 < 0 else 1.0
            else:
                sign = 1.0 if predicted_change >= 0 else -1.0
                
            contributions[fname] = round(sign * importance_pct, 3)

    return contributions


def predict(df: pd.DataFrame, ticker: str, horizon: str = "1d") -> dict:
    """
    df: columns = [date, Close, Volume]
    ticker: 종목 티커 (예: '005930.KS')
    horizon: 예측 시점 ('1d', '1w', '1m')
    반환: {predicted_close, predicted_change_pct, base_close, base_date, features_used, method}
    """
    if horizon not in HORIZON_DAYS:
        raise ValueError(f"지원하지 않는 horizon: {horizon}")

    feat_df = _build_features(df)
    
    # 매크로 피처 병합
    try:
        macro_df = get_macro_features(period="3mo")
        feat_df["date"] = pd.to_datetime(feat_df["date"]).dt.tz_localize(None)
        macro_df["date"] = pd.to_datetime(macro_df["date"]).dt.tz_localize(None)
        feat_df = pd.merge(feat_df, macro_df, on="date", how="left")
        
        macro_cols = [
            "kospi_return_1d", "kospi_return_5d", "kospi_ma20_gap", "market_turnover_change",
            "kosdaq_return_1d", "kosdaq_return_5d",
            "usdkrw_return_1d", "usdkrw_return_5d", "usdkrw_ma20_gap",
            "vix_return_1d"
        ]
        for col in macro_cols:
            feat_df[col] = feat_df[col].fillna(0.0)
        print(f"[INFO] Successfully merged macro features in predict_service for {ticker}")
    except Exception as macro_err:
        print(f"[WARN] Failed to merge macro features in predict_service: {macro_err}")
        macro_cols = [
            "kospi_return_1d", "kospi_return_5d", "kospi_ma20_gap", "market_turnover_change",
            "kosdaq_return_1d", "kosdaq_return_5d",
            "usdkrw_return_1d", "usdkrw_return_5d", "usdkrw_ma20_gap",
            "vix_return_1d"
        ]
        for col in macro_cols:
            feat_df[col] = 0.0
    
    # 텍스트 감성 지표 병합 (방식 A)
    sentiment_path = "/opt/data/processed/features.parquet"
    if os.path.exists(sentiment_path):
        try:
            sentiment_df = pd.read_parquet(sentiment_path)
            sentiment_df = sentiment_df[sentiment_df["ticker"] == ticker].copy()
            sentiment_df = sentiment_df.drop_duplicates(subset=["date"])
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.tz_localize(None)
            feat_df["date"] = pd.to_datetime(feat_df["date"]).dt.tz_localize(None)
            
            feat_df = pd.merge(feat_df, sentiment_df[["date", "news_sentiment", "community_sentiment"]], on="date", how="left")
            feat_df["news_sentiment"] = feat_df["news_sentiment"].fillna(0.0)
            feat_df["community_sentiment"] = feat_df["community_sentiment"].fillna(0.0)
            
            feat_df["news_sentiment_3d"] = feat_df["news_sentiment"].rolling(3, min_periods=1).mean()
            feat_df["news_sentiment_7d"] = feat_df["news_sentiment"].rolling(7, min_periods=1).mean()
            feat_df["community_sentiment_3d"] = feat_df["community_sentiment"].rolling(3, min_periods=1).mean()
            feat_df["community_sentiment_7d"] = feat_df["community_sentiment"].rolling(7, min_periods=1).mean()
            print(f"[INFO] Loaded and merged sentiment data from parquet for {ticker}")
        except Exception as e:
            print(f"[WARN] Failed to merge sentiment parquet in predict_service: {e}")
            feat_df["news_sentiment"] = 0.0
            feat_df["community_sentiment"] = 0.0
    else:
        print(f"[INFO] No sentiment parquet found at {sentiment_path}")
        feat_df["news_sentiment"] = 0.0
        feat_df["community_sentiment"] = 0.0

    # Ensure all sentiment features exist
    for col in ["news_sentiment_3d", "news_sentiment_7d", "community_sentiment_3d", "community_sentiment_7d"]:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 0. In-Memory 모델 캐시 확인 (반복 온디맨드 학습 병목 방지)
    cached = _get_cached_model(ticker)
    if cached is not None:
        model_reg = cached["model_reg"]
        model_clf = cached["model_clf"]
        method = cached["method"]
        model_info = cached["model_info"]
        features_to_use = cached["features_to_use"]
        print(f"[INFO] Using cached model for {ticker} (method={method}, R2={model_info.get('r2', 0.0)})")
    else:
        model_clf = None
        model_reg = None
        method = ""
        features_to_use = NEW_FEATURE_COLS
        model_info = {
            "version": "v1.0 (On-Demand AI)",
            "r2": 0.0,
            "train_date": "On-demand"
        }

        # 1. MLflow Model Registry 및 Runs로부터 모델 로드 시도
        try:
            import mlflow
            tracking_uris = [
                os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
                "http://localhost:5000",
                "http://127.0.0.1:5000"
            ]
            
            loaded = False
            for uri in tracking_uris:
                if loaded:
                    break
                if not _is_mlflow_available(uri):
                    continue
                try:
                    mlflow.set_tracking_uri(uri)
                    # 1-1. Classifier 로드 시도
                    try:
                        model_clf = mlflow.sklearn.load_model(f"models:/{ticker}_classifier/latest")
                        print(f"[INFO] Loaded classifier model for {ticker} from {uri}")
                    except Exception:
                        pass

                    # 1-2. Regressor 로드 시도 (이름: {ticker}_regressor 또는 {ticker})
                    for reg_name in [f"{ticker}_regressor", ticker]:
                        try:
                            model_reg = mlflow.sklearn.load_model(f"models:/{reg_name}/latest")
                            method = "mlflow_registry_2step"
                            client = mlflow.tracking.MlflowClient()
                            versions = client.get_latest_versions(reg_name, stages=["None", "Staging", "Production"])
                            if versions:
                                v = versions[0]
                                run = client.get_run(v.run_id)
                                r2_val = run.data.metrics.get("r2", 0.0) or 0.0
                                import datetime
                                train_date_str = datetime.datetime.fromtimestamp(run.info.start_time / 1000.0).strftime('%Y-%m-%d %H:%M')
                                model_info = {
                                    "version": f"v{v.version} (MLflow Registry)",
                                    "r2": round(float(r2_val), 4),
                                    "train_date": train_date_str
                                }
                                loaded = True
                                print(f"[INFO] Successfully loaded registered model {reg_name} (R2={r2_val:.4f}) from {uri}")
                                break
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] Failed to load model from MLflow (ticker: {ticker}): {e}")

        # 2. 로드한 모델의 피처 컬럼 수에 맞춰 피처 자동 선택 (하위 호환성 확보)
        model = model_reg
        if model is not None:
            try:
                n_features = getattr(model, "n_features_in_", len(NEW_FEATURE_COLS))
                if n_features == len(OLD_FEATURE_COLS):
                    features_to_use = OLD_FEATURE_COLS
                    print(f"[INFO] Loaded model uses older feature set (4 features)")
                else:
                    features_to_use = NEW_FEATURE_COLS
            except Exception:
                features_to_use = NEW_FEATURE_COLS
            
            # 캐시에 저장
            _set_cached_model(ticker, model_reg, model_clf, method, model_info, features_to_use)

        # 3. 모델이 없거나 로드에 실패했다면, 즉석 온디맨드 머신러닝 학습 수행 및 실제 R² 산출
        if model is None:
            features_to_use = NEW_FEATURE_COLS
            train_df = feat_df.dropna(subset=features_to_use + ["next_pct_change"])
            
            if len(train_df) < 20:
                # 데이터 부족 시 단순 추세 외삽
                last_close = float(df["Close"].iloc[-1])
                naive_change = float(df["Close"].pct_change().tail(5).mean() or 0)
                days = HORIZON_DAYS[horizon]
                predicted_change = naive_change * days
                predicted_close = last_close * (1 + predicted_change)
                return {
                    "base_date": str(df["date"].iloc[-1]),
                    "base_close": last_close,
                    "predicted_close": round(predicted_close, 2),
                    "predicted_change_pct": round(predicted_change * 100, 2),
                    "direction": "UP" if predicted_change > 0 else "DOWN",
                    "direction_confidence": 50.0,
                    "expected_return_pct": round(predicted_change * 100, 2),
                    "expected_price": round(predicted_close, 2),
                    "confidence_interval": {
                        "margin_pct": 4.9,
                        "lower_pct": round(predicted_change * 100 - 4.9, 2),
                        "upper_pct": round(predicted_change * 100 + 4.9, 2),
                        "lower_price": round(last_close * (1 + (predicted_change - 0.049)), 2),
                        "upper_price": round(last_close * (1 + (predicted_change + 0.049)), 2),
                    },
                    "is_low_confidence": True,
                    "confidence_warning_msg": "⚠️ [모델 신뢰도 경고]: 데이터 학습 수량이 제한적이어 단순 추세 외삽 방식이 적용되었습니다.",
                    "features_used": {
                        "note": "데이터 부족으로 단순 추세 외삽 사용",
                        "feature_contributions": {},
                        "display_names": FEATURE_DISPLAY_NAMES,
                    },
                    "method": "naive_trend",
                    "model_info": model_info,
                }
                
            X_train = train_df[features_to_use]
            y_train = train_df["next_pct_change"]
            
            from sklearn.metrics import r2_score
            from datetime import datetime

            # 빠른 n_jobs=-1 멀티코어 학습 (n_estimators=50) 및 검증 R² 측정
            if len(train_df) >= 30:
                split_idx = int(len(train_df) * 0.8)
                X_tr, y_tr = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
                X_val, y_val = X_train.iloc[split_idx:], y_train.iloc[split_idx:]
                
                val_model = RandomForestRegressor(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42)
                val_model.fit(X_tr, y_tr)
                val_preds = val_model.predict(X_val)
                raw_r2 = float(r2_score(y_val, val_preds))
                
                model = RandomForestRegressor(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42)
                model.fit(X_train, y_train)
            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42)
                model.fit(X_train, y_train)
                train_preds = model.predict(X_train)
                raw_r2 = float(r2_score(y_train, train_preds))

            real_r2 = round(raw_r2, 4)
            model_reg = model
            method = "on_demand_ml"

            model_info = {
                "version": "v1.0 (On-Demand Machine Learning)",
                "r2": real_r2,
                "train_date": datetime.now().strftime("%Y-%m-%d %H:%M")
            }

            # 온디맨드 학습 모델을 메모리 캐시에 저장하여 재호출 시 0.1초 내 응답
            _set_cached_model(ticker, model_reg, model_clf, method, model_info, features_to_use)

    # 4. 예측 수행
    latest_row = feat_df.dropna(subset=features_to_use).iloc[-1]
    X_latest = feat_df[features_to_use].tail(1).copy()
    X_latest = (
        X_latest
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float32)
        .fillna(0.0)
    )

    # 4-1. 1단계: 방향 분류 예측
    direction = "FLAT"
    direction_confidence = 50.0

    if model_clf is not None:
        try:
            pred_class = int(model_clf.predict(X_latest)[0])
            class_map = {0: "DOWN", 1: "FLAT", 2: "UP"}
            direction = class_map.get(pred_class, "FLAT")
            
            probs = model_clf.predict_proba(X_latest)[0]
            direction_confidence = float(max(probs)) * 100.0
            print(f"[CLF PREDICT] {ticker} - Class={direction} Conf={direction_confidence:.2f}%")
        except Exception as e:
            print(f"[WARN] Failed to predict with classifier: {e}")
            model_clf = None

    # 4-2. 2단계: 회귀 예측 및 가중 조절
    pred_val = float(model_reg.predict(X_latest)[0])
    
    if len(features_to_use) == len(NEW_FEATURE_COLS):
        # 5일 예측 타겟이므로 일일 수익률 환산
        daily_pred_change = pred_val / 5.0
    else:
        daily_pred_change = pred_val

    days = HORIZON_DAYS[horizon]
    compounded_change = (1 + daily_pred_change) ** days - 1

    # Align direction and confidence with the sign of the expected return to avoid contradictions
    if compounded_change > 0.002:
        direction = "UP"
        if model_clf is not None:
            try:
                direction_confidence = float(probs[2]) * 100.0
            except Exception:
                pass
    elif compounded_change < -0.002:
        direction = "DOWN"
        if model_clf is not None:
            try:
                direction_confidence = float(probs[0]) * 100.0
            except Exception:
                pass
    else:
        direction = "FLAT"
        if model_clf is not None:
            try:
                direction_confidence = float(probs[1]) * 100.0
            except Exception:
                pass

    # 분류기 결과가 횡보인 경우 강제로 기대수익률을 0에 가깝게 조율하여 노이즈 억제
    r2_score = model_info.get("r2", 0.0)

    if model_clf is not None:
        if direction == "FLAT":
            compounded_change = compounded_change * 0.2
    else:
        # 1. 회귀 모델 트리의 예측 부호 일치율 (Tree Voting Consensus)
        if hasattr(model_reg, "estimators_"):
            try:
                X_vals = X_latest.values if hasattr(X_latest, "values") else X_latest
                tree_preds = [float(e.predict(X_vals)[0]) for e in model_reg.estimators_]
                if compounded_change > 0:
                    agree_count = sum(1 for p in tree_preds if p > 0)
                else:
                    agree_count = sum(1 for p in tree_preds if p < 0)
                raw_consensus = (agree_count / len(tree_preds)) * 100.0
            except Exception:
                raw_consensus = 55.0
        else:
            raw_consensus = 55.0

        # 2. 연속적 R² 캘리브레이션 함수: Calibration(R²) = max(0.4, min(1.0, (R² + 1.0) / 2.0))
        r2_val = float(r2_score)
        calibration_factor = max(0.4, min(1.0, (r2_val + 1.0) / 2.0))

        # 3. 보정된 최종 방향 신뢰도 = Tree_Consensus * Calibration(R²)
        calibrated_conf = raw_consensus * calibration_factor
        direction_confidence = round(max(50.0, min(99.0, calibrated_conf)), 1)

    last_close = float(df["Close"].iloc[-1])
    predicted_close = last_close * (1 + compounded_change)

    # 95% 신뢰구간 (Confidence Interval) 및 모델 불확실성 산출
    std_pct = 2.5
    if hasattr(model_reg, "estimators_"):
        try:
            X_vals = X_latest.values if hasattr(X_latest, "values") else X_latest
            tree_preds = [float(e.predict(X_vals)[0]) for e in model_reg.estimators_]
            tree_std = float(np.std(tree_preds))
            if tree_std > 0:
                std_pct = max(1.5, tree_std * 100.0)
        except Exception:
            pass

    ci_margin = round(1.96 * std_pct, 2)
    pred_pct = round(compounded_change * 100, 2)
    ci_lower_pct = round(pred_pct - ci_margin, 2)
    ci_upper_pct = round(pred_pct + ci_margin, 2)

    ci_lower_price = round(last_close * (1 + ci_lower_pct / 100.0), 2)
    ci_upper_price = round(last_close * (1 + ci_upper_pct / 100.0), 2)

    confidence_interval = {
        "margin_pct": ci_margin,
        "lower_pct": ci_lower_pct,
        "upper_pct": ci_upper_pct,
        "lower_price": ci_lower_price,
        "upper_price": ci_upper_price,
    }

    # R² 오차 기반 신뢰도 경고 판단
    is_low_confidence = bool(r2_score < 0.0 or method in ["on_the_fly_fallback", "naive_trend"])
    
    confidence_warning_msg = None
    if is_low_confidence:
        if r2_score < 0:
            confidence_warning_msg = f"⚠️ [모델 설명력 경고 (R² = {r2_score:+.4f})]: 해당 종목은 노이즈 및 변동성이 높아 시계열 R² 설명력이 낮은 구간입니다(R² < 0). 회귀 모델 수치 예측만을 단독으로 의존하지 마시고, 아래의 기술적 지표 및 뉴스/커뮤니티 종합 전망을 함께 참고하세요."
        else:
            confidence_warning_msg = f"⚠️ [모델 신뢰도 경고 (R² = {r2_score:+.4f})]: 현재 시점의 ML 모델 예측 성능이 저조하여 오차 범위가 큽니다. 수치 예측만을 단독으로 의존하지 마시고, 아래의 기술적 지표 및 뉴스 종합 전망을 함께 고려하세요."

    importance = {}
    if hasattr(model_reg, "feature_importances_"):
        importance = dict(zip(features_to_use, model_reg.feature_importances_.round(4)))

    shap_contributions = _calculate_shap_contributions(model_reg, X_latest, features_to_use, compounded_change)

    return {
        "base_date": str(df["date"].iloc[-1]),
        "base_close": round(last_close, 2),
        "predicted_close": round(predicted_close, 2),
        "predicted_change_pct": round(compounded_change * 100, 2),
        "direction": direction,
        "direction_confidence": round(direction_confidence, 2),
        "expected_return_pct": round(compounded_change * 100, 2),
        "expected_price": round(predicted_close, 2),
        "confidence_interval": confidence_interval,
        "is_low_confidence": is_low_confidence,
        "confidence_warning_msg": confidence_warning_msg,
        "features_used": {
            "latest_values": {c: round(float(latest_row[c]), 4) for c in features_to_use},
            "feature_importance": {k: float(v) for k, v in importance.items()},
            "feature_contributions": shap_contributions,
            "display_names": FEATURE_DISPLAY_NAMES,
        },
        "method": method if method else "mlflow_registry_2step",
        "model_info": model_info,
    }


def load_from_registry(ticker: str):
    """
    MLflow Registry에서 프로덕션 모델 로드용 래퍼 (기존 구현 인터페이스 보존)
    """
    import mlflow
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{ticker}/latest")


def simulate_predict(df: pd.DataFrame, ticker: str, overrides: dict, horizon: str = "1w") -> dict:
    """
    사용자가 지정한 피처 변수(뉴스 감성, 환율, RSI 등)의 가상 조건을 오버라이드하여
    What-If 시뮬레이션 주가 예측을 계산합니다.
    """
    base_res = predict(df, ticker=ticker, horizon=horizon)
    
    feat_df = _build_features(df)
    try:
        macro_df = get_macro_features(period="3mo")
        feat_df["date"] = pd.to_datetime(feat_df["date"])
        macro_df["date"] = pd.to_datetime(macro_df["date"])
        feat_df = pd.merge(feat_df, macro_df, on="date", how="left")
    except Exception:
        pass
        
    features_to_use = NEW_FEATURE_COLS
    for c in features_to_use:
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    X_latest_sim = feat_df[features_to_use].tail(1).copy()
    X_latest_sim = X_latest_sim.apply(pd.to_numeric, errors="coerce").astype(np.float32).fillna(0.0)

    # 피처 오버라이드 적용
    for k, v in overrides.items():
        if k in X_latest_sim.columns and v is not None:
            X_latest_sim[k] = float(v)

    train_df = feat_df.dropna(subset=features_to_use + ["next_pct_change"])
    if len(train_df) >= 20:
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        model.fit(train_df[features_to_use], train_df["next_pct_change"])
        sim_pred_val = float(model.predict(X_latest_sim)[0])
        daily_sim_change = sim_pred_val / 5.0
        days = HORIZON_DAYS.get(horizon, 5)
        sim_compounded_change = (1 + daily_sim_change) ** days - 1
    else:
        sim_compounded_change = base_res.get("predicted_change_pct", 0.0) / 100.0

    last_close = base_res.get("base_close", float(df["Close"].iloc[-1]))
    sim_predicted_close = last_close * (1 + sim_compounded_change)
    sim_pct = round(sim_compounded_change * 100, 2)
    base_pct = base_res.get("predicted_change_pct", 0.0)
    diff_pct = round(sim_pct - base_pct, 2)

    return {
        "ticker": ticker,
        "horizon": horizon,
        "base_predicted_change_pct": base_pct,
        "sim_predicted_change_pct": sim_pct,
        "diff_pct": diff_pct,
        "sim_predicted_close": round(sim_predicted_close, 2),
        "overrides_applied": overrides
    }
