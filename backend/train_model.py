import os
import warnings
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# -----------------------------
# 환경설정
# -----------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000"
)

EXPERIMENT_NAME = "stock_prediction"

DEFAULT_TICKERS = [
    "005930.KS",  # 삼성전자
    "000660.KS",  # SK하이닉스
    "035420.KS",  # NAVER
]

FEATURE_COLS = [
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


# -----------------------------
# Feature Engineering
# -----------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. returns
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(periods=5)
    df["return_20d"] = df["Close"].pct_change(periods=20)

    # 2. moving averages and gaps
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma60"] = df["Close"].rolling(60).mean()

    df["ma5_gap"] = (df["Close"] - df["ma5"]) / df["ma5"]
    df["ma20_gap"] = (df["Close"] - df["ma20"]) / df["ma20"]
    df["ma60_gap"] = (df["Close"] - df["ma60"]) / df["ma60"]

    # 3. volatility
    df["volatility_5"] = df["return_1d"].rolling(5).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()

    # 4. volume ratio and change
    volume_mean = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / volume_mean.replace(0, np.nan)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    df["volume_change"] = df["Volume"].pct_change().fillna(0.0)

    # 5. rsi14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df["rsi14"] = rsi.fillna(50)

    # 6. MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 7. Bollinger Position and Band Gaps
    std20 = df["Close"].rolling(20).std()
    upper_band = df["ma20"] + 2 * std20
    lower_band = df["ma20"] - 2 * std20
    band_width = upper_band - lower_band
    df["bb_position"] = (df["Close"] - lower_band) / band_width.replace(0, np.nan)
    df["bb_position"] = df["bb_position"].fillna(0.5)
    df["bb_upper_gap"] = ((upper_band - df["Close"]) / df["Close"]).fillna(0.0)
    df["bb_lower_gap"] = ((df["Close"] - lower_band) / df["Close"]).fillna(0.0)

    # 8. ATR (Average True Range)
    high_low = df["High"] - df["Low"]
    high_close_prev = (df["High"] - df["Close"].shift(1)).abs()
    low_close_prev = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr14"] = (tr.rolling(14).mean() / df["Close"]).fillna(0.0)

    # 9. Volume Z-Score
    volume_std = df["Volume"].rolling(20).std()
    df["volume_zscore"] = (df["Volume"] - volume_mean) / volume_std.replace(0, np.nan)
    df["volume_zscore"] = df["volume_zscore"].fillna(0.0)

    # 10. Lag Features (과거 1~5일 가격·거래량 시차 데이터)
    df["close_lag1"] = df["return_1d"].shift(1).fillna(0.0)
    df["close_lag2"] = df["return_1d"].shift(2).fillna(0.0)
    df["close_lag3"] = df["return_1d"].shift(3).fillna(0.0)
    df["close_lag5"] = df["return_1d"].shift(5).fillna(0.0)

    df["volume_lag1"] = df["volume_change"].shift(1).fillna(0.0)
    df["volume_lag2"] = df["volume_change"].shift(2).fillna(0.0)
    df["volume_lag3"] = df["volume_change"].shift(3).fillna(0.0)

    # 5일 평활화 미래 수익률 예측 (타겟 변수 - 5일 후 수익률%)
    future_avg = (
        df["Close"].shift(-1)
        + df["Close"].shift(-2)
        + df["Close"].shift(-3)
        + df["Close"].shift(-4)
        + df["Close"].shift(-5)
    ) / 5.0
    df["target"] = (future_avg - df["Close"]) / df["Close"]

    return df


def get_macro_features(period: str = "5y") -> pd.DataFrame:
    print(f"[INFO] Downloading Macro economic data for period {period}...")
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
    
    return macro_df


# -----------------------------
# 데이터 수집
# -----------------------------

def load_price_data(
    ticker: str,
    period: str = "5y"
):

    print(f"[INFO] Downloading {ticker}")

    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
    )
    
    

    print("columns:", df.columns.tolist())

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df.rename(
        columns={"Date": "date"},
        inplace=True
    )

    if df.empty:
        raise ValueError(
            f"{ticker} 데이터 없음"
        )

    return df

# -----------------------------
# 모델 학습
# -----------------------------

def train_ticker(ticker: str):

    print("=" * 60)
    print(f"[TRAIN] {ticker}")

    raw_df = load_price_data(ticker)

    feat_df = build_features(raw_df)

    # 매크로 피처 병합
    try:
        macro_df = get_macro_features(period="5y")
        feat_df["date"] = pd.to_datetime(feat_df["date"])
        macro_df["date"] = pd.to_datetime(macro_df["date"])
        feat_df = pd.merge(feat_df, macro_df, on="date", how="left")
        
        macro_cols = [
            "kospi_return_1d", "kospi_return_5d", "kospi_ma20_gap", "market_turnover_change",
            "kosdaq_return_1d", "kosdaq_return_5d",
            "usdkrw_return_1d", "usdkrw_return_5d", "usdkrw_ma20_gap",
            "vix_return_1d"
        ]
        for col in macro_cols:
            feat_df[col] = feat_df[col].fillna(0.0)
        print("[INFO] Successfully merged macro features")
    except Exception as macro_err:
        print(f"[WARN] Failed to download or merge macro features: {macro_err}")
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
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
            feat_df["date"] = pd.to_datetime(feat_df["date"])
            
            feat_df = pd.merge(feat_df, sentiment_df[["date", "news_sentiment", "community_sentiment"]], on="date", how="left")
            feat_df["news_sentiment"] = feat_df["news_sentiment"].fillna(0.0)
            feat_df["community_sentiment"] = feat_df["community_sentiment"].fillna(0.0)
            
            # Sentiment rolling features
            feat_df["news_sentiment_3d"] = feat_df["news_sentiment"].rolling(3, min_periods=1).mean()
            feat_df["news_sentiment_7d"] = feat_df["news_sentiment"].rolling(7, min_periods=1).mean()
            feat_df["community_sentiment_3d"] = feat_df["community_sentiment"].rolling(3, min_periods=1).mean()
            feat_df["community_sentiment_7d"] = feat_df["community_sentiment"].rolling(7, min_periods=1).mean()
            print(f"[INFO] Merged sentiment data and built rolling features from {sentiment_path}")
        except Exception as e:
            print(f"[WARN] Failed to merge sentiment parquet: {e}")
            feat_df["news_sentiment"] = 0.0
            feat_df["community_sentiment"] = 0.0
    else:
        print(f"[INFO] No sentiment parquet found at {sentiment_path}. Defaulting to 0.0")
        feat_df["news_sentiment"] = 0.0
        feat_df["community_sentiment"] = 0.0

    # Ensure all sentiment features exist
    for col in ["news_sentiment_3d", "news_sentiment_7d", "community_sentiment_3d", "community_sentiment_7d"]:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_df = feat_df.dropna(
        subset=FEATURE_COLS + ["target"]
    )

    if len(train_df) < 100:
        raise ValueError(
            f"{ticker}: 학습 데이터 부족"
        )

    X = train_df[FEATURE_COLS]
    y = train_df["target"]

    # --- Regressor Split & Train ---
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    rf_reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    rf_reg.fit(X_train_r, y_train_r)
    rf_reg_pred = rf_reg.predict(X_test_r)
    rf_reg_rmse = np.sqrt(mean_squared_error(y_test_r, rf_reg_pred))
    rf_reg_mae = mean_absolute_error(y_test_r, rf_reg_pred)
    rf_reg_r2 = r2_score(y_test_r, rf_reg_pred)

    reg_model_type = "RandomForestRegressor"
    reg_model = rf_reg
    reg_rmse = rf_reg_rmse
    reg_mae = rf_reg_mae
    reg_r2 = rf_reg_r2
    reg_params = {"n_estimators": 300, "max_depth": 8}
    reg_feature_importance = dict(zip(FEATURE_COLS, rf_reg.feature_importances_))

    try:
        import xgboost as xgb
        xgb_reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            random_state=42,
            n_jobs=-1,
        )
        xgb_reg.fit(X_train_r, y_train_r)
        xgb_reg_pred = xgb_reg.predict(X_test_r)
        xgb_reg_rmse = np.sqrt(mean_squared_error(y_test_r, xgb_reg_pred))
        xgb_reg_mae = mean_absolute_error(y_test_r, xgb_reg_pred)
        xgb_reg_r2 = r2_score(y_test_r, xgb_reg_pred)

        print(f"[REG COMPARE] {ticker} - XGBoost R2={xgb_reg_r2:.4f} vs RandomForest R2={rf_reg_r2:.4f}")
        if xgb_reg_r2 > rf_reg_r2:
            reg_model = xgb_reg
            reg_model_type = "XGBRegressor"
            reg_rmse = xgb_reg_rmse
            reg_mae = xgb_reg_mae
            reg_r2 = xgb_reg_r2
            reg_params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03}
            reg_feature_importance = dict(zip(FEATURE_COLS, xgb_reg.feature_importances_))
    except Exception as xgb_err:
        print(f"[WARN] Failed to train XGBRegressor: {xgb_err}")

    # --- Classifier Split & Train ---
    # target_class definition: 2: UP (return > 0.01), 0: DOWN (return < -0.01), 1: FLAT (otherwise)
    y_class = np.where(y > 0.01, 2, np.where(y < -0.01, 0, 1))

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, shuffle=False, test_size=0.2
    )

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train_c, y_train_c)
    rf_clf_pred = rf_clf.predict(X_test_c)
    rf_clf_acc = accuracy_score(y_test_c, rf_clf_pred)

    clf_model_type = "RandomForestClassifier"
    clf_model = rf_clf
    clf_acc = rf_clf_acc
    clf_params = {"n_estimators": 300, "max_depth": 8}
    clf_feature_importance = dict(zip(FEATURE_COLS, rf_clf.feature_importances_))

    try:
        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            random_state=42,
            n_jobs=-1,
        )
        xgb_clf.fit(X_train_c, y_train_c)
        xgb_clf_pred = xgb_clf.predict(X_test_c)
        xgb_clf_acc = accuracy_score(y_test_c, xgb_clf_pred)

        print(f"[CLF COMPARE] {ticker} - XGBoost Acc={xgb_clf_acc:.4f} vs RandomForest Acc={rf_clf_acc:.4f}")
        if xgb_clf_acc > rf_clf_acc:
            clf_model = xgb_clf
            clf_model_type = "XGBClassifier"
            clf_acc = xgb_clf_acc
            clf_params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03}
            clf_feature_importance = dict(zip(FEATURE_COLS, xgb_clf.feature_importances_))
    except Exception as xgb_err:
        print(f"[WARN] Failed to train XGBClassifier: {xgb_err}")

    # --- MLflow Logging & Registry (Dual Registration) ---
    
    # 1. Regressor Logging
    with mlflow.start_run(run_name=f"{ticker}_regressor"):
        mlflow.set_tag("ticker", ticker)
        mlflow.set_tag("model_type", reg_model_type)
        mlflow.set_tag("task", "regression")
        
        for p_name, p_val in reg_params.items():
            mlflow.log_param(p_name, p_val)
        mlflow.log_param("train_rows", len(X_train_r))
        mlflow.log_metric("rmse", float(reg_rmse))
        mlflow.log_metric("mae", float(reg_mae))
        mlflow.log_metric("r2", float(reg_r2))

        for k, v in reg_feature_importance.items():
            mlflow.log_metric(f"feature_{k}", float(v))

        mlflow.sklearn.log_model(
            sk_model=reg_model,
            artifact_path="model",
            registered_model_name=f"{ticker}_regressor"
        )
        # Standard registry name mapping for backward compatibility
        mlflow.sklearn.log_model(
            sk_model=reg_model,
            artifact_path="model",
            registered_model_name=ticker
        )

    # 2. Classifier Logging
    with mlflow.start_run(run_name=f"{ticker}_classifier"):
        mlflow.set_tag("ticker", ticker)
        mlflow.set_tag("model_type", clf_model_type)
        mlflow.set_tag("task", "classification")
        
        for p_name, p_val in clf_params.items():
            mlflow.log_param(p_name, p_val)
        mlflow.log_param("train_rows", len(X_train_c))
        mlflow.log_metric("accuracy", float(clf_acc))

        for k, v in clf_feature_importance.items():
            mlflow.log_metric(f"feature_{k}", float(v))

        mlflow.sklearn.log_model(
            sk_model=clf_model,
            artifact_path="model",
            registered_model_name=f"{ticker}_classifier"
        )

    print(
        f"[DONE] {ticker} | "
        f"Reg({reg_model_type}): R2={reg_r2:.4f} | "
        f"Clf({clf_model_type}): Acc={clf_acc:.4f}"
    )


# -----------------------------
# Main
# -----------------------------

def main():
    target_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    for test_uri in [target_uri, "http://localhost:5000", "http://127.0.0.1:5000"]:
        try:
            mlflow.set_tracking_uri(test_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)
            print(f"[INFO] MLflow tracking URI set to {test_uri}")
            break
        except Exception as e:
            print(f"[WARN] Could not connect to MLflow at {test_uri}: {e}")

    tickers = os.getenv(
        "TRAIN_TICKERS"
    )

    if tickers:
        ticker_list = [
            t.strip()
            for t in tickers.split(",")
        ]
    else:
        ticker_list = DEFAULT_TICKERS

    print(
        f"[INFO] Training start "
        f"{datetime.now()}"
    )

    for ticker in ticker_list:

        try:
            train_ticker(ticker)

        except Exception as e:

            print(
                f"[ERROR] {ticker}: {e}"
            )

    print(
        f"[INFO] Training complete "
        f"{datetime.now()}"
    )


if __name__ == "__main__":
    main()