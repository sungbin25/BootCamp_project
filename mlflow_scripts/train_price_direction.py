"""
가격 방향(상승/하락/보합) 예측 모델 학습 스크립트
Airflow의 MLflow 트리거 task에서 spark-submit 대신 python으로 직접 호출되거나,
BashOperator/PythonOperator에서 실행됩니다.

입력: /opt/airflow/data/processed/features.parquet
      (Spark 단계에서 만든 가격 + 감성 피처 결합 테이블)
출력: MLflow Model Registry에 새 버전 등록
"""
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
FEATURE_PATH = os.environ.get("FEATURE_PATH", "/opt/airflow/data/processed/features.parquet")
MODEL_NAME = "stock_direction_classifier"

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
    "rsi14",
    "news_sentiment",
    "community_sentiment",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "volume_zscore",
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
TARGET_COL = "direction_label"   # -1(하락) / 0(보합) / 1(상승)


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

    # 4. volume ratio
    volume_mean = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / volume_mean.replace(0, np.nan)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

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

    # 7. Bollinger Position
    std20 = df["Close"].rolling(20).std()
    upper_band = df["ma20"] + 2 * std20
    lower_band = df["ma20"] - 2 * std20
    band_width = upper_band - lower_band
    df["bb_position"] = (df["Close"] - lower_band) / band_width.replace(0, np.nan)
    df["bb_position"] = df["bb_position"].fillna(0.5)

    # 8. Volume Z-Score
    volume_std = df["Volume"].rolling(20).std()
    df["volume_zscore"] = (df["Volume"] - volume_mean) / volume_std.replace(0, np.nan)
    df["volume_zscore"] = df["volume_zscore"].fillna(0.0)

    return df


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    
    # Calculate rolling sentiment features for each ticker
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Pre-populate rolling sentiment columns
    df["news_sentiment_3d"] = df.groupby("ticker")["news_sentiment"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["news_sentiment_7d"] = df.groupby("ticker")["news_sentiment"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["community_sentiment_3d"] = df.groupby("ticker")["community_sentiment"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["community_sentiment_7d"] = df.groupby("ticker")["community_sentiment"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    # Apply build_features to generate technical indicators
    df = df.groupby("ticker", group_keys=False).apply(build_features)
    
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("stock_direction_prediction")

    df = load_features(FEATURE_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. RandomForestClassifier
    rf_params = {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5}
    rf = RandomForestClassifier(random_state=42, **rf_params)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average="macro")

    model = rf
    model_type = "RandomForestClassifier"
    preds = rf_preds
    acc = rf_acc
    f1 = rf_f1
    params = rf_params

    # 2. XGBClassifier comparison
    try:
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        xgb_clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            random_state=42,
            n_jobs=-1
        )
        xgb_clf.fit(X_train, y_train_enc)
        xgb_preds_enc = xgb_clf.predict(X_test)
        xgb_preds = le.inverse_transform(xgb_preds_enc)
        xgb_acc = accuracy_score(y_test, xgb_preds)
        xgb_f1 = f1_score(y_test, xgb_preds, average="macro")

        print(f"[COMPARE CLASSIFIER] XGBoost F1={xgb_f1:.4f} vs RandomForest F1={rf_f1:.4f}")
        if xgb_f1 > rf_f1:
            model = xgb_clf
            model_type = "XGBClassifier"
            preds = xgb_preds
            acc = xgb_acc
            f1 = xgb_f1
            params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03}
    except Exception as xgb_err:
        print(f"[WARN] Failed to train XGBClassifier: {xgb_err}")

    with mlflow.start_run():
        mlflow.set_tag("model_type", model_type)
        mlflow.log_params(params)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        print(classification_report(y_test, preds))

        # 피처 중요도
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance_path = "/tmp/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print(f"Run finished ({model_type}). accuracy={acc:.4f}, f1_macro={f1:.4f}")


if __name__ == "__main__":
    main()
