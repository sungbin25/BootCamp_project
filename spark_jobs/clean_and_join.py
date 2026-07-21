"""
Spark 정제 파이프라인
1. 가격 데이터: 결측치 처리, 이동평균/변동률 피처 생성
2. 뉴스/커뮤니티 텍스트: HTML 태그 제거, 감성 점수 부여(간이 사전 기반 -> 추후 KR-FinBert로 교체 가능)
3. 날짜/종목 기준으로 조인해 최종 피처 테이블 생성

실행: spark-submit --master spark://spark-master:7077 clean_and_join.py \
        --raw_dir /opt/data/raw --out /opt/data/processed/features.parquet
"""
import argparse
import re

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F, types as T

import os
import sys

# 백엔드/공통 감성 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.append("/opt/airflow/backend")
sys.path.append("/opt/backend")

from sentiment_utils import (
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    simple_sentiment_score as simple_sentiment,
)


def strip_html(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"<[^>]+>", "", text)


def build_spark(app_name: str = "stock_feature_pipeline") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .getOrCreate()
    )


def clean_price(df):
    w = Window.partitionBy("ticker").orderBy("date")
    df = (
        df.withColumn("prev_close", F.lag("Close", 1).over(w))
          .withColumn("prev_close_pct_change",
                      (F.col("Close") - F.col("prev_close")) / F.col("prev_close"))
          .withColumn("ma5", F.avg("Close").over(w.rowsBetween(-4, 0)))
          .withColumn("ma20", F.avg("Close").over(w.rowsBetween(-19, 0)))
          .withColumn("ma5_gap", (F.col("Close") - F.col("ma5")) / F.col("ma5"))
          .withColumn("ma20_gap", (F.col("Close") - F.col("ma20")) / F.col("ma20"))
          .withColumn("vol_avg20", F.avg("Volume").over(w.rowsBetween(-19, 0)))
          .withColumn("vol_std20", F.stddev("Volume").over(w.rowsBetween(-19, 0)))
          .withColumn("volume_zscore",
                      F.when(F.col("vol_std20") > 0,
                             (F.col("Volume") - F.col("vol_avg20")) / F.col("vol_std20"))
                       .otherwise(0.0))
          # 5일 평활화 미래 수익률 및 방향성 계산
          .withColumn("lead1", F.lead("Close", 1).over(w))
          .withColumn("lead2", F.lead("Close", 2).over(w))
          .withColumn("lead3", F.lead("Close", 3).over(w))
          .withColumn("lead4", F.lead("Close", 4).over(w))
          .withColumn("lead5", F.lead("Close", 5).over(w))
          .withColumn("future_avg", (F.col("lead1") + F.col("lead2") + F.col("lead3") + F.col("lead4") + F.col("lead5")) / 5.0)
          .withColumn("next_pct_change",
                      (F.col("future_avg") - F.col("Close")) / F.col("Close"))
          .withColumn(
              "direction_label",
              F.when(F.col("next_pct_change") > 0.005, 1)
               .when(F.col("next_pct_change") < -0.005, -1)
               .otherwise(0)
          )
    )
    return df.select(
        "ticker", "date", "Close", "Volume",
        "prev_close_pct_change", "ma5_gap", "ma20_gap", "volume_zscore",
        "direction_label",
    )


def clean_community(spark: SparkSession, path: str):
    df = spark.read.parquet(path)
    # 6자리 티커를 야후 파이낸스 형식(005930.KS)으로 변경
    df = df.withColumn(
        "ticker",
        F.when(F.length(F.col("ticker")) == 6, F.concat(F.col("ticker"), F.lit(".KS")))
         .otherwise(F.col("ticker"))
    )
    strip_udf = F.udf(strip_html, T.StringType())
    sentiment_udf = F.udf(simple_sentiment, T.DoubleType())

    df = (
        df.withColumn("clean_text", strip_udf(F.col("title")))
          .withColumn("sentiment_score", sentiment_udf(F.col("clean_text")))
          .withColumn("date", F.to_date(F.col("crawled_at")))
    )

    agg = (
        df.groupBy("ticker", "date")
          .agg(
              F.avg("sentiment_score").alias("community_sentiment"),
              F.count("*").alias("community_mention_count"),
          )
    )
    return agg


def clean_text_source(spark: SparkSession, path: str, text_col: str, date_col: str):
    df = spark.read.parquet(path)
    strip_udf = F.udf(strip_html, T.StringType())
    sentiment_udf = F.udf(simple_sentiment, T.DoubleType())

    df = (
        df.withColumn("clean_text", strip_udf(F.col(text_col)))
          .withColumn("sentiment_score", sentiment_udf(F.col("clean_text")))
          .withColumn("date", F.to_date(F.col(date_col)))
    )

    agg = (
        df.groupBy("ticker", "date")
          .agg(
              F.avg("sentiment_score").alias("avg_sentiment"),
              F.count("*").alias("mention_count"),
              F.stddev("sentiment_score").alias("sentiment_volatility"),
          )
    )
    w = Window.partitionBy("ticker").orderBy("date")
    agg = agg.withColumn("sentiment_lag1", F.lag("avg_sentiment", 1).over(w))
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="/opt/data/raw")
    parser.add_argument("--out", default="/opt/data/processed/features.parquet")
    args = parser.parse_args()

    spark = build_spark()

    import glob
    # 1. 가격 데이터 로드 (파일 매칭 실패시 에러 예방)
    price_files = glob.glob(f"{args.raw_dir}/price/*.parquet")
    if price_files:
        raw_price_df = spark.read.parquet(f"{args.raw_dir}/price/*.parquet")
        raw_price_df = raw_price_df.withColumn("date", F.to_date("Date"))
        raw_price_df = raw_price_df.dropDuplicates(["ticker", "date"])

        # 1. Macro economic features 계산
        w_date = Window.orderBy("date")

        # KOSPI
        kospi_df = raw_price_df.filter(F.col("ticker") == "^KS11").select("date", "Close", "Volume")
        kospi_features = (
            kospi_df
            .withColumn("kospi_prev_close", F.lag("Close", 1).over(w_date))
            .withColumn("kospi_return_1d", (F.col("Close") - F.col("kospi_prev_close")) / F.col("kospi_prev_close"))
            .withColumn("kospi_prev_close_5", F.lag("Close", 5).over(w_date))
            .withColumn("kospi_return_5d", (F.col("Close") - F.col("kospi_prev_close_5")) / F.col("kospi_prev_close_5"))
            .withColumn("kospi_ma20", F.avg("Close").over(w_date.rowsBetween(-19, 0)))
            .withColumn("kospi_ma20_gap", (F.col("Close") - F.col("kospi_ma20")) / F.col("kospi_ma20"))
            .withColumn("kospi_amount", F.col("Close") * F.col("Volume"))
            .withColumn("kospi_prev_amount", F.lag("kospi_amount", 1).over(w_date))
            .withColumn("market_turnover_change", (F.col("kospi_amount") - F.col("kospi_prev_amount")) / F.col("kospi_prev_amount"))
            .select("date", "kospi_return_1d", "kospi_return_5d", "kospi_ma20_gap", "market_turnover_change")
        )

        # KOSDAQ
        kosdaq_df = raw_price_df.filter(F.col("ticker") == "^KQ11").select("date", "Close")
        kosdaq_features = (
            kosdaq_df
            .withColumn("kosdaq_prev_close", F.lag("Close", 1).over(w_date))
            .withColumn("kosdaq_return_1d", (F.col("Close") - F.col("kosdaq_prev_close")) / F.col("kosdaq_prev_close"))
            .withColumn("kosdaq_prev_close_5", F.lag("Close", 5).over(w_date))
            .withColumn("kosdaq_return_5d", (F.col("Close") - F.col("kosdaq_prev_close_5")) / F.col("kosdaq_prev_close_5"))
            .select("date", "kosdaq_return_1d", "kosdaq_return_5d")
        )

        # USD/KRW
        usdkrw_df = raw_price_df.filter(F.col("ticker") == "KRW=X").select("date", "Close")
        usdkrw_features = (
            usdkrw_df
            .withColumn("usdkrw_prev_close", F.lag("Close", 1).over(w_date))
            .withColumn("usdkrw_return_1d", (F.col("Close") - F.col("usdkrw_prev_close")) / F.col("usdkrw_prev_close"))
            .withColumn("usdkrw_prev_close_5", F.lag("Close", 5).over(w_date))
            .withColumn("usdkrw_return_5d", (F.col("Close") - F.col("usdkrw_prev_close_5")) / F.col("usdkrw_prev_close_5"))
            .withColumn("usdkrw_ma20", F.avg("Close").over(w_date.rowsBetween(-19, 0)))
            .withColumn("usdkrw_ma20_gap", (F.col("Close") - F.col("usdkrw_ma20")) / F.col("usdkrw_ma20"))
            .select("date", "usdkrw_return_1d", "usdkrw_return_5d", "usdkrw_ma20_gap")
        )

        # VIX
        vix_df = raw_price_df.filter(F.col("ticker") == "^VIX").select("date", "Close")
        vix_features = (
            vix_df
            .withColumn("vix_prev_close", F.lag("Close", 1).over(w_date))
            .withColumn("vix_return_1d", (F.col("Close") - F.col("vix_prev_close")) / F.col("vix_prev_close"))
            .select("date", "vix_return_1d")
        )

        # Merge Macro Features
        macro_df = (
            kospi_features
            .join(kosdaq_features, "date", "left")
            .join(usdkrw_features, "date", "left")
            .join(vix_features, "date", "left")
        )

        # 2. Individual stock features
        macro_tickers = ["^KS11", "^KQ11", "KRW=X", "^VIX"]
        stock_raw_df = raw_price_df.filter(~F.col("ticker").isin(macro_tickers))
        price_df = clean_price(stock_raw_df)

        # 3. Join Macro Features
        price_df = price_df.join(macro_df, "date", "left")
    else:
        from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
        schema = StructType([
            StructField("ticker", StringType(), True),
            StructField("date", DateType(), True),
            StructField("Close", DoubleType(), True),
            StructField("Volume", DoubleType(), True),
            StructField("prev_close_pct_change", DoubleType(), True),
            StructField("ma5_gap", DoubleType(), True),
            StructField("ma20_gap", DoubleType(), True),
            StructField("volume_zscore", DoubleType(), True),
            StructField("direction_label", LongType(), True),
            StructField("kospi_return_1d", DoubleType(), True),
            StructField("kospi_return_5d", DoubleType(), True),
            StructField("kospi_ma20_gap", DoubleType(), True),
            StructField("market_turnover_change", DoubleType(), True),
            StructField("kosdaq_return_1d", DoubleType(), True),
            StructField("kosdaq_return_5d", DoubleType(), True),
            StructField("usdkrw_return_1d", DoubleType(), True),
            StructField("usdkrw_return_5d", DoubleType(), True),
            StructField("usdkrw_ma20_gap", DoubleType(), True),
            StructField("vix_return_1d", DoubleType(), True),
        ])
        price_df = spark.createDataFrame([], schema)

    # 2. 뉴스 데이터 로드 및 정제
    news_files = glob.glob(f"{args.raw_dir}/news/*.parquet")
    if news_files:
        news_agg = clean_text_source(spark, f"{args.raw_dir}/news/*.parquet",
                                      text_col="title", date_col="crawled_at")
        news_agg = news_agg.withColumn("news_sentiment", F.col("avg_sentiment"))
    else:
        from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
        schema = StructType([
            StructField("ticker", StringType(), True),
            StructField("date", DateType(), True),
            StructField("avg_sentiment", DoubleType(), True),
            StructField("news_sentiment", DoubleType(), True),
            StructField("mention_count", LongType(), True),
            StructField("sentiment_volatility", DoubleType(), True),
            StructField("sentiment_lag1", DoubleType(), True),
        ])
        news_agg = spark.createDataFrame([], schema)

    # 3. 커뮤니티 데이터 로드 및 정제
    community_files = glob.glob(f"{args.raw_dir}/community/*.parquet")
    if community_files:
        comm_agg = clean_community(spark, f"{args.raw_dir}/community/*.parquet")
    else:
        from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
        schema = StructType([
            StructField("ticker", StringType(), True),
            StructField("date", DateType(), True),
            StructField("community_sentiment", DoubleType(), True),
            StructField("community_mention_count", LongType(), True),
        ])
        comm_agg = spark.createDataFrame([], schema)

    # 4. 전체 데이터 조인
    feature_df = (
        price_df.join(news_agg, ["ticker", "date"], "left")
        .join(comm_agg, ["ticker", "date"], "left")
        .fillna({
            "avg_sentiment": 0.0,
            "news_sentiment": 0.0,
            "mention_count": 0,
            "sentiment_volatility": 0.0,
            "sentiment_lag1": 0.0,
            "community_sentiment": 0.0,
            "community_mention_count": 0,
            "kospi_return_1d": 0.0,
            "kospi_return_5d": 0.0,
            "kospi_ma20_gap": 0.0,
            "market_turnover_change": 0.0,
            "kosdaq_return_1d": 0.0,
            "kosdaq_return_5d": 0.0,
            "usdkrw_return_1d": 0.0,
            "usdkrw_return_5d": 0.0,
            "usdkrw_ma20_gap": 0.0,
            "vix_return_1d": 0.0,
        })
    )

    feature_df.write.mode("overwrite").parquet(args.out)
    print(f"[OK] feature table written to {args.out}, rows={feature_df.count()}")

    spark.stop()


if __name__ == "__main__":
    main()
