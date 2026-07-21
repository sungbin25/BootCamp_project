# pyspark의 spark-submit이 동작하려면 JDK가 필요합니다.
# 공식 apache/airflow 이미지에는 JDK가 없어서 커스텀 이미지로 추가합니다.
FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Spark 클러스터(apache/spark:3.5.3)와 동일한 버전으로 맞춰야
# spark-submit이 원격 마스터에 정상적으로 job을 제출할 수 있습니다.
RUN pip install --no-cache-dir \
    pyspark==3.5.3 \
    yfinance \
    beautifulsoup4 \
    requests \
    pandas \
    pyarrow \
    minio \
    mlflow==2.16.0 \
    xgboost
