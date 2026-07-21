"""
주식 예측 파이프라인 메인 DAG

흐름:
  [수집: 가격/뉴스/커뮤니티] -> [Spark 정제+피처결합]
  -> [재학습 필요 여부 분기] -> (필요시) MLflow 학습 -> [LLM 요약 생성]

스케줄: 매일 장 마감 후 1회 (한국장 기준 15:40 KST)
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

TICKERS = ["005930.KS", "000660.KS", "035420.KS"]  # 삼성전자, SK하이닉스, 네이버

default_args = {
    "owner": "portfolio-demo",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="stock_pipeline_dag",
    default_args=default_args,
    description="정량+정성 데이터 수집 -> Spark 정제 -> MLflow 학습 -> LLM 요약",
    schedule_interval="40 15 * * 1-5",  # 평일 15:40 KST (장 마감 후)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["stock", "llm", "portfolio"],
) as dag:

    start = EmptyOperator(task_id="start")

    # ---------------- 1. 데이터 수집 ----------------
    crawl_price = BashOperator(
        task_id="crawl_price",
        bash_command=(
            "python /opt/airflow/crawlers/crawl_price.py "
            f"--tickers {' '.join(TICKERS)} "
            "--out /opt/airflow/data/raw/price"
        ),
    )

    crawl_news_tasks = []
    for ticker in TICKERS:
        t = BashOperator(
            task_id=f"crawl_news_{ticker.split('.')[0]}",
            bash_command=(
                f"python /opt/airflow/crawlers/crawl_news.py "
                f"--query '{ticker}' --ticker {ticker} "
                "--out /opt/airflow/data/raw/news"
            ),
        )
        crawl_news_tasks.append(t)

    crawl_community_tasks = []
    for ticker in TICKERS:
        code = ticker.split(".")[0]
        t = BashOperator(
            task_id=f"crawl_community_{code}",
            bash_command=(
                f"python /opt/airflow/crawlers/crawl_community.py "
                f"--ticker {code} --pages 3 "
                "--out /opt/airflow/data/raw/community"
            ),
        )
        crawl_community_tasks.append(t)

    # ---------------- 2. Spark 정제 ----------------
    spark_clean = BashOperator(
        task_id="spark_clean_and_join",
        bash_command=(
            "spark-submit --master local[*] "
            "/opt/spark_jobs/clean_and_join.py "
            "--raw_dir /opt/data/raw --out /opt/data/processed/features.parquet"
        ),
    )

    # ---------------- 3. 재학습 필요 여부 분기 ----------------
    def check_retrain_needed(**context):
        """
        간단 기준: 신규 데이터 누적 row 수가 임계치를 넘었는지 확인.
        실무에서는 여기에 모델 드리프트(예측 정확도 하락) 체크 로직을 추가하면 더 설득력 있음.
        """
        import pandas as pd
        df = pd.read_parquet("/opt/airflow/data/processed/features.parquet")
        threshold = 50
        if len(df) >= threshold:
            return "trigger_mlflow_training"
        return "skip_training"

    branch = BranchPythonOperator(
        task_id="check_retrain_needed",
        python_callable=check_retrain_needed,
    )

    trigger_training = BashOperator(
        task_id="trigger_mlflow_training",
        bash_command="python /opt/airflow/backend/train_model.py && python /opt/airflow/mlflow_scripts/train_price_direction.py",
    )

    skip_training = EmptyOperator(task_id="skip_training")

    join_after_branch = EmptyOperator(
        task_id="join_after_branch",
        trigger_rule="none_failed_min_one_success",
    )

    # ---------------- 4. LLM 요약 생성 ----------------
    generate_llm_summary = BashOperator(
        task_id="generate_llm_summary",
        bash_command="python /opt/airflow/mlflow_scripts/generate_summary.py",
    )

    end = EmptyOperator(task_id="end")

    # ---------------- DAG 의존성 ----------------
    start >> crawl_price
    start >> crawl_news_tasks
    start >> crawl_community_tasks

    [crawl_price, *crawl_news_tasks, *crawl_community_tasks] >> spark_clean
    spark_clean >> branch
    branch >> trigger_training >> join_after_branch
    branch >> skip_training >> join_after_branch
    join_after_branch >> generate_llm_summary >> end
