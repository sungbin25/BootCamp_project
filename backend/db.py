"""
MySQL 연결 설정.
UR 요구사항 자체는 대부분 실시간 API 응답이라 영속 저장이 필수는 아니지만,
'조회 이력'과 '예측 결과'를 MySQL에 남겨서 추후 모니터링/재현에 쓸 수 있게 합니다.
"""
import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

MYSQL_HOST = os.environ.get("MYSQL_HOST", "mysql")
MYSQL_PORT = os.environ.get("MYSQL_PORT", "3306")
MYSQL_USER = os.environ.get("MYSQL_USER", "stockapp")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "stockapp")
MYSQL_DB = os.environ.get("MYSQL_DATABASE", "stock_insight")

DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=280)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), index=True)
    horizon = Column(String(10))            # 1d / 1w / 1m
    base_date = Column(String(20))
    base_close = Column(Float)
    predicted_close = Column(Float)
    predicted_change_pct = Column(Float)
    reasoning_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class NewsAnalysisLog(Base):
    __tablename__ = "news_analysis_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), index=True)
    event_date = Column(String(20))
    price_change_pct = Column(Float)
    summary = Column(Text)
    impact_reasons_json = Column(Text)
    impact_score = Column(Integer)          # 1~5 (★ 개수)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
