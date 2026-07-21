"""
공통 감성 분석 사전 및 유틸리티 모듈 (UR-07 / Spark Jobs / Backend 공용)
"""

POSITIVE_WORDS = ["상승", "호재", "성장", "역대", "최고", "개선", "흑자", "급등", "상향"]
NEGATIVE_WORDS = ["하락", "악재", "부진", "손실", "적자", "급락", "우려", "하향", "위기"]

BULLISH_EXPLICIT_WORDS = [
    "상한가", "떡상", "풀매수", "가즈아", "가자", "드가자", "매수추천", "대박", "익절", "줍줍",
    "상승세", "탑승", "호재", "급등", "흑자전환", "신고가", "영차", "떡상각", "매수하자",
    "양전", "주워담", "오르", "날아가", "양전하고", "갈 거", "안전한", "가겠다", "상승",
    "사자", "살까", "추천", "반등", "양전했", "양전하", "매수", "타점", "담자", "수익", "목표가",
    "양전했네", "양전해서", "양전하네", "안전한", "안전", "사야 하는", "사야하는", "무조건 오를",
    "올라갈거", "상승장", "우상향", "사라", "달려"
]

BEARISH_EXPLICIT_WORDS = [
    "하한가", "떡락", "풀매도", "폭락", "나락", "상폐", "망함", "손절",
    "개미지옥", "개잡주", "개잡", "설사", "한강", "악재", "급락", "신저가", "손절치자", "도망쳐",
    "지옥", "인버스", "음전", "가난", "하락", "-5퍼", "-3퍼", "-10퍼", "-5%", "-3%", "-10%",
    "더하락", "더떨어졌", "더떨어", "떨어졌", "떨어지", "하락시키", "축하합니다", "뒤쳐지", "망했",
    "처물", "물렸다", "물린", "매도", "내려", "하락하", "조정", "밀리", "내려가", "하락하는구나",
    "던지고 보자", "던지기시작", "던지기", "던지네", "던지자", "던져", "떠나라", "탈출해야하나",
    "탈출해야", "탈출", "빼자", "뺀다", "빼는", "아작", "처박", "박살", "폭망", "거지",
    "거지될때까지", "하락빔", "폭탄투하", "폭탄", "금융위기", "빨리 끝났으면", "아작내는구나"
]

COMMUNITY_POSITIVE_WORDS = POSITIVE_WORDS + BULLISH_EXPLICIT_WORDS
COMMUNITY_NEGATIVE_WORDS = NEGATIVE_WORDS + BEARISH_EXPLICIT_WORDS

SPAM_PATTERNS = [
    "카톡", "텔레그램", "리딩방", "무료체험", "http://", "https://", "010-",
    "VIP", "적중률", "상담받기", "문자주세요", "추천주", "카카오톡"
]

NEWS_EXPLICIT_WORDS = [
    "[속보]", "[단독]", "[공시]", "[기사]", "속보", "공시", "리포트", "특징주", "실적발표"
]

QUESTION_EXPLICIT_WORDS = [
    "주식 시작", "공부해야", "주린이 질문", "초보 질문", "어떻게 공부", "어떻게 해야", "왜 올라갈거", "알려주세요", "오긴올까"
]

HUMOR_EXPLICIT_WORDS = [
    "ㅋㅋㅋㅋㅋㅋ", "ㅎㅎㅎㅎㅎㅎ", "웃기네", "능지", "감자전"
]

GARBLED_CHARS = set("臾ㅻ쇱깆醫紐猷諛곕⑤④二쇱吏⑤⑥⑦⑧⑨⑩")

def is_garbled_korean(text: str) -> bool:
    """깨진 EUC-KR / UTF-8 한글 인코딩 오복원 문자 패킷 판단"""
    if not text:
        return True
    return any(ch in GARBLED_CHARS for ch in text)


def is_spam_post(text: str) -> bool:
    """광고/스팸 게시글 여부 판단"""
    if not text:
        return True
    if is_garbled_korean(text):
        return True
    lower = text.lower()
    return any(p in lower for p in SPAM_PATTERNS)


def pre_classify_intent_dict(title: str) -> dict | None:
    """
    주식 용어 사전 기반 1차 판별 (투자 의도 Bullish/Bearish 우선 판별).
    단순 '?'나 'ㅋㅋ'는 무조건 유머/질문으로 치부하지 않고 맥락(상승/하락 의도)을 최우선 평가합니다.
    """
    if not title or is_spam_post(title):
        return None

    title_clean = title.strip()
    import re

    # 1. 뉴스/공시 표현
    if any(w in title_clean for w in NEWS_EXPLICIT_WORDS):
        return {"intent": "News", "method": "dictionary", "method_tag": "⚡사전", "score": 0.0}

    # 2. 상승(Bullish) / 하락(Bearish) 매수·매도 의도 우선 검사 (정규식 포함)
    has_bull = any(w in title_clean for w in BULLISH_EXPLICIT_WORDS) or bool(re.search(r'\b\d+층\b|양전|드가자|\+\d+%|오르|안전|가겠|반등|상승|사자|수익|사야|우상향', title_clean))
    has_bear = any(w in title_clean for w in BEARISH_EXPLICIT_WORDS) or bool(re.search(r'-\d+%|-\d+퍼|인버스|음전|더떨|떨어|하락|신저가|개잡|뒤쳐|축하|던지|떠나|탈출|뺀다|아작|폭탄|거지|폭망|하락빔', title_clean))

    if has_bull and not has_bear:
        return {"intent": "Bullish", "method": "dictionary", "method_tag": "⚡사전", "score": 0.8}

    if has_bear and not has_bull:
        return {"intent": "Bearish", "method": "dictionary", "method_tag": "⚡사전", "score": -0.8}

    # 3. 명확한 순수 초보 질문/학습 표현 (상승/하락 맥락이 없을 때만)
    if any(w in title_clean for w in QUESTION_EXPLICIT_WORDS):
        return {"intent": "Question", "method": "dictionary", "method_tag": "⚡사전", "score": 0.0}

    # 4. 명확한 순수 유머/밈 표현 (상승/하락 맥락이 없을 때만)
    if any(w in title_clean for w in HUMOR_EXPLICIT_WORDS):
        return {"intent": "Humor", "method": "dictionary", "method_tag": "⚡사전", "score": 0.0}

    # 애매하거나 맥락 파악이 필요한 문장은 모두 2차 LLM 처리로 넘김
    return None



def simple_sentiment_dict(text: str, pos_words: list = None, neg_words: list = None) -> dict:
    """
    텍스트에 대한 간이 감성 딕셔너리 {"positive": float, "neutral": float, "negative": float} 반환.
    news_service.py 등에서 사용.
    """
    if not text:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    if pos_words is None:
        pos_words = POSITIVE_WORDS
    if neg_words is None:
        neg_words = NEGATIVE_WORDS

    lower = text.lower()
    pos = 1 if any(w in lower for w in pos_words) else 0
    neg = 1 if any(w in lower for w in neg_words) else 0

    if pos and not neg:
        return {"positive": 0.8, "neutral": 0.2, "negative": 0.0}
    if neg and not pos:
        return {"positive": 0.0, "neutral": 0.2, "negative": 0.8}
    return {"positive": 0.1, "neutral": 0.8, "negative": 0.1}


def simple_sentiment_score(text: str, pos_words: list = None, neg_words: list = None) -> float:
    """
    텍스트에 대한 단어 개수 기반 감성 점수 (-1.0 ~ 1.0) 반환.
    clean_and_join.py 등 Spark UDF에서 사용.
    """
    if not text:
        return 0.0

    if pos_words is None:
        pos_words = POSITIVE_WORDS
    if neg_words is None:
        neg_words = NEGATIVE_WORDS

    pos = sum(text.count(w) for w in pos_words)
    neg = sum(text.count(w) for w in neg_words)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total
