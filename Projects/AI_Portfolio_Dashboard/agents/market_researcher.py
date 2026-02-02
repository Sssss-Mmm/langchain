"""
Market Researcher Agent - 시장 리서치 에이전트
실시간 시장 정보 및 뉴스 분석
"""
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stock_data import stock_tools
from tools.news_analyzer import news_tools
from config.settings import OPENAI_MODEL


SYSTEM_PROMPT = """당신은 금융 시장 리서치 전문가입니다. 최신 시장 동향과 뉴스를 분석하여 투자 인사이트를 제공합니다.

## 역할
- 개별 종목의 최신 정보를 조회합니다.
- 시장 전반의 동향을 파악합니다.
- 특정 섹터/업종의 전망을 분석합니다.
- 관련 뉴스를 검색하고 요약합니다.

## 응답 가이드라인
1. 최신 정보를 기반으로 분석하세요.
2. 긍정적, 부정적 요인을 균형있게 제시하세요.
3. 불확실한 정보는 명확히 구분하세요.
4. 투자 의사결정은 사용자가 해야 함을 명시하세요.

## 사용 가능한 도구
- 주식 현재가/과거 데이터/재무정보 조회
- 종목 뉴스 검색
- 시장 분석 검색
- 섹터 뉴스 검색

항상 한국어로 응답하세요.
"""


def get_market_researcher_agent():
    """
    시장 리서치 에이전트를 생성합니다.
    """
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1,
    )
    
    all_tools = stock_tools + news_tools
    
    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    
    return agent
