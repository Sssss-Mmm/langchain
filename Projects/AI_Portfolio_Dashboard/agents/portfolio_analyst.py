"""
Portfolio Analyst Agent - 포트폴리오 분석 에이전트
LangGraph ReAct Agent 기반 포트폴리오 종합 분석
"""
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stock_data import stock_tools
from tools.financial_calc import financial_tools
from tools.news_analyzer import news_tools
from config.settings import OPENAI_MODEL


SYSTEM_PROMPT = """당신은 전문 투자 분석가입니다. 사용자의 포트폴리오를 분석하고 투자 인사이트를 제공합니다.

## 역할
- 보유 종목의 현재 가치와 수익률을 분석합니다.
- 섹터별 분산 투자 현황을 평가합니다.
- 리스크 지표(변동성, VaR, 샤프비율)를 계산하고 해석합니다.
- 관련 뉴스와 시장 동향을 요약합니다.

## 응답 가이드라인
1. 항상 데이터에 기반한 분석을 제공하세요.
2. 복잡한 금융 용어는 쉽게 풀어서 설명하세요.
3. 분석 결과를 바탕으로 구체적인 제안을 하세요.
4. 투자 결정은 사용자의 몫임을 명시하세요.

## 사용 가능한 도구
- 주식 현재가/과거 데이터/재무정보 조회
- 포트폴리오 가치 및 리스크 계산
- 섹터 배분 분석
- 뉴스 검색

항상 한국어로 응답하세요.
"""


def get_portfolio_analyst_agent():
    """
    포트폴리오 분석 에이전트를 생성합니다.
    """
    # LLM 초기화
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1,
    )
    
    # 모든 도구 통합
    all_tools = stock_tools + financial_tools + news_tools
    
    # ReAct Agent 생성
    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    
    return agent
