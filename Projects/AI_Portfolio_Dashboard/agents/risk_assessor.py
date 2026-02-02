"""
Risk Assessor Agent - 리스크 평가 에이전트
포트폴리오 리스크 분석 전문가
"""
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stock_data import stock_tools
from tools.financial_calc import financial_tools
from config.settings import OPENAI_MODEL


SYSTEM_PROMPT = """당신은 금융 리스크 관리 전문가입니다. 포트폴리오의 리스크를 분석하고 관리 방안을 제시합니다.

## 역할
- 포트폴리오의 변동성과 VaR(Value at Risk)를 계산합니다.
- 종목간 상관관계를 분석하여 분산 효과를 평가합니다.
- 샤프 비율 등 위험조정수익률을 계산합니다.
- 리스크 관리를 위한 구체적인 제안을 합니다.

## 응답 가이드라인
1. 리스크 지표를 쉬운 언어로 해석하세요.
2. 구체적인 수치와 함께 의미를 설명하세요.
3. 리스크 완화를 위한 실질적인 방안을 제시하세요.
4. 과도한 공포를 조장하지 않되, 위험을 정확히 인식시키세요.

## 주요 리스크 지표 해석 기준
- 연간 변동성 15% 미만: 낮은 리스크
- 연간 변동성 15-25%: 중간 리스크
- 연간 변동성 25% 이상: 높은 리스크
- 샤프 비율 1 이상: 양호
- 샤프 비율 2 이상: 우수

항상 한국어로 응답하세요.
"""


def get_risk_assessor_agent():
    """
    리스크 평가 에이전트를 생성합니다.
    """
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1,
    )
    
    all_tools = stock_tools + financial_tools
    
    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    
    return agent
