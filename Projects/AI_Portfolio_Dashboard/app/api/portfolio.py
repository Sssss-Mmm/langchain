from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os

# 프로젝트 루트 경로 추가 (상위 디렉토리의 agents 접근)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.portfolio_analyst import get_portfolio_analyst_agent

router = APIRouter()

class PortfolioRequest(BaseModel):
    query: str
    portfolio: Dict[str, Any]  # 포트폴리오 JSON 데이터

class AnalysisResponse(BaseModel):
    answer: str
    tool_calls: List[str] = []

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_portfolio(request: PortfolioRequest):
    """
    포트폴리오 분석 에이전트를 실행하여 질문에 대한 답변을 생성합니다.
    """
    try:
        agent = get_portfolio_analyst_agent()
        
        # 포트폴리오 정보를 문자열 컨텍스트로 변환
        context = f"\n\n[Portfolio Context]\n{request.portfolio}"
        full_query = request.query + context
        
        # 에이전트 실행 (LangGraph)
        # LangGraph의 invoke는 기본적으로 동기 함수이므로, FastAPI의 스레드풀에서 실행됩니다.
        response = agent.invoke({"messages": [("user", full_query)]})
        
        # 결과 파싱
        # 마지막 메시지가 AI의 최종 답변이라고 가정
        final_answer = ""
        tool_calls = []
        
        if 'messages' in response and response['messages']:
            final_answer = response['messages'][-1].content
            
            # 도구 사용 이력 추출
            for msg in response['messages']:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(tc.get('name', 'unknown'))
        
        return AnalysisResponse(answer=str(final_answer), tool_calls=tool_calls)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc()) # 서버 로그에 에러 출력
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
