from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.market_researcher import get_market_researcher_agent

router = APIRouter()

class MarketRequest(BaseModel):
    query: str

class MarketResponse(BaseModel):
    answer: str
    tool_calls: List[str] = []

@router.post("/research", response_model=MarketResponse)
async def research_market(request: MarketRequest):
    try:
        agent = get_market_researcher_agent()
        
        # 에이전트 실행
        response = agent.invoke({"messages": [("user", request.query)]})
        
        # 결과 파싱
        final_answer = ""
        tool_calls = []
        
        if 'messages' in response and response['messages']:
            final_answer = response['messages'][-1].content
            for msg in response['messages']:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(tc.get('name', 'unknown'))
        
        return MarketResponse(answer=str(final_answer), tool_calls=tool_calls)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
