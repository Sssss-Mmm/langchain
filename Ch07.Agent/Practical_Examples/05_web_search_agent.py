import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

# 0. 환경 변수 로드
load_dotenv()

# ==========================================
# 1. 도구 설정 (Web Search)
# ==========================================
# DuckDuckGoZeroSearchRun은 별도의 API Key 없이 무료로 검색 가능합니다.
search_tool = DuckDuckGoSearchRun()

tools = [search_tool]

# ==========================================
# 2. 에이전트 설정
# ==========================================

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# LangGraph ReAct Agent 생성
agent_executor = create_react_agent(llm, tools)

def run_agent_and_print_process(query: str):
    print(f"\nUser: {query}")
    print("-" * 50)
    
    # 에이전트 실행
    result = agent_executor.invoke({"messages": [("user", query)]})
    
    # 실행 과정 및 결과 출력
    for msg in result['messages']:
        if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            for tool_call in msg.tool_calls:
                print(f"  ▶ [Tool Call] {tool_call['name']}: {tool_call['args']}")
        elif msg.type == "tool":
            # 검색 결과가 너무 길 수 있으므로 일부만 출력
            content = str(msg.content)
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"  ▷ [Tool Result] {preview}") 
        elif msg.type == "ai" and not msg.tool_calls:
            print(f"\nAgent: {msg.content}")
    print("-" * 50)

# ==========================================
# 3. 실습 시나리오
# ==========================================
print("=== Web Search Agent (DuckDuckGo Analysis) ===")

# 시나리오 1: 단순 정보 검색
run_agent_and_print_process(input("User: " ))

# 시나리오 2: 최신 뉴스 검색 (모델 학습 데이터 시점 이후의 정보)
run_agent_and_print_process("2024년 이후 발표된 아이폰 16(또는 최신 모델)에 대한 루머나 뉴스를 찾아줘.")
