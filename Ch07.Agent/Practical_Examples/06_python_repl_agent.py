import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# 0. 환경 변수 로드
load_dotenv()

# ==========================================
# 1. 도구 설정 (Python REPL)
# ==========================================
# Python 코드를 실행할 수 있는 도구입니다. 보안에 주의해야 합니다.
# 로컬 환경에서 실행되므로 파일 시스템 접근 등이 가능합니다.
python_repl_tool = PythonREPLTool()

tools = [python_repl_tool]

# ==========================================
# 2. 에이전트 설정
# ==========================================

# 코드 생성 능력은 gpt-4o가 탁월합니다.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# System Prompt를 통해 Python 전문가 페르소나 부여
system_message = "당신은 파이썬 코드를 작성하고 실행하여 사용자의 질문을 해결하는 전문가입니다. 문제 해결을 위해 필요한 코드를 생성하고 실행 결과를 확인한 뒤 답변하세요."

agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

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
            content = str(msg.content)
            # 결과가 너무 길면 자름
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"  ▷ [Tool Result] {preview}") 
        elif msg.type == "ai" and not msg.tool_calls:
            print(f"\nAgent: {msg.content}")
    print("-" * 50)

# ==========================================
# 3. 실습 시나리오
# ==========================================
print("=== Python REPL Agent (Code Interpreter) ===")

# 시나리오 1: 복잡한 수학 계산 (LLM이 직접 계산하기 어려운 것)
run_agent_and_print_process("1부터 100까지의 소수(prime number)를 모두 구해서 그 합계를 알려줘.")

# 시나리오 2: 데이터 처리 (가상의 데이터 생성 및 분석)
# 실제 파일이 없으므로 데이터를 직접 생성해서 처리하도록 유도
run_agent_and_print_process("Python의 random 모듈을 사용해서 100명의 학생 점수(0~100점)를 생성하고, 평균, 표준편차, 최고점, 최저점을 계산해줘.")
