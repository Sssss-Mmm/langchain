import os
import datetime
from dotenv import load_dotenv
from typing import Optional

from langchain_openai import ChatOpenAI
# from langchain.agents import AgentExecutor, create_tool_calling_agent (Removed)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# 0. 환경 변수 로드
load_dotenv()

# ==========================================
# 1. 나만의 Custom Tool 만들기 (핵심!)
# ==========================================

@tool
def list_files(directory: Optional[str] = ".") -> str:
    """
    지정된 경로의 파일 목록을 반환합니다. 
    경로를 입력하지 않으면 현재 디렉토리를 보여줍니다.
    """
    try:
        # 안전을 위해 현재 프로젝트 디렉토리 내부만 접근 가능하도록 제한하는 로직을 추가할 수도 있습니다.
        target_dir = directory if directory else "."
        files = os.listdir(target_dir)
        return f"Directory '{target_dir}' contents:\n" + "\n".join(files)
    except Exception as e:
        return f"Error reading directory: {e}"

@tool
def get_system_time(format: Optional[str] = "%Y-%m-%d %H:%M:%S") -> str:
    """
    현재 시스템 시간을 반환합니다.
    """
    return datetime.datetime.now().strftime(format)

@tool
def create_report_file(filename: str, content: str) -> str:
    """
    주어진 내용으로 리포트 파일을 생성합니다.
    파일 이름과 내용을 입력받습니다.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully created file: {filename}"
    except Exception as e:
        return f"Error creating file: {e}"

# 도구 리스트 정의
tools = [list_files, get_system_time, create_report_file]


# ==========================================
# 2. 에이전트 설정 및 실행
# ==========================================

llm = ChatOpenAI(model="gpt-4o", temperature=0)

from langgraph.prebuilt import create_react_agent

# ==========================================
# 2. 에이전트 설정 및 실행 (LangGraph 사용)
# ==========================================

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# LangGraph의 prebuilt agent 사용 (가장 최신 권장 방식)
# 이 방식은 AgentExecutor보다 더 유연하고 투명합니다.
agent_executor = create_react_agent(llm, tools)


# ==========================================
# 3. 실습 시나리오
# ==========================================
print("=== System Admin Agent (Custom Tools via LangGraph) ===")

# 시나리오 1: 시간 확인 및 파일 목록 조회
print("\n[Case 1] 현재 시간과 파일 목록 확인")
query1 = "지금 몇 시야? 그리고 현재 폴더에 무슨 파일들이 있는지 알려줘."
print(f"User: {query1}")

# LangGraph Agent는 invoke 시 {"messages": [...]} 형태를 받습니다.
result1 = agent_executor.invoke({"messages": [("user", query1)]})
print(f"Agent: {result1['messages'][-1].content}")

# 시나리오 2: 정보를 조합하여 파일 생성
print("\n[Case 2] 리포트 생성")
query2 = "현재 폴더의 파일 목록을 바탕으로 'file_summary.txt'라는 파일을 만들어줘. 내용에는 생성 시각도 포함해줘."
print(f"User: {query2}")

# 이전 대화 내용을 포함하여 실행하려면 메시지를 누적해야 하지만, 
# 여기서는 독립적인 실행으로 보여줍니다.
result2 = agent_executor.invoke({"messages": [("user", query2)]})
print(f"Agent: {result2['messages'][-1].content}")

print("\n--- 실습 완료: 각 단계별로 Agent가 어떤 도구를 호출했는지 확인해보세요! ---")
