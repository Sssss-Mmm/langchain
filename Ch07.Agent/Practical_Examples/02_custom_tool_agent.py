import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 0. 환경 변수 로드
load_dotenv()

# 1. 사용자 정의 도구(Custom Tool) 만들기
# @tool 데코레이터를 사용하면 파이썬 함수를 쉽게 도구로 변환할 수 있습니다.
# Docstring은 에이전트가 이 도구를 언제 사용해야 하는지 판단하는 기준이 되므로 상세히 적어야 합니다.

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters."""
    print(f"--- [Tool Log] Calculating length for: '{text}' ---")
    # 공백 포함 길이 반환
    return len(text)

@tool
def reverse_text(text: str) -> str:
    """Reverses the given text string."""
    print(f"--- [Tool Log] Reversing text: '{text}' ---")
    return text[::-1]

tools = [get_text_length, reverse_text]

# 2. LLM 및 에이전트 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 3. 실행 예제
print("=== Custom Tool Agent ===")
question = "단어 'Supercalifragilisticexpialidocious'의 글자 수는 몇 개고, 거꾸로 쓰면 뭐야?"
print(f"\n질문: {question}")

result = agent_executor.invoke({"input": question})
print(f"\n최종 답변: {result['output']}")
