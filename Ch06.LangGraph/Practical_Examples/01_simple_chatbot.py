import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 0. 환경 변수 로드
load_dotenv()

# 1. 상태(State) 정의
# 메시지 리스트를 관리하며, 새로운 메시지가 추가될 때마다 리스트에 append 됩니다.
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. 그래프 생성
graph_builder = StateGraph(State)

# 3. LLM 설정
llm = ChatOpenAI(model="gpt-4o")

# 4. 노드 함수 정의
def chatbot(state: State):
    """
    현재 상태(대화 기록)를 LLM에 전달하고 응답을 생성합니다.
    """
    print("\n--- Assistant 생각 중... ---")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 5. 노드 추가 및 엣지 연결
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 6. 그래프 컴파일 (실행 가능한 앱으로 변환)
graph = graph_builder.compile()

# 7. 실행 루프
print("=== LangGraph 챗봇 (종료하려면 'q' 입력) ===")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # 스트리밍 실행
    # "messages" 키에 튜플 (role, content) 형태로 전달하거나 HumanMessage 객체로 전달 가능
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            if "messages" in value:
                # 마지막 추가된 메시지(Assistant의 응답) 출력
                last_msg = value["messages"][-1]
                print(f"Assistant: {last_msg.content}")
