import os
import json
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_core.tools import tool

# 0. 환경 변수 로드
load_dotenv()

# 1. 도구(Tools) 정의
# Tavily 검색 도구 사용 (API KEY 필요)
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

# 2. 상태(State) 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 3. 그래프 빌더 초기화
graph_builder = StateGraph(State)

# 4. LLM 설정 (Tool Binding)
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# 5. 노드 함수 정의

def chatbot(state: State):
    """
    LLM이 메시지를 처리하고 응답을 생성하는 노드
    """
    print("\n--- Assistant (LLM) 처리 중... ---")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

class BasicToolNode:
    """
    LLM이 도구 호출을 요청했을 때, 실제로 도구를 실행하는 노드
    """
    def __init__(self, tools: list) -> None:
        self.tool_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        outputs = []
        for tool_call in message.tool_calls:
            print(f"--- Tool 호출: {tool_call['name']} ---")
            tool_result = self.tool_by_name[tool_call["name"]].invoke(tool_call["args"])
            
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[search_tool])

# 6. 조건부 엣지(Conditional Edge) 함수 정의
def route_tools(state: State) -> Literal["tools", "__end__"]:
    """
    마지막 메시지에 tool_calls가 있으면 'tools' 노드로, 없으면 종료(__end__)로 라우팅
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No message found in input state: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"

# 7. 노드 추가 및 엣지 연결
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

# chatbot 노드 이후에 조건부 엣지 실행
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {
        "tools": "tools",
        "__end__": END
    }
)

# 도구 실행 후에는 다시 chatbot으로 돌아와서 결과를 자연어로 정리하도록 함
graph_builder.add_edge("tools", "chatbot")

# 8. 그래프 컴파일
graph = graph_builder.compile()

# 9. 실행 루프
print("=== LangGraph 에이전트 (검색 가능) ===")
print("예시: '서울 날씨 알려줘', '애플 주가 어때?'")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    for event in graph.stream({"messages": [("user", user_input)]}):
        for key, value in event.items():
            if key == "chatbot":
                msg = value["messages"][-1]
                if not msg.tool_calls: # 최종 응답일 때만 출력
                    print(f"Assistant: {msg.content}")
