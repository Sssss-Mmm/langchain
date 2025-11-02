import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# 조건문 구현하기
from langchain_community.tools.tavily_search import TavilySearchResults

# Tavily검색 엔진을 도구로 정의
tool = TavilySearchResults(max_results=2)
tools =[tool]
print(tool.invoke("내일 대한민국 서울의 날씨는?"))

from typing import Annotated

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages


# 그래프 상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 그래프 정의
graph_builder = StateGraph(State)

# 오픈AI 클라이언트 정의
llm = ChatOpenAI(model="gpt-4o-mini")
# 오픈AI 클라이언트에 Tavily 검색 엔진 도구를 할당
llm_with_tools = llm.bind_tools(tools)

# 챗봇 함수 정의
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 그래프에 챗봇 노드 추가
graph_builder.add_node("chatbot", chatbot)

import json

from langchain_core.messages import ToolMessage

# 도구 노드로 사용될 클래스
class BasicTooleNode:

    # 도구 노드에서 사용될 초기 파라미터 정의
    def __init__(self,tools: list)->None:
        self.tool_by_name = {tool.name: tool for tool in tools}
    
    # 도구 노드가 호출되었을때의 행동 정의
    def __call__(self, inputs:dict):
        if messages := inputs.get("messages",[]):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        # 메세지의 tool_calls 에 도구호출을 위한 값들이 존재한다면 이를 활용해 도구 호출
        outputs = []
        for tool_call in messages.tool_calls:
            tool_result = self.tool_by_name[tool_call["name"]].invoke(tool_call["args"])
            
            # 도구 호출의 결과물을 ToolMessages로 정의하여 출력값에 저장
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result,ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id = tool_call["id"]
                )
            )
        return {"messages":outputs}

tool_node = BasicTooleNode(tools=[tool])
graph_builder.add_node("tools",tool_node)

from typing import Literal

# 도구노드 호출 여부를 결정하는 함수 정의
def route_tools(state:State)-> Literal["tools","__end__"]:
    # 상태값의 가장 최근 메세지를 정의
    if isinstance(state,list):
        ai_message = state[-1]
    elif messages := state.get("messages",[]):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No message found in input state to tool_edge: {state}")
    
    # 가장 최근 메시지가 tool_calls 속성을 포함하고 있다면 tools 노드를 아니라면 종료지점을 반환
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"

# 도구 노드와 챗봇 노드 연결
graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge(START,"chatbot")
graph = graph_builder.compile()

# 스트리밍
from langchain_core.messages import BaseMessage

while True:
    # 사용자의 질문을 입력받습니다
    user_input = input("User: ")
    print("User:", user_input)
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # 업데이트된 내용을 확인할 수 있는 그래프 스트리밍을 정의합니다.
    events = graph.stream(input={"messages": [("user", user_input)]}, stream_mode="updates")

    # 그래프 이벤트 내의 메세지를 출력합니다.
    for event in events:
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                print("Assistant:", value["messages"][-1].content)
