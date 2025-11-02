import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# 루프 구현하기
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages : Annotated[list,add_messages]

graph_builder =StateGraph(State)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# 오픈AI를 호출하여 응답을 받아 온 뒤, 상태값에 저장하여 반환하는 챗봇 함수 정의
def chatbot(state:State) :
    return{"messages":[llm.invoke(state["messages"])]}

# 챗봇 노드 정의
graph_builder.add_node("chatbot",chatbot)


graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()


while True:
    # 사용자의 질의 입력 받음
    user_input = input("User: ")
    print("User: ",user_input)

    # 사용자가 quit 혹은 exit 혹은 q를 입력했다면 루프 종료
    if user_input.lower() in ["quit","exit","q"]:
       print("Goodbye!")
       break
    
    # 사용자의 입력을 그래프에 입력하여 정의된 흐름 실행
    for event in graph.stream({"messages":("user",user_input)}):
        for value in event.values():
            print("Assistant:",value["messages"][-1].content)



