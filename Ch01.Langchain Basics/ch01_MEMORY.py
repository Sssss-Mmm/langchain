from dotenv import load_dotenv
import os
from langchain_openai import OpenAI

load_dotenv()

api_key =os.getenv("OPENAI_API_KEY")

llm = OpenAI()

# <이전 대화를 포함한 메시지 전달>

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini")


# 프롬프트 템플릿 정의 : 금융 상담 역할
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 금융 상담사입니다. 사용자에게 최선의 금융 조언을 제공합니다."),
        ("placeholder", "{messages}"),  # 대화 이력 추가
    ]
)


# 프롬프트와 모델을 연결하여 체인 생성
chain = prompt | chat

# 이전 대화를 포함한 메시지 전달
ai_msg = chain.invoke(
    {
        "messages": [
            ("human","저축을 늘리기 위해 무엇을 할 수 있나요?"), # 사용자의 첫 질문
            ("ai", "저축 목표를 설정하고, 매달 자동 이체로 일정 금액을 저축하세요."), # 챗봇의 답변
            ("human","방금 뭐라고 했나요?"), # 사용자의 재확인 질문
        ],
    }
)

print(ai_msg.content)

# <'ChatMessageHistory'를 사용한 메시지 관리>

from langchain_community.chat_message_histories import ChatMessageHistory

# 대화 이력 저장을 위한 클래스 초기화
chat_history = ChatMessageHistory()

# 사용자 메시지 추가
chat_history.add_user_message("저축을 늘리기 위해 무엇을 할 수 있나요?")
chat_history.add_ai_message("저축 목표를 설정하고, 매달 자동 이체로 일정 금액을 저축하세요.")

# 새로운 질문 추가 후 다시 체인 실행
chat_history.add_user_message("방금 뭐라고 했나요?")
ai_response = chain.invoke({"messages": chat_history.messages})
print(ai_response.content)

# < RunnableWithMessageHistory' 를 사용한 메시지 관리>
from langchain_core.runnables.history import RunnableWithMessageHistory

# 시스템 메시지와 대화 이력을 사용하는 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 금융 상담사입니다. 모든 질문에 최선을 다해 답변하십시오."),
        ("placeholder", "{chat_history}"), # 이전 대화 이력
        ("human","{input}"), # 사용자의 새로운 질문
    ]
)

# 대화 이력을 관리할 체인 설정
chat_history =ChatMessageHistory()
chain = prompt | chat

# RunnableWithMessageHistory 클래스를 사용해 체인을 감쌉니다.

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history, # 세선 ID에 따라 대화 이력을 불러오는 함수
    input_messages_key = "input", # 입력 메시지의 키 설정
    history_messages_key= "chat_history", # 대화 이력의 키 설정
)

print(chain_with_message_history.invoke(
    {"input": "저축을 늘리기 위해 무엇을 할 수 있나요?"},
    {"configurable": {"session_id":"unused"}}
).content)

print(chain_with_message_history.invoke(
    {"input": "내가 방금 뭐라고 했나요?"},  # 사용자의 질문
    {"configurable": {"session_id": "unused"}}  # 세션 ID 설정
).content)

# <메시지 트리밍 예제>

from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# 메시지 트리밍 유틸리티 설정

trimmer = trim_messages(strategy="last",max_tokens=2, token_counter=len)

# 트리밍된 대화 이력과 함께 체인 실행

chain_with_trimming = (
    RunnablePassthrough.assign(chat_history=itemgetter("chat_history")| trimmer)
    | prompt
    | chat
)

# 트리밍된 대화 이력을 사용하는 체인 설정

chain_with_trimmed_history = RunnableWithMessageHistory(
    chain_with_trimming,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

print(chain_with_trimmed_history.invoke(
    {"input": "저는 5년 내에 집을 사기 위해 어떤 재정 계획을 세워야 하나요?"}, # 사용자의 질문
    {"configurable": {"session_id": "finance_session_1"}} # 세션 ID 설정
))

print(chain_with_trimmed_history.invoke(
    {"input": "내가 방금 뭐라고 그랬나요?"},
    {"configurable": {"session_id":"finance_session_1"}}
).content)

# <이전 대화 요약 내용 기반으로 답변하기>

def summarize_messages(chain_input):
    store_messages = chat_history.messages
    if len(store_messages) == 0 :
        return False
    # 대화를 요약하기 위한 프롬프트 템플릿 설정
    summarize_prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"), # 이전 대화 이력
            (
                "user",
                "이전 대화를 요약해 주세요. 가능한 한 많은 세부 정보를 포함하십시오", # 요약 요청 메시지
            )
        ]
    )

    # 요약 체인 생성 및 실행
    summarization_chain = summarize_prompt | chat
    summary_message = summarization_chain.invoke({"chain_history": store_messages})
    chat_history.clear() # 요약 후 이전 대화 삭제
    chat_history.add_message(summary_message) # 요약된 메시지를 대화 이력에 추가
    return True

# 대화 요약을 처리하는 체인 설정
chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized = summarize_messages)
    | chain_with_message_history
)

print(chain_with_summarization.invoke(
    {"input": "저에게 어떤 재정적 조언을 해주셨나요?"}, # 사용자의 질문
    {"configurable":{"session_id": "unused"}} # 세션 ID 설정
).content)