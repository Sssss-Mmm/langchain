from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로드

# 환경 변수에서 API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
# 오픈AI 대규모 언어 모델 초기화
llm = ChatOpenAI(model="gpt-4o",api_key=api_key)

result = llm.invoke("안녕")
print(result)

from openai import OpenAI
from typing import List

# 기본 오픈AI 클라이언트 사용
client = OpenAI(api_key=api_key)

# # "안녕" 메세지를 보내고 응답을 받음
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "안녕"}
    ]
)
print (response.choices[0].message.content)

# 요청에 사용할 프롬프트 템플릿 정의
prompt_template = "주제 {topic}에 대해 짧은 설명을 해줘"

# 메시지를 보내고 모델의 응답을 받는 함수
def call_chat_model(messages: List[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

# 주어진 주제에 따라 설명을 요청하는 함수
def invoke_chain(topic: str) -> str:
    prompt_value = prompt_template.format(topic=topic)
    messages = [{"role": "user", "content": prompt_value}]
    return call_chat_model(messages)
# "더블딥"에 대한 설명 요청
print(invoke_chain("더블딥"))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# 주어진 주제에 대한 짧은 설명을 요청하는 프롬프트 템플릿 정의
prompt_template = ChatPromptTemplate.from_template("주제 {topic}에 대해 짧은 설명을 해줘")

# 출력 파서를 문자열로 설정
output_parser = StrOutputParser()
# 오픈AI의 gpt-4o 모델 초기화
model = ChatOpenAI(model="gpt-4o", api_key=api_key)

# 파이프라인 설정: 주제를 받아 프롬프트를 생성하고, 모델을 통해 응답을 생성한 후 문자열로 파싱
chain = (
    {"topic": RunnablePassthrough()}
    | prompt_template
    | model
    | output_parser
)
# "더블딥"에 대한 설명 요청 및 출력
print(chain.invoke("더블딥"))

