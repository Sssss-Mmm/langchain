from dotenv import load_dotenv
import os
from langchain_openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# <'get_format_instructions' 메소드>

from langchain_core.output_parsers import JsonOutputParser

# JSON 출력 파서 불러오기
parser = JsonOutputParser()

instructions  = parser.get_format_instructions()
print(instructions)

# <'parse' 메서드>

ai_response = '{"이름": "김철수" , "나이": 30}'

parsed_response = parser.parse(ai_response)
print(parsed_response)

#<'parse_with_prompt' 메서드>

from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_openai import ChatOpenAI

# 파서 설정
parser = RetryWithErrorOutputParser.from_llm(parser=JsonOutputParser,llm=ChatOpenAI)

question = "가장 큰 대륙은?"
ai_response = "아시아입니다."

try:
    result = parser.parse_with_prompt(ai_response,question)
    print(result)
except Exception as e :
    print(f"오류 발생: {e}")

# <Pydantic 출력 파서 예시>

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator

model = ChatOpenAI(model="gpt-4o", temperature=0.0)

# 원하는 데이터 구조 정의
class FinancialAdvice(BaseModel):
    setup: str = Field(description="금융 조언 상황을 설정하기 위한 질문")
    advice: str = Field(description="질문을 해결하기 위한 금융 답변")
    # Pydantic을 사용한 사용자 정의 검증 로직
    @model_validator(mode="before")
    @classmethod
    def question_ends_with_question_mark(cls, values: dict) -> dict:
        setup = values.get("setup", "")
        if not setup.endswith("?"):
            raise ValueError("잘못된 질문 형식입니다! 질문은 '?'로 끝나야 합니다.")
        return values

# 파서 설정 및 프롬프트 템플릿에 지침 삽입
parser = PydanticOutputParser(pydantic_object=FinancialAdvice)
prompt = PromptTemplate(
    template="다음 금융 관련 질문에 답변해 주세요.\n{format_instructions}\n질문: {query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
# 언어 모델을 사용해 데이터 구조를 채우도록 프롬프트와 모델 설정
chain = prompt | model | parser


# 체인 실행 및 결과 출력
try:
    result = chain.invoke({"query": "부동산에 관련하여 금융 조언을 받을 수 있는 질문하여라."})
    print(result)
except Exception as e:
    print(f"오류 발생: {e}")


# <SimpleJsonOutputParser 출력 파서 예시>

from langchain.output_parsers.json import SimpleJsonOutputParser
# JSON 포맷의 응답을 생성하는 프롬프트 템플릿 설정
json_prompt = PromptTemplate.from_template(
    "다음 질문에 대한 답변이 포함된 JSON 객체를 반환하십시오: {question}"
)
json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser

#스트리밍 예시: 질문에 대한 답변이 부분적으로 구문 분석됨
# print(list(json_chain.stream({"question":"비트코인에 대한 짧은 한문장 설명."})))

# <JsonOutputParser 출력 파서 예시>

class FinancialAdvice(BaseModel):
    setup : str = Field(description="금융 조언 상황을 설정하기 위한 질문")
    advice : str = Field(description="질문을 해결하기 위한 금융 답변")

# JSON 출력 파서 설정 및 프롬프트 템플릿에 지침 삽입

parser = JsonOutputParser(pydantic_object=FinancialAdvice)
prompt = PromptTemplate(
    template="다음 금융 관련 질문에 답변해 주세요.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

chain = prompt | model | parser

print(chain.invoke({"query":"부동산에 관련하여 금융 조언을 받을 수 있는 질문하여라."}))