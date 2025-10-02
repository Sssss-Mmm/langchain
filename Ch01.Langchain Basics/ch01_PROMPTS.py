from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

#<문자열 프롬프트 템플릿>
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# 오픈AI 대규모 언어 모델 초기화
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

prompt_template = PromptTemplate.from_template("주제 {topic}에 대해 금융 관련 짧은 조언을 해줘")
# '투자' 주제로 프롬프트 템플릿 호출
# print(prompt_template.invoke({"topic": "투자"}))

# <챗 프롬프트 템플릿>
from langchain_core.prompts import ChatPromptTemplate

# 챗 프롬프트 템플릿 정의: 사용자와 시스템 간의 메세지를 포함
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 유능한 금융 조언가입니다."),
    ("user", "주제 {topic}에 대해 금융 관련 짧은 조언을 해주세요")
])
# '주식' 주제로 챗 프롬프트 템플릿 호출
# print(prompt_template.invoke({"topic":"주식"}))

# <메시지 자리 표시자 템플릿>

# 라이브러리 불러오기
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotPromptTemplate
from langchain_core.messages import HumanMessage

# (방법1) 메시지 자리 표시자를 포함한 챗 프롬프트 템플릿 정의
prompt_template = ChatPromptTemplate.from_messages([
    ("system","당신은 유능한 금융 조언가입니다."),
    MessagesPlaceholder("msgs")
])

# 메시지 리스트를 'msgs' 자리 표시자에 전달하여 호출
# print(prompt_template.invoke({"msgs":[HumanMessage(content="안녕하세요!")]}))

# (방법2) MessagePlaceholder 클래스를 명시적으로 사용하지 않고도 비슷한 작업을 수행할 수 있는 방법
prompt_template = ChatPromptTemplate.from_messages([
    ("system","당신은 유능한 금융 조언가입니다."),
    ("placeholder", "{msgs}") # <- 여기서 'msgs'가 자리 표시자로 사용됩니다.
])

# 메시지 리스트를 'msgs' 자리 표시자에 전달하여 호출
# print(prompt_template.invoke({"msgs":[HumanMessage(content="안녕하세요!")]}))

# <'PromptTemplate'를 이요한 퓨샷 프롬프트>

#질문과 답변을 포맷하는 프롬프트 템플릿 정의
example_prompt = PromptTemplate.from_template("질문: {question}\n 답변: {answer}")

examples = [
    {
        "question":"주식 투자와 예금 중 어느 것이 더 수익률이 높은가?",
        "answer": """
후속 질문이 필요한가요: 네.
후속 질문: 주식 투자의 평균 수익률은 얼마인가요?
중간 답변: 주식 투자의 평균 수익률은 연 7%입니다.
후속 질문: 예금의 평균 이자율은 얼마인가요?
중간 답변: 예금의 평균 이자율은 연 1%입니다.
따라서 최종 답변은: 주식 투자
"""
    },
    {
        "question": "부동산과 채권 중 어느 것이 더 안정적인 투자처인가?",
        "answer": """
후속 질문이 필요한가요: 네.
후속 질문: 부동산 투자의 위험도는 어느 정도인가요?
중간 답변: 부동산 투자의 위험도는 중간 수준입니다.
후속 질문: 채권의 위험도는 어느 정도인가요?
중간 답변: 채권의 위험도는 낮은 편입니다.
따라서 최종 답변은: 채권
""",
    }
]

print(example_prompt.invoke(examples[0]).to_string())

# <'FewShotPromptTemplate' 를 이용한 퓨샷 프롬프트>

# FewShotPromptTemplate 생성
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="질문: {input}",
    input_variables=["input"],
)

print(prompt.invoke({"input":"부동산 투자의 장점은 무엇인가?"}).to_string())

# <예제 선택기를 이용한 퓨샷 프롬프트>

#라이브러리 불러오기
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# 예제 선택기 초기화
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, # 사용할 예제 목록
    OpenAIEmbeddings(api_key=api_key), # 임베딩 생성에 사용되는 클래스
    Chroma, # 임베딩을 저장하고 유사도 검색을 수행하는 벡터 스토어 클래스
    k=1, # 선택할 예제의 수
)

# 입력과 가장 유사한 예제 선택
question = "부동산 투자의 장점은 무엇인가?"
selected_examples = example_selector.select_examples({"question":question})

# 선택된 예제 출력
print(f"입력과 가장 유사한 예제: {question}")
for example in selected_examples:
    print("\n")
    for k,v in example.items():
        print(f"{k}:{v}")

# <퓨샷 프롬프트 AI 모델 적용>

# 퓨샷 프롬프트 템플릿 설정
example_prompt = PromptTemplate(
    input_variables=["question","answer"],
    template="질문: {question}\n 답변: {answer}"
)

# 퓨샷 프롬프트 템플릿 설정
prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = example_prompt,
    prefix="다음은 금융 관련 질문과 답변의 예시입니다. 어르신이 이해할 수 있게 답변해주세요",
    suffix= "질문: {input}\n답변:",
    input_variables = ["input"]
)

model = ChatOpenAI(model_name="gpt-4o")

chain = prompt | model

response = chain.invoke({"input":"부동산 투자의 장점은 무엇인가?"}) # invoke 메소드 사용

print(response.content)