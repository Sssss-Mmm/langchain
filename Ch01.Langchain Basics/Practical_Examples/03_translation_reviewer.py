from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 1. 모델 설정
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# 2. 번역 체인 (1단계)
# 텍스트를 대상 언어로 번역합니다.
translation_prompt = ChatPromptTemplate.from_template(
    "다음 텍스트를 {language}로 번역해주세요:\n\n{text}"
)
translation_chain = translation_prompt | model | parser

# 3. 검토 체인 (2단계)
# 번역된 결과와 원문을 비교하여 품질을 평가하고 개선안을 제안합니다.
review_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 전문 번역 감수자입니다. 원문과 번역문을 비교하여 번역의 정확성, 자연스러움을 평가하고 개선된 번역을 제안해주세요."),
    ("human", """
    [원문]
    {original_text}
    
    [번역문]
    {translated_text}
    
    위 내용을 바탕으로 다음 항목을 작성해주세요:
    1. 번역 평가 (비평)
    2. 개선된 번역문
    """)
])
review_chain = review_prompt | model | parser

# 4. 통합 체인 구성 (RunnableLambda 사용)
# 번역 체인의 출력을 검토 체인의 입력으로 연결합니다.

def prepare_review_input(input_data):
    # 입력 데이터에서 원문과 대상 언어를 추출
    text = input_data["text"]
    language = input_data["language"]
    
    # 1단계: 번역 실행
    # invoke를 사용하여 동기적으로 실행시킵니다.
    translated_text = translation_chain.invoke({"text": text, "language": language})
    
    print(f"--- 1단계 번역 결과 ---\n{translated_text}\n-----------------------")
    
    # 2단계 입력 구성
    return {
        "original_text": text,
        "translated_text": translated_text
    }

# RunnableLambda를 사용하여 함수를 체인의 일부로 만듭니다.
# review_chain 앞에 전처리 단계를 연결합니다.
full_chain = RunnableLambda(prepare_review_input) | review_chain

# 5. 실행 예시
print("=== 번역 및 감수기 ===")

text_to_translate = "The quick brown fox jumps over the lazy dog."
target_language = "한국어"

print(f"원문: {text_to_translate}")
print(f"대상 언어: {target_language}")
print("\n[실행 시작]")

result = full_chain.invoke({
    "text": text_to_translate,
    "language": target_language
})

print("\n[최종 감수 결과]")
print(result)
