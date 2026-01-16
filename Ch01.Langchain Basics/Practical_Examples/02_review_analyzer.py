from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 1. 모델 설정
model = ChatOpenAI(model="gpt-4o")

# 2. 출력 데이터 구조 정의 (Pydantic 모델)
# AI가 반환해야 할 데이터의 형식을 명확하게 정의합니다.
class ReviewAnalysis(BaseModel):
    sentiment: str = Field(description="리뷰의 전반적인 감성 (긍정, 부정, 중립)")
    summary: str = Field(description="리뷰의 내용을 한 문장으로 요약")
    pros: List[str] = Field(description="사용자가 언급한 장점 목록 (최대 3개)")
    cons: List[str] = Field(description="사용자가 언급한 단점 목록 (최대 3개)")
    language: str = Field(description="리뷰가 작성된 언어 (예: 한국어, 영어)")

# 3. 출력 파서 설정
# Pydantic 모델을 기반으로 파서를 생성합니다.
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

# 4. 프롬프트 템플릿 정의
# 파서에서 제공하는 포맷 지시사항(get_format_instructions)을 프롬프트에 포함시켜야 합니다.
review_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 e-커머스 리뷰 분석 전문가입니다. 다음 리뷰를 분석하여 요청된 형식으로 정보를 추출해주세요.\n\n{format_instructions}"),
    ("human", "{review_text}")
])

# 5. 체인 구성
# 프롬프트 -> 모델 -> 파서
review_chain = review_prompt | model | parser

# 6. 실행 예시
print("=== 리뷰 분석기 ===")

# 예제 리뷰 1: 무선 이어폰
review_text_1 = """
이 이어폰 정말 물건이네요! 음질도 기대 이상이고 노이즈 캔슬링 기능이 정말 뛰어납니다. 
출퇴근 시간에 지하철에서 쓰는데 소음이 거의 안 들려요. 
다만 배터리 지속 시간이 조금 짧은 것 같아서 아쉽고, 케이스가 좀 미끄러워서 떨어뜨릴 뻔 했네요. 
그래도 이 가격에 이 정도 성능이면 대만족입니다.
"""

print(f"\n[리뷰 원문]\n{review_text_1.strip()}")
print("\n[분석 결과]")
try:
    result_1 = review_chain.invoke({
        "review_text": review_text_1,
        "format_instructions": parser.get_format_instructions()
    })
    # 결과는 ReviewAnalysis 객체로 반환됩니다.
    print(f"- 감성: {result_1.sentiment}")
    print(f"- 요약: {result_1.summary}")
    print(f"- 장점: {', '.join(result_1.pros)}")
    print(f"- 단점: {', '.join(result_1.cons)}")
    print(f"- 언어: {result_1.language}")
    
except Exception as e:
    print(f"오류 발생: {e}")

# 예제 리뷰 2: 스마트폰 케이스 (영어)
print("\n" + "-"*50)
review_text_2 = """
The case looks beautiful and fits perfectly. The material feels fastastic in hand.
However, it collects dust and lint way too easily. I have to clean it multiple times a day.
Also, the buttons are a bit hard to press.
"""

print(f"\n[리뷰 원문]\n{review_text_2.strip()}")
print("\n[분석 결과]")
try:
    result_2 = review_chain.invoke({
        "review_text": review_text_2,
        "format_instructions": parser.get_format_instructions()
    })
    print(f"- 감성: {result_2.sentiment}")
    print(f"- 요약: {result_2.summary}")
    print(f"- 장점: {', '.join(result_2.pros)}")
    print(f"- 단점: {', '.join(result_2.cons)}")
    print(f"- 언어: {result_2.language}")

except Exception as e:
    print(f"오류 발생: {e}")
