from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Pydantic 모델 정의 (중첩 구조)
class ProductFeature(BaseModel):
    name: str = Field(description="기능의 이름 (예: 카메라, 배터리)")
    sentiment: str = Field(description="해당 기능에 대한 감성 (Positive, Negative, Neutral)")
    details: str = Field(description="관련된 세부 내용 요약")

class ReviewAnalysis(BaseModel):
    summary: str = Field(description="리뷰 전체 요약")
    rating_prediction: int = Field(description="리뷰 내용을 바탕으로 예측한 별점 (1~5)")
    features: List[ProductFeature] = Field(description="언급된 주요 기능별 분석 리스트")
    is_spam: bool = Field(description="스팸이나 광고성 리뷰인지 여부")

def structured_review_analyzer():
    print("=== Structured Output Parser (Review Analysis) ===")

    # 2. 파서 및 모델 설정
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

    # 3. 프롬프트 정의
    # format_instructions가 가장 중요함.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 e-커머스 리뷰 분석 AI입니다. 사용자의 리뷰를 분석하여 지정된 형식의 JSON으로 반환해주세요.\n\n{format_instructions}"),
        ("human", "{review_text}")
    ])

    # 4. 체인 연결
    chain = prompt | model | parser

    # 5. 실행
    review_sample = """
    이 노트북 배터리가 너무 빨리 닳아요. 3시간도 못 쓰는 것 같습니다. 
    무게는 가벼워서 들고 다니기는 좋은데 충전기를 항상 챙겨야 하니 의미가 없네요.
    화면 화질은 쨍하고 좋습니다. 디자인도 깔끔하구요.
    근데 가격 생각하면 추천하기 어렵네요.
    """
    
    print(f"\n[Original Review]:\n{review_sample.strip()}")
    print("\n[Analyzing...]")

    try:
        result = chain.invoke({
            "review_text": review_sample,
            "format_instructions": parser.get_format_instructions()
        })
        
        # 6. 결과 활용 (Python 객체로 다룸)
        print(f"\n- 요약: {result.summary}")
        print(f"- 예측 별점: {result.rating_prediction}점")
        print(f"- 스팸 여부: {result.is_spam}")
        
        print("\n[기능별 분석]")
        for feature in result.features:
            print(f"* {feature.name}: {feature.sentiment} ({feature.details})")

    except Exception as e:
        print(f"Error parsing output: {e}")

if __name__ == "__main__":
    structured_review_analyzer()
