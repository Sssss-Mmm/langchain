from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import os
import random
from dotenv import load_dotenv

load_dotenv()

# 1. 데이터 구조 정의
class QA(BaseModel):
    question: str = Field(description="생성된 질문")
    answer: str = Field(description="해당 질문에 대한 정답")
    difficulty: str = Field(description="난이도 (상, 중, 하)")

class QASet(BaseModel):
    qa_list: List[QA] = Field(description="질문-답변 쌍의 리스트")

def generate_qa_from_pdf():
    print("=== 부동산 보고서 기반 문제 생성기 ===")
    
    # 2. PDF 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "..", "Data", "2024_KB_부동산_보고서_최종.pdf")

    if not os.path.exists(pdf_path):
        print(f"오류: 파일을 찾을 수 없습니다: {pdf_path}")
        return

    print("Loading PDF...")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception as e:
        print(f"PDF 로드 오류: {e}")
        return

    # 3. 텍스트 추출 (랜덤 페이지 선택)
    # 전체 문서를 다 넣으면 토큰 비용이 많이 들 수 있으므로, 랜덤하게 페이지를 하나 골라 문제를 출제합니다.
    # 실무에서는 특정 챕터를 지정하거나 슬라이딩 윈도우 방식을 사용할 수 있습니다.
    if not pages:
        print("PDF에 내용이 없습니다.")
        return

    selected_page = random.choice(pages)
    text_content = selected_page.page_content
    page_num = selected_page.metadata.get('page', 0) + 1
    
    print(f"Selected Page: {page_num}")
    print(f"Content Length: {len(text_content)} characters")

    if len(text_content) < 50:
        print("텍스트 내용이 너무 적어 문제를 생성할 수 없습니다. 다시 실행해주세요.")
        return

    # 4. 모델 및 체인 설정
    model = ChatOpenAI(model="gpt-4o", temperature=0.7)
    parser = PydanticOutputParser(pydantic_object=QASet)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 부동산 전문가이자 교육 자료 개발자입니다. 주어진 텍스트 내용을 바탕으로 핵심 내용을 묻는 퀴즈를 {num_questions}개 생성해주세요.\n\n{format_instructions}"),
        ("human", "다음 텍스트(부동산 보고서 일부)를 기반으로 문제를 출제해주세요:\n\n{text}")
    ])

    chain = prompt | model | parser

    # 5. 실행
    print("Generating Q&A pairs...")
    try:
        result = chain.invoke({
            "text": text_content,
            "num_questions": 3,
            "format_instructions": parser.get_format_instructions()
        })
        
        print(f"\n[생성된 문항 세트] (출처: {page_num}페이지)\n")
        for i, qa in enumerate(result.qa_list):
            print(f"Q{i+1}. {qa.question}")
            print(f"A{i+1}. {qa.answer}")
            print(f"   (난이도: {qa.difficulty})")
            print("-" * 30)

    except Exception as e:
        print(f"Error generating QA: {e}")

if __name__ == "__main__":
    generate_qa_from_pdf()
