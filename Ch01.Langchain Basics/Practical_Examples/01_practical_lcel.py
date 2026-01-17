from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def advanced_lcel_chain():
    print("=== Advanced LCEL Chain (Translation + Critique) ===")

    # 1. 컴포넌트 설정
    model = ChatOpenAI(model="gpt-4o")
    parser = StrOutputParser()

    # 2. 첫 번째 체인: 번역
    translate_prompt = ChatPromptTemplate.from_template(
        "다음 텍스트를 {target_language}로 번역해주세요:\n\n{text}"
    )
    translate_chain = translate_prompt | model | parser

    # 3. 두 번째 체인: 감수 (Critique)
    critique_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 전문 번역가입니다. 원문과 번역문을 비교하여 평가하고, 더 자연스러운 표현이 있다면 추천해주세요."),
        ("human", """
        [원문]
        {original}
        
        [초벌 번역]
        {translation}
        
        위 내용을 평가하고 개선안을 제시해주세요.
        """)
    ])
    critique_chain = critique_prompt | model | parser

    # 4. 전체 파이프라인 구성 (RunnablePassthrough 활용)
    # 입력 데이터를 유지하면서 중간 결과를 전달하기 위해 assign 사용
    
    # 1단계: 번역 실행 및 결과 저장
    # {"text": ..., "target_language": ...} -> {"original": ..., "translation": ...}
    
    overall_chain = (
        RunnablePassthrough.assign(
            translation=translate_chain
        )
        # 2단계: 키 매핑 (critique_prompt가 기대하는 변수명으로 맞춤)
        | RunnablePassthrough.assign(
            original=itemgetter("text")
        )
        # 3단계: 감수 체인 실행
        | critique_chain
    )

    # 5. 실행
    input_data = {
        "text": "The early bird catches the worm, but the second mouse gets the cheese.",
        "target_language": "한국어"
    }
    
    print(f"\n[Input]: {input_data['text']}")
    print("Running Chain...")
    
    result = overall_chain.invoke(input_data)
    
    print(f"\n[Final Critique Result]:\n{result}")

if __name__ == "__main__":
    advanced_lcel_chain()
