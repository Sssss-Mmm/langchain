from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# 환경 변수 로드 (API 키 등)
load_dotenv()

# 1. 모델 설정
# gpt-4o 모델을 사용하여 고품질의 텍스트 생성을 보장합니다.
model = ChatOpenAI(model="gpt-4o")

# 2. 프롬프트 템플릿 정의
# 시스템 메시지로 AI의 페르소나(비즈니스 이메일 작성자)를 지정합니다.
# 사용자 메시지로는 이메일 작성에 필요한 핵심 정보(수신자, 핵심 내용, 어조)를 받습니다.
email_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 전문적인 비즈니스 이메일 작성 도우미입니다. 사용자의 요청사항을 바탕으로 명확하고 정중한 이메일 초안을 작성해주세요."),
    ("human", """
    수신자: {recipient}
    핵심 내용: {key_points}
    어조: {tone}
    
    위 내용을 바탕으로 이메일 제목과 본문을 작성해주세요.
    """)
])

# 3. 출력 파서 설정
# 모델의 출력을 단순 문자열로 변환합니다.
output_parser = StrOutputParser()

# 4. 체인 구성 (LCEL)
# 프롬프트 -> 모델 -> 출력 파서 순으로 연결합니다.
email_chain = email_prompt | model | output_parser

# 5. 실행 예시
print("=== 이메일 초안 작성기 ===")

# 예제 1: 프로젝트 지연 공지
print("\n[Case 1: 프로젝트 일정 지연]")
inputs_1 = {
    "recipient": "김철수 팀장님",
    "key_points": "- 데이터베이스 마이그레이션 중 예상치 못한 오류 발생\n- 문제 해결을 위해 2일 정도 추가 시간 필요\n- 다음 주 월요일까지 최종 보고서 제출 예정",
    "tone": "죄송하지만 해결책을 제시하는 전문적인 어조"
}
response_1 = email_chain.invoke(inputs_1)
print(response_1)

# 예제 2: 회식 제안
print("\n[Case 2: 팀 회식 제안]")
inputs_2 = {
    "recipient": "개발팀 전원",
    "key_points": "- 이번 프로젝트 성공적 런칭 축하\n- 이번 주 금요일 저녁 6시\n- 장소는 회사 근처 '맛있는 고기집'\n- 참석 여부 목요일까지 회신 요망",
    "tone": "활기차고 격려하는 어조"
}
response_2 = email_chain.invoke(inputs_2)
print(response_2)
