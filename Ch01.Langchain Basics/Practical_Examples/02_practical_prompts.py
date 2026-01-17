from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate, 
    FewShotChatMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

def few_shot_email_drafter():
    print("=== Few-Shot Email Drafter (Style Transfer) ===")

    # 1. 예시 데이터 (Few-Shot Examples)
    # AI에게 '원하는 스타일'을 학습시키기 위한 예시들입니다.
    # 여기서는 간결하고 명확한(Terse/Clear) 스타일을 가르칩니다.
    examples = [
        {
            "input": "회의 늦을 것 같다고 팀장님께 연락해줘.",
            "output": "팀장님, 금일 회의에 약 10분 정도 늦을 예정입니다. 불편을 드려 죄송합니다. 최대한 빨리 참석하겠습니다."
        },
        {
            "input": "거래처에 견적서 보냈다고 알려줘.",
            "output": "안녕하세요. 요청하신 견적서를 첨부하여 메일로 발송해 드렸습니다. 확인 부탁드립니다. 감사합니다."
        }
    ]

    # 2. 예시 포맷 정의 (Example Prompt)
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    # 3. Few-Shot 프롬프트 템플릿 생성
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # 4. 전체 프롬프트 구성
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 비즈니스 이메일 작성 비서입니다. 제공된 예시의 어조와 형식을 참고하여 이메일을 작성하세요."),
            few_shot_prompt, # 여기에 예시들이 삽입됩니다.
            ("human", "{input}"),
        ]
    )

    # 5. 체인 구성
    model = ChatOpenAI(model="gpt-4o", temperature=0) # 스타일 일관성을 위해 temperature를 낮춤
    chain = final_prompt | model | StrOutputParser()

    # 6. 실행
    user_request = "다음 주 휴가라고 팀원들에게 공지해줘."
    print(f"\n[User Request]: {user_request}")
    
    print("\n[Generated Email]:")
    result = chain.invoke({"input": user_request})
    print(result)

if __name__ == "__main__":
    few_shot_email_drafter()
