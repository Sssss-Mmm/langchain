from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv

load_dotenv()

def advanced_memory_chatbot():
    print("=== Chatbot with Summary Buffer Memory ===")
    
    # 1. 모델 설정
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # 2. 메모리 설정 (ConversationSummaryBufferMemory)
    # 대화 내용이 max_token_limit을 넘어가면, 오래된 대화를 요약(Summary)하여 저장합니다.
    memory = ConversationSummaryBufferMemory(
        llm=model, # 요약을 수행할 LLM
        max_token_limit=150, # 매우 짧게 설정하여 요약이 빨리 일어나도록 테스트
        return_messages=True
    )

    # 3. 체인 생성 (ConversationChain)
    # 메모리 관리를 자동으로 수행하는 기본 체인입니다.
    conversation = ConversationChain(
        llm=model,
        memory=memory,
        verbose=True # 내부 동작(요약 과정 등)을 보기 위해 True 설정
    )

    # 4. 대화 시뮬레이션
    print("\n[Conversation Start]")
    
    # Turn 1
    user_input_1 = "안녕하세요. 저는 AI 개발 공부를 하고 있는 김철수라고 합니다."
    print(f"\nUser: {user_input_1}")
    response_1 = conversation.predict(input=user_input_1)
    print(f"AI: {response_1}")

    # Turn 2
    user_input_2 = "제가 요즘 관심 있는 분야는 RAG랑 에이전트 쪽이에요. 공부할 만한 자료가 있을까요?"
    print(f"\nUser: {user_input_2}")
    response_2 = conversation.predict(input=user_input_2)
    print(f"AI: {response_2}")

    # Turn 3 (여기서 토큰 제한을 넘기면 요약이 발생할 것임)
    user_input_3 = "그리고 LangChain을 이용해서 실무 프로젝트도 하나 해보고 싶습니다. 어떤 주제가 좋을까요? 추천 좀 해주세요. 아주 구체적으로요."
    print(f"\nUser: {user_input_3}")
    response_3 = conversation.predict(input=user_input_3)
    print(f"AI: {response_3}")

    # 5. 메모리 상태 확인
    print("\n[Current Memory State]")
    # 메모리 버퍼(최근 대화)
    print(f"- Buffer Length: {len(memory.chat_memory.messages)}")
    # 이동된 요약 내용
    print(f"- Moving Summary: {memory.moving_summary_buffer}")
    
    # Turn 4 (요약된 내용을 바탕으로 대화가 이어지는지 확인)
    user_input_4 = "제 이름 기억하시나요?"
    print(f"\nUser: {user_input_4}")
    response_4 = conversation.predict(input=user_input_4)
    print(f"AI: {response_4}")

if __name__ == "__main__":
    advanced_memory_chatbot()
