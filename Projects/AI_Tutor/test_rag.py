from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv() # 현재 디렉토리
load_dotenv("/home/sssss_mmm/langchain/Projects/AI_Tutor/.env")

from state import TutorState
from nodes import generate_question_node

def test_rag_generation():
    print("Testing 'multiple_choice' question generation (RAG)...")
    
    # DB 존재 여부 확인
    if not os.path.exists("./vector_store"):
        print("Error: ./vector_store not found. Please run ingest.py first.")
        return

    state: TutorState = {
        "topic": "데이터베이스 설계", # PDF에 있을만한 주제
        "difficulty": "Medium",
        "question_type": "multiple_choice",
        "messages": [],
        "attempt_count": 0,
        "question_solved": False,
        "last_evaluation": None
    }
    
    try:
        res = generate_question_node(state)
        print(f"\n[Generated Question]\n{res['current_question']}")
        print(f"\n[Correct Answer] {res.get('correct_answer')}")
        print(f"[Explanation] {res.get('explanation')}")
        
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    test_rag_generation()
