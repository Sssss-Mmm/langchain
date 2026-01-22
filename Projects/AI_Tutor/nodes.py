from state import TutorState
from chains import generate_question_chain, evaluate_answer_chain, provide_hint_chain
from langchain_core.messages import AIMessage

def generate_question_node(state: TutorState) -> dict:
    """
    주제와 난이도에 맞는 새로운 문제를 생성하여 상태에 업데이트합니다.
    """
    topic = state.get("topic", "Python Basics")
    difficulty = state.get("difficulty", "Easy")
    
    print(f"\n[System] Generating new question... (Topic: {topic}, Level: {difficulty})")
    
    question = generate_question_chain.invoke({"topic": topic, "difficulty": difficulty})
    
    # 문제 출제 시 초기화해야 할 상태들
    return {
        "current_question": question,
        "question_solved": False,
        "attempt_count": 0,
        "last_evaluation": None,
        "messages": [AIMessage(content=f"[문제] {question}")] # 사용자에게 문제 보여줌
    }

def evaluate_answer_node(state: TutorState) -> dict:
    """
    사용자의 답변을 평가하고 성공 여부를 업데이트합니다.
    """
    current_question = state["current_question"]
    # 가장 최근 메시지는 사용자의 답변(HumanMessage)이라고 가정
    user_answer = state["messages"][-1].content
    
    print("\n[System] Evaluating answer...")
    
    eval_result = evaluate_answer_chain.invoke({
        "current_question": current_question,
        "user_answer": user_answer
    })
    
    # 평가 결과 파싱 ([PASS] / [FAIL])
    is_pass = "[PASS]" in eval_result
    
    # 상태 업데이트
    return {
        "question_solved": is_pass,
        "last_evaluation": "Pass" if is_pass else "Fail",
        "messages": [AIMessage(content=eval_result)]
    }

def provide_hint_node(state: TutorState) -> dict:
    """
    사용자가 틀렸을 때 힌트를 제공하고 시도 횟수를 증가시킵니다.
    """
    current_question = state["current_question"]
    attempt_count = state["attempt_count"]
    
    print(f"\n[System] Providing hint... (Attempt: {attempt_count + 1})")
    
    hint = provide_hint_chain.invoke({
        "current_question": current_question,
        "attempt_count": attempt_count
    })
    
    return {
        "attempt_count": attempt_count + 1,
        "messages": [AIMessage(content=f"[힌트] {hint}")]
    }
