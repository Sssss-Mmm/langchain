from state import TutorState
from chains import generate_question_chain, evaluate_answer_chain, provide_hint_chain, rag_question_chain, retriever, parser, get_random_documents
from langchain_core.messages import AIMessage

def generate_question_node(state: TutorState) -> dict:
    """
    주제와 난이도에 맞는 새로운 문제를 생성하여 상태에 업데이트합니다.
    """
    # 사용자가 주제/난이도를 선택하지 않으므로 랜덤 모드로 동작
    # 기본값 설정
    topic = state.get("topic", "정보처리기사 필기") 
    difficulty = state.get("difficulty", "Random")
    question_type = state.get("question_type", "multiple_choice")
    
    print(f"\n[System] Generating new question... (Topic/Subject: {topic}, Level: {difficulty}, Type: {question_type})")
    
    if question_type == "multiple_choice":
        # RAG 검색 (과목 필터링 포함)
        # state['topic']에 '1.소프트웨어 설계' 등이 들어있으면 필터링됨
        selected_docs = get_random_documents(k=3, subject=topic)
        
        if not selected_docs:
            # 문서가 없는 경우 fallback
            context = "정보처리기사 필기 시험 관련 일반 지식"
        else:
            # 문맥 병합
            context = "\n\n".join([doc.page_content for doc in selected_docs])
        
        # 문제 생성 체인 호출
        result = rag_question_chain.invoke({
            "context": context,
            "topic": topic,
            "format_instructions": parser.get_format_instructions()
        })
        
        # result is a dict because of JsonOutputParser
        question_text = f"{result['question']}\n"
        for opt in result['options']:
            question_text += f"{opt}\n"
            
        print(f"[Debug] Answer: {result['answer']}")
        
        return {
            "current_question": question_text,
            "options": result['options'],
            "correct_answer": result['answer'],
            "explanation": result['explanation'],
            "question_solved": False,
            "attempt_count": 0,
            "last_evaluation": None,
            "messages": [AIMessage(content=f"[문제] {question_text}")]
        }
    
    else:
        # 기존 로직
        question = generate_question_chain.invoke({
            "topic": topic, 
            "difficulty": difficulty,
            "question_type": question_type
        })
        
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
    question_type = state.get("question_type", "conceptual")
    
    # 객관식인 경우 단순 비교
    if question_type == "multiple_choice":
        user_answer = state["messages"][-1].content.strip()
        correct_answer = str(state.get("correct_answer", "")).strip()
        
        # 간단한 번호 비교 (예: "1" == "1")
        # 혹은 "1." 같은 포맷 처리 필요할 수 있음. 일단 단순 비교.
        is_pass = (user_answer == correct_answer)
        
        result_msg = "[PASS] 정답입니다!" if is_pass else f"[FAIL] 오답입니다. (정답: {correct_answer})"
        
        if is_pass and state.get("explanation"):
            result_msg += f"\n\n[해설] {state['explanation']}"
            
        return {
            "question_solved": is_pass,
            "last_evaluation": "Pass" if is_pass else "Fail",
            "messages": [AIMessage(content=result_msg)]
        }

    else:
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
