from dotenv import load_dotenv
from typing import Literal
from langgraph.graph import StateGraph, START, END

from state import TutorState
from nodes import generate_question_node, evaluate_answer_node, provide_hint_node

# 0. 환경 설정
load_dotenv()

# 1. 엣지 함수 (조건부 로직)
def check_result(state: TutorState) -> Literal["generate_question", "provide_hint"]:
    """
    채점 결과를 확인하여 다음 단계를 결정합니다.
    - 정답(True) -> 다음 문제 출제 (난이도 조정 가능)
    - 오답(False) -> 힌트 제공
    """
    if state["question_solved"]:
        print("\n[System] 정답입니다! 다음 문제로 넘어갑니다.")
        return "generate_question"
    else:
        print("\n[System] 오답입니다. 힌트를 제공합니다.")
        return "provide_hint"

# 2. 그래프 생성
workflow = StateGraph(TutorState)

# 3. 노드 추가
workflow.add_node("generate_question", generate_question_node)
workflow.add_node("evaluate_answer", evaluate_answer_node)
workflow.add_node("provide_hint", provide_hint_node)

# 4. 엣지 연결

# 시작 -> 문제 생성
workflow.add_edge(START, "generate_question")

# 문제 생성 후 -> 사용자 입력을 기다려야 함 (human-in-the-loop)
# 여기서는 LangGraph의 interrupt_before 기능을 사용하거나, 
# main loop에서 그래프 실행을 끊어서 처리할 수 있습니다.
# CLI 환경에서는 단순히 노드 실행 후 종료하고, 사용자 입력을 받아 다시 실행하는 방식을 씁니다.
# 하지만 여기서는 흐름상 evaluate로 바로 이어지지 않고, 외부 입력을 받아야 하므로
# evaluate_answer 노드 앞에는 명시적인 연결을 끊고, main loop에서 주입합니다.

# 힌트 제공 -> 사용자 입력을 다시 받아야 하므로, 여기서도 흐름이 끊깁니다. (Loop)
# 따라서 그래프 구조는 "단일 턴 처리" 형태로 구성하는 것이 유리합니다.

# 수정된 그래프 설계 (대화형 Loop에 최적화)
# 1. (Generate) -> [END] -> User Input -> (Evaluate) -> (Check) -> (Hint/Generate) -> [END]
# 이 방식은 상태 유지가 복잡할 수 있습니다.
# LangGraph의 'checkpointer' 없이 간단히 구현하기 위해, 
# 하나의 거대한 그래프보다는 '단계별 실행'을 메인 루프에서 제어하겠습니다.

# 하지만 LangGraph의 장점을 살리기 위해 다음과 같이 구성합니다:
# [Generate] -> [Human Input 대기 (가상의 노드)] -> [Evaluate] -> [Check] -> ...
# 여기서는 가장 간단한 형태로 "Evaluate -> Check -> (Generate / Hint)" 부분만 그래프로 묶고,
# Generate는 초기화 단계나 정답 후 단계에서 별도로 호출하는 것이 나을 수도 있습니다.

# [최종 구조]
# START -> generate_question -> END
# (User Input) -> evaluate_answer -> provide_hint (if fail) -> END
# (User Input) -> evaluate_answer -> generate_question (if pass) -> END

# 복잡성을 줄이기 위해, "Evaluate -> Feedback -> Next Action"을 하나의 그래프 흐름으로 만듭니다.
# Generate Question은 "초기 실행" 혹은 "정답 후 실행"되는 노드입니다.

workflow.add_edge("generate_question", END) # 문제 내고 대기
workflow.add_edge("provide_hint", END)      # 힌트 주고 대기

workflow.add_conditional_edges(
    "evaluate_answer",
    check_result,
    {
        "generate_question": "generate_question", # 정답 -> 새 문제
        "provide_hint": "provide_hint"            # 오답 -> 힌트
    }
)

app = workflow.compile()


# 5. 메인 실행 루프
if __name__ == "__main__":
    print("=== AI Python Tutor (종료: q) ===")
    
    # 초기 상태 설정
    state = {
        "topic": "Python Dictionary",
        "difficulty": "Easy",
        "messages": [],
        "attempt_count": 0,
        "question_solved": False
    }
    
    # 1. 첫 문제 생성 실행
    # 'generate_question' 노드만 강제로 실행하여 시작
    initial_events = app.stream(state, config={"configurable": {"thread_id": "1"}})
    # generate_question은 START에 연결되어 있으므로 바로 실행됨.
    
    current_question = ""
    
    # 초기 실행 (문제 출제)
    for event in initial_events:
        for key, value in event.items():
            if "current_question" in value:
                current_question = value["current_question"]
                state.update(value) # 상태 동기화
                print(f"\n{value['messages'][-1].content}")

    while True:
        user_input = input("\n답변 입력: ")
        if user_input.lower() in ["q", "quit"]:
            break
            
        # 사용자 입력을 메시지에 추가
        from langchain_core.messages import HumanMessage
        state["messages"].append(HumanMessage(content=user_input))
        
        # 2. 답변 평가 실행
        # 그래프 진입점을 'evaluate_answer'로 설정하고 싶지만, 
        # StateGraph는 START부터 시작합니다.
        # 따라서 현재 상태에 따라 분기하는 라우터가 필요하거나,
        # 매번 START -> Check State -> Evaluate 로 가야합니다.
        
        # 간단한 해결책: 
        # 이미 문제가 출제된 상태(question_solved=False)라면 
        # 입력을 받아 Evaluate로 가는 별도 진입점을 만듭니다.
        # 하지만 LangGraph는 단일 진입점이 기본이므로, 
        # 여기서는 매번 app.invoke를 하되, 내부적으로 "현재 단계"를 판단하는 노드를 두거나,
        # 단순히 evaluate 로직부터 시작하는 subgraph를 쓸 수도 있습니다.
        
        # [Refined Logic]
        # app.stream(..., input={"messages": [user_input]}) 
        # -> 하지만 이러면 START부터 다시 Generate로 갈 위험이 있음.
        
        # 가장 쉬운 방법:
        # 이 예제에서는 Graph를 "Evaluate -> Result" 처리용으로만 쓰고,
        # Generate는 Python 코드에서 제어하거나,
        # Graph 내에 "Router" 노드를 두어 (문제 있음/없음)을 판단하게 합니다.
        
        # 여기서는 CLI의 유연함을 위해 
        # `evaluate_answer_node`를 직접 래핑한 그래프를 별도로 돌리거나, 
        # 위에서 정의한 그래프를 그대로 쓰되, START -> evaluate_answer 가 되도록 수정합니다.
        
        # ==> workflow 재정의 (평가 전용)
        eval_workflow = StateGraph(TutorState)
        eval_workflow.add_node("evaluate_answer", evaluate_answer_node)
        eval_workflow.add_node("provide_hint", provide_hint_node)
        eval_workflow.add_node("generate_question", generate_question_node)
        
        # 평가부터 시작
        eval_workflow.add_edge(START, "evaluate_answer")
        
        eval_workflow.add_conditional_edges(
            "evaluate_answer",
            check_result,
            {
                "generate_question": "generate_question",
                "provide_hint": "provide_hint"
            }
        )
        # 각 결과 노드 후 종료 (다시 사용자 입력 대기)
        eval_workflow.add_edge("generate_question", END)
        eval_workflow.add_edge("provide_hint", END)
        
        eval_app = eval_workflow.compile()
        
        # 실행
        # 현재 상태를 그대로 전달
        events = eval_app.stream(state)
        
        for event in events:
            for key, value in event.items():
                # 상태 업데이트
                state.update(value)
                if "messages" in value:
                    print(f"\n{value['messages'][-1].content}")
                
                if key == "generate_question":
                    # 새 문제가 나오면 기존 메시지는 초기화하거나 유지 (여기선 유지)
                    # current_question 업데이트
                    current_question = value["current_question"]
