from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class TutorState(TypedDict):
    """
    AI 코딩 튜터의 그래프 상태 정의
    """
    # 대화 이력을 저장 (HumanMessage, AIMessage 등)
    messages: Annotated[list, add_messages]
    
    # 현재 학습 주제 (예: "Python List", "Functions")
    topic: str
    
    # 현재 난이도 ("Easy", "Medium", "Hard")
    difficulty: str
    
    # 현재 출제된 문제
    current_question: str
    
    # 현재 문제의 정답 해설 (채점 시 참고용, 사용자에게는 바로 안 보여줌)
    current_answer_context: str
    
    # 현재 문제가 해결되었는지 여부 (True면 다음 문제로 넘어감)
    question_solved: bool
    
    # 현재 문제에 대한 사용자의 시도 횟수
    attempt_count: int
    
    # 마지막 평가 결과 ("Pass", "Fail")
    last_evaluation: Literal["Pass", "Fail", None]
