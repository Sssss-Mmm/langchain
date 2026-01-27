
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# 기존 로직 import
from nodes import generate_question_node, evaluate_answer_node, provide_hint_node
from state import TutorState

# 환경 변수 로드 (API Key)
load_dotenv()

st.title("🤖 AI Python Tutor")
st.caption("당신의 개인 코딩 튜터와 함께 파이썬을 마스터하세요!")

# 1. 사이드바 설정
with st.sidebar:
    st.header("학습 설정")
    topic = st.text_input("주제 (Topic)", value="Python Basics")
    difficulty = st.selectbox("난이도 (Difficulty)", ["Easy", "Medium", "Hard"])
    question_type = st.selectbox("문제 유형 (Type)", ["conceptual", "coding", "debugging"])
    
    if st.button("새로운 학습 시작 / 리셋"):
        st.session_state.messages = []
        st.session_state.current_question = ""
        st.session_state.question_solved = False
        st.session_state.attempt_count = 0
        st.session_state.generated = False # 문제 생성 여부 플래그
        st.rerun()

# 2. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "generated" not in st.session_state:
    st.session_state.generated = False
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "attempt_count" not in st.session_state:
    st.session_state.attempt_count = 0

# 3. 초기 문제 생성 (아직 생성 안 됐거나 리셋 직후)
if not st.session_state.generated:
    # State 초기화
    initial_state: TutorState = {
        "topic": topic,
        "difficulty": difficulty,
        "question_type": question_type,
        "messages": [],
        "attempt_count": 0,
        "question_solved": False,
        "last_evaluation": None
    }
    
    # 문제 생성 노드 실행
    with st.spinner("문제를 생성하고 있습니다..."):
        result = generate_question_node(initial_state)
    
    # 결과 저장
    st.session_state.current_question = result["current_question"]
    st.session_state.messages.append(AIMessage(content=result["messages"][0].content))
    st.session_state.generated = True
    st.rerun()

# 4. 채팅 기록 표시
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# 5. 사용자 입력 처리
if user_input := st.chat_input("답변을 입력하세요..."):
    # 사용자 메시지 표시 및 저장
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # 평가를 위한 State 구성
    # 주의: nodes.py의 evaluate_answer_node는 state["messages"][-1]을 참조함
    current_state: TutorState = {
        "topic": topic,
        "difficulty": difficulty,
        "question_type": question_type,
        "current_question": st.session_state.current_question,
        "messages": st.session_state.messages,
        "attempt_count": st.session_state.attempt_count,
        "question_solved": False, # 평가 전
        "last_evaluation": None
    }
    
    # 5-1. 답변 평가
    with st.spinner("채점 중입니다..."):
        eval_result = evaluate_answer_node(current_state)
    
    is_pass = eval_result["question_solved"]
    eval_msg = eval_result["messages"][0]
    
    # 평가 결과 표시 및 저장
    st.chat_message("assistant").write(eval_msg.content)
    st.session_state.messages.append(eval_msg)
    
    if is_pass:
        # 5-2. 정답인 경우 -> 다음 문제 자동 생성
        st.success("정답입니다! 잠시 후 다음 문제가 출제됩니다.")
        
        # 설정된 주제/난이도 유지하여 새 문제 생성
        # (만약 난이도 자동 조절을 원하면 여기서 difficulty를 변경해서 넘김)
        next_state = current_state.copy()
        next_state["question_solved"] = True
        
        with st.spinner("다음 문제 생성 중..."):
            new_q_result = generate_question_node(next_state)
        
        new_q_msg = new_q_result["messages"][0]
        st.session_state.current_question = new_q_result["current_question"]
        st.session_state.attempt_count = 0
        
        # 결과 표시 및 저장
        st.chat_message("assistant").write(new_q_msg.content)
        st.session_state.messages.append(new_q_msg)
        
    else:
        # 5-3. 오답인 경우 -> 힌트 제공
        st.session_state.attempt_count += 1
        hint_state = current_state.copy()
        hint_state["attempt_count"] = st.session_state.attempt_count
        
        with st.spinner("힌트를 준비 중입니다..."):
            hint_result = provide_hint_node(hint_state)
            
        hint_msg = hint_result["messages"][0]
        
        # 결과 표시 및 저장
        st.chat_message("assistant").write(hint_msg.content)
        st.session_state.messages.append(hint_msg)
