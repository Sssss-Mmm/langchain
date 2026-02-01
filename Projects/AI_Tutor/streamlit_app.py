import streamlit as st
import os
from dotenv import load_dotenv

# 환경 변수 로드
# 시스템 환경 변수(예: ~/.zshrc)가 이미 설정되어 있다면 그것을 우선 사용합니다.
# .env 파일이 존재할 경우에만 로드합니다 (override=False가 기본값).
load_dotenv()

# backend imports
from state import TutorState
from nodes import generate_question_node, evaluate_answer_node, provide_hint_node
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 0. 페이지 설정
# ==========================================
st.set_page_config(page_title="Information Processing AI Tutor", page_icon="🎓", layout="wide")

st.title("🎓 정보처리기사 필기 AI Tutor")
st.markdown("RAG(검색 증강 생성) 기반으로 최신 기출문제를 학습할 수 있습니다.")

# ==========================================
# 1. 상태 관리 (Session State)
# ==========================================
if "tutor_state" not in st.session_state:
    st.session_state.tutor_state = None  # 학습 시작 전

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 화면 표시용 대화 기록

# ==========================================
# 2. 사이드바 (설정 & 제어)
# ==========================================
with st.sidebar:
    st.header("⚙️ 학습 설정")
    
    st.info("과목을 선택하거나 랜덤으로 풀 수 있습니다.")
    
    selected_subject = st.selectbox(
        "과목 선택",
        ["Random", "1.소프트웨어 설계", "2.소프트웨어 개발", "3.데이터베이스 구축", "4.프로그래밍 언어 활용", "5.정보시스템 구축 관리"],
        index=0
    )
    
    st.markdown("---")
    
    if st.button("🚀 문제 풀기 / 초기화", type="primary"):
        # 상태 초기화
        st.session_state.tutor_state = {
            "topic": selected_subject, # 과목명을 topic으로 사용
            "difficulty": "Random",
            "question_type": "multiple_choice",
            "messages": [],
            "attempt_count": 0,
            "question_solved": False,
            "last_evaluation": None,
            "correct_answer": "",
            "explanation": "",
            "options": []
        }
        st.session_state.chat_history = []
        
        # 첫 문제 생성
        with st.spinner("문제를 생성하고 있습니다..."):
            new_state_update = generate_question_node(st.session_state.tutor_state)
            st.session_state.tutor_state.update(new_state_update)
            
            # 채팅 기록에 문제 추가
            q_msg = new_state_update["messages"][-1].content
            st.session_state.chat_history.append({"role": "assistant", "content": q_msg})
        
        st.rerun()

# ==========================================
# 3. 메인 콘텐츠 (문제 및 채팅)
# ==========================================

# (1) 대화 기록 표시
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 학습이 시작되지 않았으면 안내 문구
if st.session_state.tutor_state is None:
    st.info("왼쪽 사이드바에서 설정을 확인하고 [학습 시작] 버튼을 눌러주세요.")
    st.stop()

# (2) 현재 문제 및 입력 영역
# 문제가 해결되지 않은 상태일 때만 입력 폼 표시
current_state = st.session_state.tutor_state

if not current_state.get("question_solved", False):
    st.markdown("---")
    st.subheader("📝 문제 풀기")
    
    # 객관식인 경우
    if current_state.get("question_type") == "multiple_choice" and current_state.get("options"):
        # 라디오 버튼으로 보기 표시
        options = current_state["options"]
        # 보기에 번호가 이미 포함되어 있다고 가정 (1. ... )
        # 하지만 라디오 버튼 선택 값을 쉽게 하기 위해 인덱스나 전체 텍스트 사용
        
        selected_option = st.radio(
            "정답을 선택하세요:",
            options,
            index=None,
            key="option_radio"
        )
        
        if st.button("제출하기"):
            if selected_option:
                # 번호만 추출할지, 전체 텍스트를 보낼지 결정
                # 여기서는 evaluate_answer_node가 텍스트 비교 or 번호 비교를 함.
                # 보통 "1. 설명" 형태라면 앞의 "1"만 추출해서 비교하는 게 안전할 수 있음.
                # 하지만 node 로직에 따라 다름. 일단 전체 텍스트에서 번호(첫글자)만 추출하여 보냄.
                
                # 답안 가공 (예: "1. 설명" -> "1")
                answer_to_submit = selected_option.split(".")[0].strip()
                
                # 사용자 메시지 추가
                st.session_state.chat_history.append({"role": "user", "content": f"정답: {answer_to_submit}"})
                current_state["messages"].append(HumanMessage(content=answer_to_submit))
                
                # 채점
                with st.spinner("채점 중..."):
                    eval_result = evaluate_answer_node(current_state)
                    current_state.update(eval_result)
                    
                    # 결과 메시지 추가
                    r_msg = eval_result["messages"][-1].content
                    st.session_state.chat_history.append({"role": "assistant", "content": r_msg})
                
                st.rerun()
            else:
                st.warning("보기를 선택해주세요.")
                
    else:
        # 주관식/코딩/디버깅 등
        user_input = st.chat_input("답변을 입력하세요...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            current_state["messages"].append(HumanMessage(content=user_input))
            
            with st.spinner("AI 튜터가 확인 중..."):
                eval_result = evaluate_answer_node(current_state)
                current_state.update(eval_result)
                
                r_msg = eval_result["messages"][-1].content
                st.session_state.chat_history.append({"role": "assistant", "content": r_msg})
                
                # 오답이면 힌트 제공 로직 (선택적)
                if not eval_result.get("question_solved"):
                    # 힌트 제공 (자동)
                    hint_result = provide_hint_node(current_state)
                    current_state.update(hint_result)
                    h_msg = hint_result["messages"][-1].content
                    st.session_state.chat_history.append({"role": "assistant", "content": h_msg})
            
            st.rerun()

# (3) 문제 해결 후 다음 문제 버튼
if current_state.get("question_solved", False):
    st.markdown("---")
    st.success("🎉 정답입니다! 다음 문제로 넘어갈까요?")
    if st.button("다음 문제 ->", type="primary"):
        # 상태 일부 초기화 (점수 등은 유지 가능하나 여기선 문제 단위 루프)
        # topic/difficulty 등은 유지
        current_state["messages"] = []
        current_state["attempt_count"] = 0
        current_state["question_solved"] = False
        current_state["last_evaluation"] = None
        current_state["correct_answer"] = ""
        current_state["explanation"] = ""
        current_state["options"] = []
        
        with st.spinner("다음 문제를 생성 중입니다..."):
            new_state_update = generate_question_node(current_state)
            current_state.update(new_state_update)
            
            # 채팅 기록에 구분선 및 새 문제 추가
            st.session_state.chat_history.append({"role": "assistant", "content": "--- \n **[새로운 문제]**"})
            q_msg = new_state_update["messages"][-1].content
            st.session_state.chat_history.append({"role": "assistant", "content": q_msg})
        
        st.rerun()
