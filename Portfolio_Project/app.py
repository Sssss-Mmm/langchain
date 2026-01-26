import streamlit as st
import os
from dotenv import load_dotenv

# Import Agent Constructors
from agents.web_search import get_web_search_agent
from agents.python_repl import get_python_repl_agent
from agents.youtube_analysis import get_youtube_agent

# 0. 환경 변수 로드
load_dotenv()

# ==========================================
# 1. UI 설정
# ==========================================
st.set_page_config(page_title="AI Multi-Agent Portfolio", page_icon="🤖", layout="wide")

st.title("🤖 AI Multi-Agent Dashboard")
st.markdown("""
이 대시보드는 **LangChain**과 **LangGraph**를 활용하여 구축된 다양한 AI 에이전트들을 체험할 수 있는 공간입니다.
왼쪽 사이드바에서 원하는 에이전트를 선택하고 대화를 시작해보세요.
""")

# ==========================================
# 2. 사이드바 (에이전트 선택)
# ==========================================
with st.sidebar:
    st.header("🕵️‍♂️ 에이전트 선택")
    agent_type = st.radio(
        "사용할 에이전트를 선택하세요:",
        ("🌐 Web Search Agent", "🐍 Python Code Interpreter", "📺 YouTube Analyst")
    )
    
    st.markdown("---")
    st.markdown("### 📝 기능 설명")
    if agent_type == "🌐 Web Search Agent":
        st.info("DuckDuckGo를 통해 인터넷의 최신 정보를 검색하고 요약합니다.")
    elif agent_type == "🐍 Python Code Interpreter":
        st.info("파이썬 코드를 직접 작성하고 실행하여 복잡한 계산이나 데이터를 처리합니다.")
    elif agent_type == "📺 YouTube Analyst":
        st.info("유튜브 영상의 자막을 분석하여 내용을 요약하거나 질문에 답변합니다.")

# ==========================================
# 3. 세션 상태 관리 (채팅 기록 및 에이전트 로드)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 에이전트가 변경되면 대화 내용 초기화
if "current_agent" not in st.session_state:
    st.session_state.current_agent = agent_type
elif st.session_state.current_agent != agent_type:
    st.session_state.messages = []
    st.session_state.current_agent = agent_type

# 에이전트 로드 함수
@st.cache_resource
def load_agent(agent_name):
    if agent_name == "🌐 Web Search Agent":
        return get_web_search_agent()
    elif agent_name == "🐍 Python Code Interpreter":
        return get_python_repl_agent()
    elif agent_name == "📺 YouTube Analyst":
        return get_youtube_agent()
    return None

agent_executor = load_agent(agent_type)

# ==========================================
# 4. 채팅 인터페이스
# ==========================================

# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("무엇을 도와드릴까요?"):
    # 1. 사용자 메시지 표시 및 저장
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. 에이전트 실행 및 응답 처리
    with st.chat_message("assistant"):
        # Streamlit Status Container를 사용하여 내부 동작(Tool Call) 시각화
        with st.status("🛠️ AI가 생각하고 도구를 사용하는 중...", expanded=True) as status:
            try:
                # LangGraph 실행 (스트리밍 방식이 아니므로 invoke 사용)
                # 실제 동작 과정을 보여주기 위해 중간 단계 출력 로직이 필요하지만,
                # Streamlit 구조상 invoke 결과를 받아서 사후에 보여주거나, callback을 써야 함.
                # 여기서는 간단히 invoke 결과를 파싱해서 보여주는 방식으로 구현.
                
                response = agent_executor.invoke({"messages": [("user", prompt)]})
                
                # Tool Call 및 결과 시각화
                for msg in response['messages']:
                    if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
                        for tool_call in msg.tool_calls:
                            st.write(f"👉 **Tool Call**: `{tool_call['name']}`")
                            st.json(tool_call['args'])
                    elif msg.type == "tool":
                        with st.expander(f"✅ Tool Result ({len(str(msg.content))} chars)"):
                            st.code(str(msg.content)[:1000] + ("..." if len(str(msg.content)) > 1000 else ""))
                
                # 최종 답변 추출
                final_answer = response['messages'][-1].content
                
                status.update(label="✅ 완료!", state="complete", expanded=False)
                
                # 최종 답변 표시
                st.markdown(final_answer)
                
                # 대화 기록 저장
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
            except Exception as e:
                status.update(label="❌ 오류 발생", state="error")
                st.error(f"에러가 발생했습니다: {e}")

# ==========================================
# 5. 사이드바 하단 정보
# ==========================================
with st.sidebar:
    st.markdown("---")
    st.caption("Powered by LangChain & OpenAI GPT-4o")
    st.caption("Developed by [Your Name]")
