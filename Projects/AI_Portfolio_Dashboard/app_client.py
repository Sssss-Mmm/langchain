"""
AI 자산 포트폴리오 분석 대시보드 (Client App)
FastAPI 백엔드와 통신하는 Streamlit 프론트엔드
"""
import streamlit as st
import json
import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv
import yfinance as yf

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.visualization import (
    create_portfolio_pie_chart,
    create_sector_bar_chart,
    create_profit_loss_bar,
    create_risk_gauge,
    create_performance_line_chart,
)
from config.settings import APP_TITLE, APP_ICON

# API 설정
API_BASE_URL = "http://localhost:8000/api/v1"

# 환경 변수 로드
load_dotenv()

# ==========================================
# 페이지 설정
# ==========================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 유틸리티 함수
# ==========================================
def load_sample_portfolio():
    """샘플 포트폴리오 로드"""
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_portfolio.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_portfolio_data(holdings):
    """포트폴리오 데이터 조회 (Frontend에서 직접 계산)"""
    # 백엔드로 이관 가능하지만, 시각화를 위해 일단 유지
    portfolio_data = []
    total_invested = 0
    total_current = 0
    
    for holding in holdings:
        ticker = holding["ticker"]
        quantity = holding["quantity"]
        avg_price = holding["avg_price"]
        invested = quantity * avg_price
        total_invested += invested
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get("currentPrice", info.get("regularMarketPrice", avg_price))
            current_value = quantity * current_price
            total_current += current_value
            
            portfolio_data.append({
                "ticker": ticker,
                "name": holding.get("name", info.get("shortName", ticker)),
                "sector": holding.get("sector", info.get("sector", "N/A")),
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "invested": invested,
                "current_value": current_value,
                "profit_loss": current_value - invested,
                "profit_loss_pct": ((current_value - invested) / invested) * 100 if invested > 0 else 0
            })
        except Exception as e:
            portfolio_data.append({
                "ticker": ticker,
                "name": holding.get("name", ticker),
                "sector": holding.get("sector", "N/A"),
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": avg_price,
                "invested": invested,
                "current_value": invested,
                "profit_loss": 0,
                "profit_loss_pct": 0
            })
    
    return portfolio_data, total_invested, total_current

# ==========================================
# 세션 상태 초기화
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_sample_portfolio()
if "current_agent" not in st.session_state:
    st.session_state.current_agent = None

# ==========================================
# 사이드바
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/profit.png", width=80)
    st.title("🏦 포트폴리오 설정")
    
    st.markdown("---")
    
    # 에이전트 선택
    st.subheader("🤖 AI 에이전트 선택")
    agent_type = st.radio(
        "분석 유형을 선택하세요:",
        [
            "📊 포트폴리오 분석가",
            "🔍 시장 리서처",
            "⚠️ 리스크 평가사"
        ],
        help="각 에이전트는 특화된 분석 기능을 제공합니다."
    )
    
    # 에이전트 변경 시 대화 초기화
    if st.session_state.current_agent != agent_type:
        st.session_state.messages = []
        st.session_state.current_agent = agent_type
    
    # 설명 표시
    if agent_type == "📊 포트폴리오 분석가":
        st.info("보유 종목의 가치 분석, 수익률 계산, 섹터 분산 평가를 수행합니다.")
    elif agent_type == "🔍 시장 리서처":
        st.info("실시간 시장 동향, 뉴스 분석, 종목 정보 조회를 수행합니다.")
    else:
        st.info("변동성, VaR, 상관관계 등 리스크 지표를 분석합니다.")
    
    st.markdown("---")
    
    # 포트폴리오 입력
    if st.button("📥 샘플 데이터 로드", use_container_width=True):
        st.session_state.portfolio = load_sample_portfolio()
        st.success("샘플 포트폴리오를 로드했습니다!")
    
    # 대화 초기화
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Powered by FastAPI & LangChain")


# ==========================================
# 메인 컨텐츠
# ==========================================
st.markdown('<h1 class="main-header">🏦 AI 자산 포트폴리오 분석기 (Client)</h1>', unsafe_allow_html=True)

# 탭 구성
tab1, tab2, tab3, tab4 = st.tabs(["📈 대시보드", "💬 AI 상담", "📋 포트폴리오 상세", "📜 약관 분석"])

# ==========================================
# 탭 1: 대시보드
# ==========================================
with tab1:
    if st.session_state.portfolio:
        holdings = st.session_state.portfolio.get("holdings", [])
        
        with st.spinner("데이터 갱신 중..."):
            portfolio_data, total_invested, total_current = get_portfolio_data(holdings)
        
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 총 투자금", f"₩{total_invested:,.0f}")
        col2.metric("💵 현재 가치", f"₩{total_current:,.0f}", f"₩{total_pnl:,.0f}")
        col3.metric("📊 총 수익률", f"{total_pnl_pct:+.2f}%", "수익" if total_pnl >= 0 else "손실")
        col4.metric("📦 보유 종목 수", f"{len(holdings)}개")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🥧 포트폴리오 구성")
            st.plotly_chart(create_portfolio_pie_chart(portfolio_data), use_container_width=True)
        with col2:
            st.subheader("📊 종목별 손익")
            st.plotly_chart(create_profit_loss_bar(portfolio_data), use_container_width=True)
            
        # 섹터 차트는 생략 (길어서)
        
    else:
        st.warning("포트폴리오 데이터가 없습니다.")

# ==========================================
# 탭 2: AI 상담 (API 연동)
# ==========================================
with tab2:
    st.subheader(f"💬 {agent_type} AI 상담")
    
    # 예시 질문
    examples = {
        "📊 포트폴리오 분석가": ["내 포트폴리오 통계 알려줘", "반도체 비중이 얼마나 돼?", "수익률이 가장 낮은 종목은?"],
        "🔍 시장 리서처": ["삼성전자 최신 뉴스해줘", "미국 금리 전망은?", "테슬라 주가 흐름 분석해줘"],
        "⚠️ 리스크 평가사": ["내 포트폴리오 위험도는?", "분산 투자가 잘 되어 있나?"]
    }
    
    cols = st.columns(3)
    for i, ex in enumerate(examples.get(agent_type, [])):
        if cols[i].button(ex, key=f"ex_{i}"):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.session_state["processing_prompt"] = ex
            st.rerun()

    # 채팅 히스토리
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # 입력 처리
    prompt = None
    if chat_input := st.chat_input(f"{agent_type}에게 질문하세요..."):
        prompt = chat_input
        st.session_state.messages.append({"role": "user", "content": prompt})
    elif "processing_prompt" in st.session_state and st.session_state["processing_prompt"]:
        prompt = st.session_state["processing_prompt"]
        del st.session_state["processing_prompt"]
        
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.status("🚀 API 서버에 요청 중...", expanded=True) as status:
                try:
                    # API Payload 준비
                    payload = {"query": prompt}
                    endpoint = ""
                    
                    if "시장" in agent_type:
                        endpoint = "/market/research"
                    elif "리스크" in agent_type:
                        # 리스크도 일단 포트폴리오 API 사용 (추후 분리)
                        endpoint = "/portfolio/analyze"
                        payload["portfolio"] = st.session_state.portfolio
                    else:
                        endpoint = "/portfolio/analyze"
                        payload["portfolio"] = st.session_state.portfolio
                    
                    # API 호출
                    api_url = f"{API_BASE_URL}{endpoint}"
                    st.write(f"Connecting to: `{api_url}`") # DEBUG
                    
                    response = requests.post(api_url, json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "응답 없음")
                        tool_calls = data.get("tool_calls", [])
                        
                        status.update(label="✅ 답변 수신 완료!", state="complete", expanded=False)
                        st.markdown(answer)
                        
                        if tool_calls:
                            with st.expander("🔧 사용된 도구"):
                                st.write(tool_calls)
                                
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        status.update(label="❌ API 오류", state="error")
                        st.error(f"Status Code: {response.status_code}")
                        st.json(response.json())
                        
                except requests.exceptions.ConnectionError:
                    status.update(label="❌ 접속 실패", state="error")
                    st.error("API 서버에 연결할 수 없습니다. 서버가 실행 중인가요?")
                    st.code("uv run uvicorn app.main:app --reload")
                except Exception as e:
                    status.update(label="❌ 오류 발생", state="error")
                    st.error(f"Error: {str(e)}")

# ==========================================
# 탭 3: 상세 (생략 - 기존과 동일)
# ==========================================
with tab3:
    st.info("대시보드 탭에서 상세 정보를 확인할 수 있습니다.")

# ==========================================
# 탭 4: 약관 분석 (준비 중)
# ==========================================
with tab4:
    st.subheader("📜 약관 분석 (API 연동 예정)")
    st.warning("이 기능은 백엔드 API 이관 작업 중입니다.")
