"""
AI 자산 포트폴리오 분석 대시보드
LangChain + Streamlit 기반 금융 분석 시스템
"""
import streamlit as st
import json
import os
import sys
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.portfolio_analyst import get_portfolio_analyst_agent
from agents.market_researcher import get_market_researcher_agent
from agents.risk_assessor import get_risk_assessor_agent
from utils.visualization import (
    create_portfolio_pie_chart,
    create_sector_bar_chart,
    create_profit_loss_bar,
    create_risk_gauge,
    create_correlation_heatmap,
    create_performance_line_chart,
)
from config.settings import APP_TITLE, APP_ICON

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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 8px;
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
    """포트폴리오 데이터 조회"""
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
    
    # 에이전트 설명
    if agent_type == "📊 포트폴리오 분석가":
        st.info("보유 종목의 가치 분석, 수익률 계산, 섹터 분산 평가를 수행합니다.")
    elif agent_type == "🔍 시장 리서처":
        st.info("실시간 시장 동향, 뉴스 분석, 종목 정보 조회를 수행합니다.")
    else:
        st.info("변동성, VaR, 상관관계 등 리스크 지표를 분석합니다.")
    
    st.markdown("---")
    
    # 포트폴리오 입력
    st.subheader("📝 포트폴리오 관리")
    
    if st.button("📥 샘플 데이터 로드", use_container_width=True):
        st.session_state.portfolio = load_sample_portfolio()
        st.success("샘플 포트폴리오를 로드했습니다!")
    
    # 대화 초기화
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Powered by LangChain & OpenAI GPT-4o")
    st.caption("© 2026 AI Portfolio Analyzer")

# ==========================================
# 메인 컨텐츠
# ==========================================
st.markdown('<h1 class="main-header">🏦 AI 자산 포트폴리오 분석기</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; color: #888; margin-bottom: 2rem;">
    LangChain과 GPT-4를 활용한 지능형 포트폴리오 분석 시스템
</p>
""", unsafe_allow_html=True)

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📈 대시보드", "💬 AI 상담", "📋 포트폴리오 상세"])

# ==========================================
# 탭 1: 대시보드
# ==========================================
with tab1:
    if st.session_state.portfolio:
        holdings = st.session_state.portfolio.get("holdings", [])
        
        with st.spinner("포트폴리오 데이터를 불러오는 중..."):
            portfolio_data, total_invested, total_current = get_portfolio_data(holdings)
        
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        # 상단 지표 카드
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 총 투자금",
                value=f"₩{total_invested:,.0f}",
            )
        
        with col2:
            st.metric(
                label="💵 현재 가치",
                value=f"₩{total_current:,.0f}",
                delta=f"₩{total_pnl:,.0f}"
            )
        
        with col3:
            st.metric(
                label="📊 총 수익률",
                value=f"{total_pnl_pct:+.2f}%",
                delta="수익" if total_pnl >= 0 else "손실"
            )
        
        with col4:
            st.metric(
                label="📦 보유 종목 수",
                value=f"{len(holdings)}개"
            )
        
        st.markdown("---")
        
        # 차트 영역
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥧 포트폴리오 구성")
            fig_pie = create_portfolio_pie_chart(portfolio_data)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📊 종목별 손익")
            fig_pnl = create_profit_loss_bar(portfolio_data)
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        # 섹터 분석
        st.subheader("🏢 섹터별 분포")
        sector_data = {}
        for item in portfolio_data:
            sector = item.get("sector", "Unknown")
            if sector not in sector_data:
                sector_data[sector] = {"value": 0, "percentage": 0}
            sector_data[sector]["value"] += item["current_value"]
        
        for sector in sector_data:
            sector_data[sector]["percentage"] = (sector_data[sector]["value"] / total_current * 100) if total_current > 0 else 0
        
        fig_sector = create_sector_bar_chart(sector_data)
        st.plotly_chart(fig_sector, use_container_width=True)
        
    else:
        st.warning("포트폴리오 데이터가 없습니다. 사이드바에서 샘플 데이터를 로드하세요.")

# ==========================================
# 탭 2: AI 상담
# ==========================================
with tab2:
    st.subheader(f"💬 {agent_type} AI 상담")
    
    # 에이전트 로드
    @st.cache_resource
    def load_agent(agent_name):
        if "포트폴리오 분석가" in agent_name:
            return get_portfolio_analyst_agent()
        elif "시장 리서처" in agent_name:
            return get_market_researcher_agent()
        elif "리스크 평가사" in agent_name:
            return get_risk_assessor_agent()
        return None
    
    # 에이전트 변경 시 대화 초기화
    if st.session_state.current_agent != agent_type:
        st.session_state.messages = []
        st.session_state.current_agent = agent_type
    
    agent = load_agent(agent_type)
    
    # 예시 질문 버튼
    st.markdown("**💡 예시 질문:**")
    example_cols = st.columns(3)
    
    examples = {
        "📊 포트폴리오 분석가": [
            "내 포트폴리오 현재 가치는?",
            "섹터별 분산이 잘 되어 있나요?",
            "삼성전자 비중을 조절해야 할까요?"
        ],
        "🔍 시장 리서처": [
            "삼성전자 최신 뉴스 알려줘",
            "반도체 업종 전망은?",
            "KOSPI 시장 분석해줘"
        ],
        "⚠️ 리스크 평가사": [
            "내 포트폴리오 리스크 수준은?",
            "VaR 계산해줘",
            "종목간 상관관계 분석해줘"
        ]
    }
    
    for i, example in enumerate(examples.get(agent_type, [])):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()
    
    st.markdown("---")
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input(f"{agent_type}에게 질문하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.status("🤔 AI가 분석 중...", expanded=True) as status:
                try:
                    # 포트폴리오 컨텍스트 추가
                    context = ""
                    if st.session_state.portfolio:
                        context = f"\n\n[현재 포트폴리오 정보]\n{json.dumps(st.session_state.portfolio, ensure_ascii=False)}"
                    
                    full_prompt = prompt + context
                    
                    # 에이전트 실행
                    response = agent.invoke({"messages": [("user", full_prompt)]})
                    
                    # 도구 사용 시각화
                    for msg in response.get('messages', []):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                st.write(f"🔧 **도구 사용**: `{tool_call['name']}`")
                        elif hasattr(msg, 'type') and msg.type == "tool":
                            with st.expander(f"📄 도구 결과"):
                                content_str = str(msg.content)
                                st.code(content_str[:500] + ("..." if len(content_str) > 500 else ""))
                    
                    # 최종 응답
                    final_answer = response['messages'][-1].content
                    status.update(label="✅ 분석 완료!", state="complete", expanded=False)
                    
                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    
                except Exception as e:
                    status.update(label="❌ 오류 발생", state="error")
                    st.error(f"오류가 발생했습니다: {str(e)}")

# ==========================================
# 탭 3: 포트폴리오 상세
# ==========================================
with tab3:
    st.subheader("📋 보유 종목 상세")
    
    if st.session_state.portfolio:
        holdings = st.session_state.portfolio.get("holdings", [])
        portfolio_data, _, _ = get_portfolio_data(holdings)
        
        # DataFrame으로 변환
        df = pd.DataFrame(portfolio_data)
        
        # 컬럼명 한글화
        df_display = df.rename(columns={
            "ticker": "티커",
            "name": "종목명",
            "sector": "섹터",
            "quantity": "수량",
            "avg_price": "평균단가",
            "current_price": "현재가",
            "invested": "투자금",
            "current_value": "평가금",
            "profit_loss": "손익",
            "profit_loss_pct": "수익률(%)"
        })
        
        # 숫자 포맷팅
        st.dataframe(
            df_display.style.format({
                "평균단가": "₩{:,.0f}",
                "현재가": "₩{:,.0f}",
                "투자금": "₩{:,.0f}",
                "평가금": "₩{:,.0f}",
                "손익": "₩{:,.0f}",
                "수익률(%)": "{:+.2f}%"
            }).applymap(
                lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red' if isinstance(x, (int, float)) and x < 0 else '',
                subset=["손익", "수익률(%)"]
            ),
            use_container_width=True,
            height=400
        )
        
        # 포트폴리오 메모
        st.markdown("---")
        st.subheader("📝 포트폴리오 메모")
        notes = st.session_state.portfolio.get("notes", "")
        st.info(notes if notes else "등록된 메모가 없습니다.")
        
    else:
        st.warning("포트폴리오 데이터가 없습니다.")
