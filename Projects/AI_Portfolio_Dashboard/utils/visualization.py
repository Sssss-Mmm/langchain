"""
Visualization Utilities - 시각화 유틸리티
Plotly 기반 인터랙티브 차트 생성
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List


def create_portfolio_pie_chart(holdings: List[Dict], title: str = "포트폴리오 구성") -> go.Figure:
    """
    포트폴리오 구성 비중을 파이 차트로 시각화합니다.
    
    Args:
        holdings: 보유 종목 정보 목록
        title: 차트 제목
    
    Returns:
        Plotly Figure 객체
    """
    names = [h.get("name", h.get("ticker", "Unknown")) for h in holdings]
    values = [h.get("current_value", 0) for h in holdings]
    
    fig = go.Figure(data=[go.Pie(
        labels=names,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='#FFFFFF', width=2)
        )
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        height=500,
        template="plotly_dark"
    )
    
    return fig


def create_sector_bar_chart(sector_data: Dict, title: str = "섹터별 투자 비중") -> go.Figure:
    """
    섹터별 투자 비중을 바 차트로 시각화합니다.
    
    Args:
        sector_data: 섹터별 데이터 딕셔너리
        title: 차트 제목
    
    Returns:
        Plotly Figure 객체
    """
    sectors = list(sector_data.keys())
    percentages = [sector_data[s].get("percentage", 0) for s in sectors]
    
    # 섹터명 한글화
    sector_mapping = {
        "Technology": "기술",
        "Healthcare": "헬스케어",
        "Financial Services": "금융",
        "Consumer Cyclical": "경기소비재",
        "Consumer Defensive": "필수소비재",
        "Industrials": "산업재",
        "Energy": "에너지",
        "Materials": "소재",
        "Communication Services": "통신서비스",
        "Real Estate": "부동산",
        "Utilities": "유틸리티",
    }
    sectors_kr = [sector_mapping.get(s, s) for s in sectors]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sectors_kr,
            y=percentages,
            text=[f'{p:.1f}%' for p in percentages],
            textposition='outside',
            marker_color=px.colors.qualitative.Plotly[:len(sectors)]
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="섹터",
        yaxis_title="비중 (%)",
        height=400,
        template="plotly_dark",
        xaxis_tickangle=-45
    )
    
    return fig


def create_performance_line_chart(
    price_data: pd.DataFrame, 
    title: str = "가격 추이"
) -> go.Figure:
    """
    주가 추이를 라인 차트로 시각화합니다.
    
    Args:
        price_data: 날짜를 인덱스로 하는 가격 데이터
        title: 차트 제목
    
    Returns:
        Plotly Figure 객체
    """
    fig = go.Figure()
    
    for column in price_data.columns:
        # 수익률로 정규화 (시작점 = 100)
        normalized = (price_data[column] / price_data[column].iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=normalized,
            name=column,
            mode='lines',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="날짜",
        yaxis_title="수익률 (시작=100)",
        height=400,
        template="plotly_dark",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


def create_correlation_heatmap(
    correlation_matrix: Dict, 
    title: str = "종목간 상관관계"
) -> go.Figure:
    """
    종목간 상관관계를 히트맵으로 시각화합니다.
    
    Args:
        correlation_matrix: 상관관계 매트릭스 딕셔너리
        title: 차트 제목
    
    Returns:
        Plotly Figure 객체
    """
    # Dict를 DataFrame으로 변환
    df = pd.DataFrame(correlation_matrix)
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale='RdYlGn',
        zmin=-1,
        zmax=1,
        text=df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=400,
        template="plotly_dark",
        xaxis_tickangle=-45
    )
    
    return fig


def create_risk_gauge(
    volatility: float, 
    max_volatility: float = 50,
    title: str = "포트폴리오 변동성"
) -> go.Figure:
    """
    포트폴리오 변동성을 게이지 차트로 시각화합니다.
    
    Args:
        volatility: 변동성 (%)
        max_volatility: 최대 표시 변동성
        title: 차트 제목
    
    Returns:
        Plotly Figure 객체
    """
    # 리스크 레벨 결정
    if volatility < 15:
        color = "green"
        risk_level = "낮음"
    elif volatility < 25:
        color = "yellow"
        risk_level = "중간"
    else:
        color = "red"
        risk_level = "높음"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=volatility,
        title={'text': f"{title}<br><span style='font-size:0.8em;color:gray'>리스크 레벨: {risk_level}</span>"},
        number={'suffix': '%', 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, max_volatility], 'ticksuffix': '%'},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 15], 'color': 'rgba(0,255,0,0.3)'},
                {'range': [15, 25], 'color': 'rgba(255,255,0,0.3)'},
                {'range': [25, max_volatility], 'color': 'rgba(255,0,0,0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': volatility
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        template="plotly_dark"
    )
    
    return fig


def create_profit_loss_bar(holdings: List[Dict], title: str = "종목별 손익") -> go.Figure:
    """
    종목별 손익을 바 차트로 시각화합니다.
    
    Args:
        holdings: 보유 종목 정보 목록
        title: 차트 제목
    
    Returns:
        Plotly Figure 객체
    """
    names = [h.get("name", h.get("ticker", "Unknown")) for h in holdings]
    pnl = [h.get("profit_loss_pct", 0) for h in holdings]
    colors = ['green' if p >= 0 else 'red' for p in pnl]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=pnl,
            text=[f'{p:+.1f}%' for p in pnl],
            textposition='outside',
            marker_color=colors
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="종목",
        yaxis_title="수익률 (%)",
        height=400,
        template="plotly_dark",
        xaxis_tickangle=-45
    )
    
    # 0선 추가
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    
    return fig
