"""
News Analyzer Tool - 뉴스 분석 도구
금융 관련 뉴스 검색 및 분석
"""
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import json


# DuckDuckGo 검색 인스턴스
search = DuckDuckGoSearchRun()


@tool
def search_stock_news(query: str) -> str:
    """
    주식 또는 금융 관련 뉴스를 검색합니다.
    
    Args:
        query: 검색할 키워드 (예: '삼성전자 실적', 'KOSPI 전망')
    
    Returns:
        관련 뉴스 검색 결과
    """
    try:
        # 금융 뉴스 검색을 위한 쿼리 최적화
        enhanced_query = f"{query} 주식 뉴스 증권"
        results = search.run(enhanced_query)
        return results
    except Exception as e:
        return f"뉴스 검색 오류: {str(e)}"


@tool
def search_market_analysis(market: str = "한국") -> str:
    """
    시장 전반에 대한 분석 뉴스를 검색합니다.
    
    Args:
        market: 시장 (예: '한국', '미국', 'KOSPI', 'S&P500')
    
    Returns:
        시장 분석 관련 뉴스
    """
    try:
        query = f"{market} 증시 전망 분석 2026"
        results = search.run(query)
        return results
    except Exception as e:
        return f"시장 분석 검색 오류: {str(e)}"


@tool
def search_sector_news(sector: str) -> str:
    """
    특정 섹터/업종에 대한 뉴스를 검색합니다.
    
    Args:
        sector: 섹터명 (예: '반도체', '2차전지', '바이오', 'AI')
    
    Returns:
        섹터 관련 뉴스
    """
    try:
        query = f"{sector} 업종 주식 전망 투자"
        results = search.run(query)
        return results
    except Exception as e:
        return f"섹터 뉴스 검색 오류: {str(e)}"


# Export tools list
news_tools = [
    search_stock_news,
    search_market_analysis,
    search_sector_news,
]
