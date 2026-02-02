"""
Stock Data Tool - 주식 데이터 수집 도구
yfinance를 활용한 실시간 주가 및 기업 정보 조회
"""
import yfinance as yf
from langchain_core.tools import tool
from typing import Optional
import json


@tool
def get_stock_price(ticker: str) -> str:
    """
    주식의 현재가 및 기본 정보를 조회합니다.
    
    Args:
        ticker: 주식 티커 심볼 (예: '005930.KS' for 삼성전자, 'AAPL' for Apple)
    
    Returns:
        주식의 현재가, 전일 대비 변동, 시가총액 등의 정보
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 기본 정보 추출
        result = {
            "ticker": ticker,
            "name": info.get("shortName", info.get("longName", "N/A")),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
            "previous_close": info.get("previousClose", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "currency": info.get("currency", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
        }
        
        # 변동률 계산
        if result["current_price"] != "N/A" and result["previous_close"] != "N/A":
            change = result["current_price"] - result["previous_close"]
            change_pct = (change / result["previous_close"]) * 100
            result["change"] = round(change, 2)
            result["change_percent"] = round(change_pct, 2)
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"오류 발생: {str(e)}"


@tool
def get_stock_history(ticker: str, period: str = "1mo") -> str:
    """
    주식의 과거 가격 데이터를 조회합니다.
    
    Args:
        ticker: 주식 티커 심볼
        period: 조회 기간 ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
    
    Returns:
        기간별 시가, 고가, 저가, 종가, 거래량 데이터
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return f"{ticker}에 대한 데이터를 찾을 수 없습니다."
        
        # 요약 통계
        result = {
            "ticker": ticker,
            "period": period,
            "data_points": len(hist),
            "start_date": str(hist.index[0].date()),
            "end_date": str(hist.index[-1].date()),
            "start_price": round(hist['Close'].iloc[0], 2),
            "end_price": round(hist['Close'].iloc[-1], 2),
            "highest": round(hist['High'].max(), 2),
            "lowest": round(hist['Low'].min(), 2),
            "avg_volume": int(hist['Volume'].mean()),
            "total_return_pct": round(
                ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100, 2
            ),
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"오류 발생: {str(e)}"


@tool
def get_stock_financials(ticker: str) -> str:
    """
    주식의 재무 지표를 조회합니다.
    
    Args:
        ticker: 주식 티커 심볼
    
    Returns:
        PER, PBR, ROE, 배당수익률 등의 재무 지표
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        result = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            # Valuation Metrics
            "trailing_pe": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "price_to_book": info.get("priceToBook", "N/A"),
            "price_to_sales": info.get("priceToSalesTrailing12Months", "N/A"),
            # Profitability
            "profit_margins": info.get("profitMargins", "N/A"),
            "return_on_equity": info.get("returnOnEquity", "N/A"),
            "return_on_assets": info.get("returnOnAssets", "N/A"),
            # Dividend
            "dividend_yield": info.get("dividendYield", "N/A"),
            "dividend_rate": info.get("dividendRate", "N/A"),
            # Growth
            "revenue_growth": info.get("revenueGrowth", "N/A"),
            "earnings_growth": info.get("earningsGrowth", "N/A"),
            # Debt
            "debt_to_equity": info.get("debtToEquity", "N/A"),
        }
        
        # 퍼센트 값 변환
        for key in ["profit_margins", "return_on_equity", "return_on_assets", 
                    "dividend_yield", "revenue_growth", "earnings_growth"]:
            if result[key] != "N/A" and result[key] is not None:
                result[key] = f"{round(result[key] * 100, 2)}%"
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"오류 발생: {str(e)}"


@tool
def search_ticker(company_name: str) -> str:
    """
    회사명으로 티커 심볼을 검색합니다.
    
    Args:
        company_name: 검색할 회사명 (예: '삼성전자', 'Apple')
    
    Returns:
        관련 티커 심볼 목록
    """
    try:
        # yfinance의 검색 기능 활용
        search_result = yf.Ticker(company_name)
        info = search_result.info
        
        if info and "symbol" in info:
            result = {
                "query": company_name,
                "found_ticker": info.get("symbol"),
                "company_name": info.get("shortName", info.get("longName", "N/A")),
                "exchange": info.get("exchange", "N/A"),
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            return f"'{company_name}'에 대한 티커를 찾을 수 없습니다. 정확한 티커 심볼을 입력해주세요."
    
    except Exception as e:
        return f"검색 오류: {str(e)}"


# Export tools list
stock_tools = [
    get_stock_price,
    get_stock_history,
    get_stock_financials,
    search_ticker,
]
