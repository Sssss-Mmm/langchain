"""
Financial Calculation Tool - 금융 계산 도구
포트폴리오 분석에 필요한 각종 금융 계산 기능
"""
import yfinance as yf
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from typing import List
import json


@tool
def calculate_portfolio_value(holdings: str) -> str:
    """
    포트폴리오의 현재 가치와 수익률을 계산합니다.
    
    Args:
        holdings: JSON 형식의 보유 종목 정보
                  예: '[{"ticker": "005930.KS", "quantity": 100, "avg_price": 75000}, ...]'
    
    Returns:
        각 종목별 현재가치, 손익, 전체 포트폴리오 가치 및 수익률
    """
    try:
        holdings_data = json.loads(holdings)
        results = []
        total_invested = 0
        total_current_value = 0
        
        for holding in holdings_data:
            ticker = holding["ticker"]
            quantity = holding["quantity"]
            avg_price = holding["avg_price"]
            invested = quantity * avg_price
            total_invested += invested
            
            # 현재가 조회
            stock = yf.Ticker(ticker)
            current_price = stock.info.get("currentPrice", 
                            stock.info.get("regularMarketPrice", 0))
            
            current_value = quantity * current_price if current_price else 0
            total_current_value += current_value
            
            profit_loss = current_value - invested
            profit_loss_pct = (profit_loss / invested * 100) if invested > 0 else 0
            
            results.append({
                "ticker": ticker,
                "name": stock.info.get("shortName", ticker),
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "invested": invested,
                "current_value": round(current_value, 0),
                "profit_loss": round(profit_loss, 0),
                "profit_loss_pct": round(profit_loss_pct, 2),
            })
        
        total_profit_loss = total_current_value - total_invested
        total_return_pct = (total_profit_loss / total_invested * 100) if total_invested > 0 else 0
        
        summary = {
            "holdings": results,
            "summary": {
                "total_invested": round(total_invested, 0),
                "total_current_value": round(total_current_value, 0),
                "total_profit_loss": round(total_profit_loss, 0),
                "total_return_pct": round(total_return_pct, 2),
            }
        }
        
        return json.dumps(summary, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def calculate_portfolio_risk(tickers: str, period: str = "1y") -> str:
    """
    포트폴리오의 리스크 지표를 계산합니다.
    
    Args:
        tickers: 쉼표로 구분된 티커 심볼 목록 (예: "005930.KS,000660.KS,035720.KS")
        period: 분석 기간 ('3mo', '6mo', '1y', '2y')
    
    Returns:
        각 종목의 변동성, 베타, 상관관계 및 포트폴리오 전체 리스크 지표
    """
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        
        # 주가 데이터 수집
        prices = pd.DataFrame()
        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                prices[ticker] = hist['Close']
        
        if prices.empty:
            return "주가 데이터를 가져올 수 없습니다."
        
        # 일간 수익률 계산
        returns = prices.pct_change().dropna()
        
        # 개별 종목 리스크 지표
        individual_risk = {}
        for ticker in ticker_list:
            if ticker in returns.columns:
                daily_std = returns[ticker].std()
                annual_volatility = daily_std * np.sqrt(252) * 100  # 연환산 변동성
                
                individual_risk[ticker] = {
                    "daily_volatility_pct": round(daily_std * 100, 3),
                    "annual_volatility_pct": round(annual_volatility, 2),
                    "max_daily_loss_pct": round(returns[ticker].min() * 100, 2),
                    "max_daily_gain_pct": round(returns[ticker].max() * 100, 2),
                }
        
        # 상관관계 매트릭스
        correlation_matrix = returns.corr().round(3).to_dict()
        
        # 포트폴리오 리스크 (동일 가중 가정)
        n = len(ticker_list)
        weights = np.array([1/n] * n)
        cov_matrix = returns.cov() * 252  # 연환산
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance) * 100
        
        # VaR (Value at Risk) - 95% 신뢰수준
        portfolio_returns = returns.mean(axis=1)
        var_95 = np.percentile(portfolio_returns, 5) * 100
        
        result = {
            "analysis_period": period,
            "individual_risk": individual_risk,
            "correlation_matrix": correlation_matrix,
            "portfolio_metrics": {
                "portfolio_volatility_pct": round(portfolio_volatility, 2),
                "var_95_daily_pct": round(var_95, 2),
                "interpretation": f"95% 신뢰수준에서 일일 최대 예상 손실률은 {abs(round(var_95, 2))}% 입니다."
            }
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"리스크 계산 오류: {str(e)}"


@tool
def calculate_sharpe_ratio(ticker: str, risk_free_rate: float = 0.035, period: str = "1y") -> str:
    """
    샤프 비율(Sharpe Ratio)을 계산합니다.
    
    Args:
        ticker: 주식 티커 심볼
        risk_free_rate: 무위험 수익률 (기본값: 3.5% = 0.035)
        period: 분석 기간
    
    Returns:
        샤프 비율 및 해석
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return f"{ticker}에 대한 데이터를 찾을 수 없습니다."
        
        # 수익률 계산
        returns = hist['Close'].pct_change().dropna()
        
        # 연환산 수익률 및 변동성
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 샤프 비율
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # 해석
        if sharpe_ratio >= 2:
            interpretation = "매우 우수: 위험 대비 수익이 매우 높음"
        elif sharpe_ratio >= 1:
            interpretation = "우수: 위험 대비 수익이 양호함"
        elif sharpe_ratio >= 0:
            interpretation = "보통: 무위험 수익률을 상회하지만 리스크 고려 필요"
        else:
            interpretation = "미흡: 무위험 자산보다 성과가 나쁨"
        
        result = {
            "ticker": ticker,
            "period": period,
            "annual_return_pct": round(annual_return * 100, 2),
            "annual_volatility_pct": round(annual_volatility * 100, 2),
            "risk_free_rate_pct": round(risk_free_rate * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "interpretation": interpretation,
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"샤프 비율 계산 오류: {str(e)}"


@tool
def calculate_sector_allocation(holdings: str) -> str:
    """
    포트폴리오의 섹터별 배분 현황을 분석합니다.
    
    Args:
        holdings: JSON 형식의 보유 종목 정보
    
    Returns:
        섹터별 투자 비중 및 분산도 분석
    """
    try:
        holdings_data = json.loads(holdings)
        sector_values = {}
        total_value = 0
        
        for holding in holdings_data:
            ticker = holding["ticker"]
            quantity = holding["quantity"]
            
            stock = yf.Ticker(ticker)
            current_price = stock.info.get("currentPrice", 
                            stock.info.get("regularMarketPrice", 0))
            sector = stock.info.get("sector", "Unknown")
            
            value = quantity * current_price if current_price else 0
            total_value += value
            
            if sector in sector_values:
                sector_values[sector] += value
            else:
                sector_values[sector] = value
        
        # 섹터별 비중 계산
        sector_allocation = {}
        for sector, value in sector_values.items():
            pct = (value / total_value * 100) if total_value > 0 else 0
            sector_allocation[sector] = {
                "value": round(value, 0),
                "percentage": round(pct, 2)
            }
        
        # 분산도 분석 (HHI 지수)
        hhi = sum([(v["percentage"]/100)**2 for v in sector_allocation.values()])
        
        if hhi < 0.15:
            diversification = "매우 분산됨: 다양한 섹터에 골고루 투자됨"
        elif hhi < 0.25:
            diversification = "적정 분산: 섹터 분산이 양호함"
        else:
            diversification = "집중됨: 특정 섹터에 집중되어 있어 리스크 분산 필요"
        
        result = {
            "sector_allocation": sector_allocation,
            "total_value": round(total_value, 0),
            "number_of_sectors": len(sector_allocation),
            "hhi_index": round(hhi, 4),
            "diversification_analysis": diversification,
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"섹터 분석 오류: {str(e)}"


# Export tools list
financial_tools = [
    calculate_portfolio_value,
    calculate_portfolio_risk,
    calculate_sharpe_ratio,
    calculate_sector_allocation,
]
