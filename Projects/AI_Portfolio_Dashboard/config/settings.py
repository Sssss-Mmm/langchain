"""
AI Portfolio Dashboard Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"

# App Configuration
APP_TITLE = "🏦 AI 자산 포트폴리오 분석기"
APP_ICON = "📈"

# Default Portfolio Settings
DEFAULT_BENCHMARK = "^KS11"  # KOSPI Index
DEFAULT_RISK_FREE_RATE = 0.035  # 3.5% (한국 기준금리 기준)

# Supported Markets
SUPPORTED_MARKETS = {
    "KR": {"suffix": ".KS", "name": "한국 (KOSPI)"},
    "KQ": {"suffix": ".KQ", "name": "한국 (KOSDAQ)"},
    "US": {"suffix": "", "name": "미국"},
}

# Sector Mapping (Korean)
SECTOR_MAPPING = {
    "Technology": "기술",
    "Healthcare": "헬스케어",
    "Financial Services": "금융",
    "Consumer Cyclical": "경기소비재",
    "Consumer Defensive": "필수소비재",
    "Industrials": "산업재",
    "Energy": "에너지",
    "Materials": "소재",
    "Utilities": "유틸리티",
    "Real Estate": "부동산",
    "Communication Services": "통신서비스",
}
