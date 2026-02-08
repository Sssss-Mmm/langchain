from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# 프로젝트 루트 경로 추가 (모듈 import 문제 해결)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API 라우터 임포트
from app.api import portfolio, market, documents

app = FastAPI(
    title="AI Portfolio Dashboard API",
    description="Backend API for AI Portfolio Dashboard",
    version="1.0.0"
)

# CORS 설정 (Streamlit 등 프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경: 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio Analysis"])
app.include_router(market.router, prefix="/api/v1/market", tags=["Market Research"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["RAG (Documents)"])

# 헬스 체크 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "AI Portfolio Dashboard API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "docs_url": "/docs"  # Swagger UI 경로 안내
    }

# 개발 모드 실행
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
