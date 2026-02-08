from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
import shutil

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from utils.document_processor import DocumentProcessor # 의존성 문제로 일단 주석 처리

router = APIRouter()

class DocumentResponse(BaseModel):
    filename: str
    message: str
    chunk_count: int = 0

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    PDF 파일을 업로드하고 처리합니다. (현재는 스켈레톤 구현)
    """
    try:
        # 임시 저장 디렉토리
        upload_dir = "temp_uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # TODO: DocumentProcessor 연동
        # processor = DocumentProcessor()
        # splits = processor.process_pdf(file_path)
        
        return DocumentResponse(
            filename=file.filename,
            message="File uploaded successfully. Processing logic pending.",
            chunk_count=0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    query: str
    filename: Optional[str] = None

@router.post("/query")
async def query_document(request: QueryRequest):
    return {"answer": "RAG API is under construction.", "context": []}
