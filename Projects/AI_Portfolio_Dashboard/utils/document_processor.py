"""
Document Processor Utility
PDF 문서 처리 및 벡터 저장소 관리
"""
import os
import tempfile
from typing import List, Any
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings # 옵션: OpenAI 임베딩 사용 시

class DocumentProcessor:
    def __init__(self):
        # 임베딩 모델 초기화 (한국어 성능이 우수한 다국어 모델 사용 권장)
        # 여기서는 가볍고 성능 좋은 모델 사용
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask" 
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def process_pdf(self, uploaded_file) -> List[Any]:
        """
        업로드된 PDF 파일을 처리하여 문서 청크로 변환합니다.
        """
        if uploaded_file is None:
            return []
            
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            # PDF 로드
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # 텍스트 분할
            splits = self.text_splitter.split_documents(documents)
            
            # 임시 파일 삭제
            os.remove(tmp_path)
            
            return splits
            
        except Exception as e:
            st.error(f"문서 처리 중 오류 발생: {str(e)}")
            return []

    def create_vector_store(self, documents: List[Any], collection_name: str = "terms_collection"):
        """
        문서 청크로부터 벡터 저장소를 생성합니다.
        Streamlit 환경에서는 메모리에 유지하거나 세션별로 관리하는 것이 좋습니다.
        """
        if not documents:
            return None
            
        try:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                # persist_directory="./chroma_db" # 영구 저장 시 사용
            )
            return vector_store
            
        except Exception as e:
            st.error(f"벡터 저장소 생성 오류: {str(e)}")
            return None
