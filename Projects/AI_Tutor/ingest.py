from dotenv import load_dotenv
import os

# 환경 변수 로드 시도 (여러 경로)
load_dotenv() # 현재 디렉토리 및 상위
load_dotenv("/home/sssss_mmm/langchain/.env")
load_dotenv("/home/sssss_mmm/langchain/Portfolio_Project/.env")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 데이터 경로 및 저장 경로 설정
DATA_DIR = "/home/sssss_mmm/langchain/Projects/AI_Tutor/Data"
DB_PATH = "/home/sssss_mmm/langchain/Projects/AI_Tutor/VectorStore"

def ingest_data():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        return

    documents = []
    
    # PDF 파일 목록 가져오기
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDF files.")

    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        print(f"Loading {file}...")
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # 메타데이터에 소스 파일명 추가 (기본적으로 있지만 명시적으로 확인)
            for doc in docs:
                doc.metadata['source_file'] = file
            documents.extend(docs)
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    print(f"Total pages loaded: {len(documents)}")

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    print(f"Total text chunks: {len(splits)}")

    # 임베딩 및 벡터 스토어 생성
    print("Creating Vector Store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    # 로컬 저장
    vectorstore.save_local(DB_PATH)
    print(f"Vector Store saved to {DB_PATH}")

if __name__ == "__main__":
    ingest_data()
