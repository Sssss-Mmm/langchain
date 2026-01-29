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

    # 과목 정의
    subjects = {
        "1": "소프트웨어 설계",
        "2": "소프트웨어 개발",
        "3": "데이터베이스 구축",
        "4": "프로그래밍 언어 활용",
        "5": "정보시스템 구축 관리"
    }
    
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        print(f"Loading {file}...")
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # 과목 추적 로직
            current_subject = "Unknown"
            
            for doc in docs:
                content = doc.page_content
                # 헤더 감지 (워터마크나 노이즈가 있을 수 있으므로 키워드로 찾음)
                # 예: "제1과목", "1과목", "소프트웨어 설계" 등
                if "1과목" in content or "소프트웨어 설계" in content:
                    current_subject = "1.소프트웨어 설계"
                elif "2과목" in content or "소프트웨어 개발" in content:
                    current_subject = "2.소프트웨어 개발"
                elif "3과목" in content or "데이터베이스" in content:
                    current_subject = "3.데이터베이스 구축"
                elif "4과목" in content or "프로그래밍 언어" in content:
                    current_subject = "4.프로그래밍 언어 활용"
                elif "5과목" in content or "정보시스템" in content:
                    current_subject = "5.정보시스템 구축 관리"
                
                doc.metadata['source_file'] = file
                doc.metadata['subject'] = current_subject
                
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
