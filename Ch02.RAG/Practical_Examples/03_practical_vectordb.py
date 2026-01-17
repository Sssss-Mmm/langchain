from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

def filter_search_demo():
    print("=== Vector DB Filtering Demo ===")
    
    # DB 저장 경로 (테스트용이므로 매번 초기화)
    persist_dir = "./practical_db"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    # 1. 문서 생성 (메타데이터 포함)
    docs = [
        Document(page_content="2024년 서울 아파트 가격은 상승세가 예상됩니다.", metadata={"location": "Seoul", "year": 2024}),
        Document(page_content="2024년 부산 아파트 가격은 보합세가 예상됩니다.", metadata={"location": "Busan", "year": 2024}),
        Document(page_content="2023년 서울 아파트 거래량이 급감했습니다.", metadata={"location": "Seoul", "year": 2023}),
        Document(page_content="제주도 부동산 시장은 관광객 감소로 위축되었습니다.", metadata={"location": "Jeju", "year": 2024}),
    ]

    # 2. 벡터 저장소 생성
    print("Creating VectorStore with metadata...")
    db = Chroma.from_documents(
        documents=docs, 
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_dir
    )

    # 3. 기본 검색 (필터 없음)
    query = "아파트 가격 전망"
    print(f"\n[Query]: {query} (No Filter)")
    results = db.similarity_search(query, k=2)
    for doc in results:
        print(f"- {doc.page_content} (Meta: {doc.metadata})")

    # 4. 메타데이터 필터링 검색 (Seoul 지역만)
    print(f"\n[Query]: {query} (Filter: location='Seoul')")
    # Chroma에서는 filter 인자를 통해 메타데이터 필터링 가능
    results_seoul = db.similarity_search(query, k=2, filter={"location": "Seoul"})
    for doc in results_seoul:
        print(f"- {doc.page_content} (Meta: {doc.metadata})")

    # 5. 복합 필터링 (Seoul AND 2024)
    # Chroma의 구체적인 필터 문법은 버전에 따라 다를 수 있으나, $and 등을 지원함.
    # 간단하게 딕셔너리로 AND 조건을 줘봅니다.
    print(f"\n[Query]: {query} (Filter: location='Seoul' AND year=2024)")
    try:
        results_complex = db.similarity_search(
            query, 
            k=2, 
            filter={"$and": [{"location": "Seoul"}, {"year": 2024}]}
        )
        for doc in results_complex:
            print(f"- {doc.page_content} (Meta: {doc.metadata})")
    except Exception as e:
        print(f"Complex filter error: {e}")

    # 정리
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

if __name__ == "__main__":
    filter_search_demo()
