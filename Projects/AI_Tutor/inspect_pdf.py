from langchain_community.document_loaders import PyPDFLoader
import os

# 데이터 경로
DATA_DIR = "/home/sssss_mmm/langchain/Projects/AI_Tutor/Data"
sample_file = "1. 2024년1회_정보처리기사필기기출문제.pdf"
file_path = os.path.join(DATA_DIR, sample_file)

if os.path.exists(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    print(f"Total pages: {len(pages)}")
    
    # 처음 5페이지만 출력해서 구조 확인
    for i, page in enumerate(pages[:5]):
        print(f"\n--- Page {i+1} ---")
        print(page.page_content[:1000]) # 1000자까지만
else:
    print(f"File not found: {file_path}")
