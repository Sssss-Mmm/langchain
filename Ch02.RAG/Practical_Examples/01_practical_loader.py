from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os

# 1. 단일 파일 로더 (기존 방식)
def load_single_pdf():
    print("--- 1. Single PDF Loading ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "Data", "2024_KB_부동산_보고서_최종.pdf")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {os.path.basename(file_path)}")
    print(f"Sample Metadata: {docs[0].metadata}")
    return docs

# 2. 디렉토리 로더 (실무 방식: 폴더 내 모든 파일 로드)
def load_directory_pdfs():
    print("\n--- 2. Directory Loading ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "Data")
    
    # glob 패턴을 사용하여 PDF만 로드
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True 
    )
    
    try:
        docs = loader.load()
        print(f"Loaded {len(docs)} pages total from directory {data_dir}")
        
        # 파일별로 몇 페이지인지 카운트 (메타데이터 활용)
        file_counts = {}
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            file_counts[source] = file_counts.get(source, 0) + 1
            
        print("Pages per file:")
        for file, count in file_counts.items():
            print(f"- {file}: {count} pages")
            
    except Exception as e:
        print(f"Error loading directory: {e}")

if __name__ == "__main__":
    load_single_pdf()
    load_directory_pdfs()
