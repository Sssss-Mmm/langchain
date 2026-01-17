from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

def compare_splitters():
    print("=== Text Splitter Comparison (Char vs Token) ===")
    
    # 1. 문서 로드 (일부분만 사용)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "Data", "2024_KB_부동산_보고서_최종.pdf")
    
    if not os.path.exists(file_path):
        print("PDF file not found.")
        return

    loader = PyPDFLoader(file_path)
    # 첫 5페이지만 로드하여 테스트
    pages = loader.load()[:5]
    text = "\n\n".join([p.page_content for p in pages])
    print(f"Original Text Length: {len(text)} characters")

    # 2. RecursiveCharacterTextSplitter (기존 방식)
    # 문장 구조나 단락을 고려하여 자름 (기본적으로 \n\n, \n, " " 순서로 시도)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    char_chunks = char_splitter.split_text(text)
    
    # 3. TokenTextSplitter (실무 방식)
    # LLM의 컨텍스트 윈도우 한계는 '토큰' 기준이므로, 토큰 단위로 자르는 것이 더 안전함.
    # (tiktoken 라이브러리 필요)
    try:
        token_splitter = TokenTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        token_chunks = token_splitter.split_text(text)
    except Exception as e:
        print(f"TokenSplitter Error (tiktoken required): {e}")
        token_chunks = []

    # 4. 비교 결과 출력
    print(f"\n[RecursiveCharacterTextSplitter] Generated {len(char_chunks)} chunks")
    print(f"Example Chunk 1 Length: {len(char_chunks[0])} chars")
    
    if token_chunks:
        print(f"\n[TokenTextSplitter] Generated {len(token_chunks)} chunks")
        # 토큰 스플리터로 자른 것도 문자열로 반환됨. 길이는 가변적일 수 있음.
        print(f"Example Chunk 1 Length: {len(token_chunks[0])} chars")
        print("\n--> 토큰 스플리터는 LLM 입력 크기에 최적화되어 있습니다.")

if __name__ == "__main__":
    compare_splitters()
