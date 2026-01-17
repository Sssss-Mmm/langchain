from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def citation_rag_demo():
    print("=== 부동산 보고서 RAG (출처 포함) ===")
    
    # 1. 문서 로드 (기존 보고서 활용)
    # 스크립트 위치 기준으로 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Practical_Examples -> Ch02.RAG -> Data
    pdf_path = os.path.join(current_dir, "..", "Data", "2024_KB_부동산_보고서_최종.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {pdf_path}")
        return

    print(f"Loading PDF from {pdf_path}...")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"PDF 로드 중 오류 발생 (pypdf가 설치되어 있는지 확인하세요): {e}")
        return

    # 2. 텍스트 분할
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks.")

    # 3. 벡터 저장소 생성 (임시)
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # 4. 체인 구성
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_prompt = (
        "당신은 부동산 전문가입니다. 아래 제공된 Context를 바탕으로 질문에 답변해주세요. "
        "답변은 한국어로 작성하고, 3문장 이내로 요약해주세요. "
        "모르는 내용이면 솔직히 모른다고 답해주세요."
        "\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. 질의 및 실행
    query = "2024년 주택 매매 가격 전망은 어떻습니까?"
    print(f"\n질문: {query}")
    print("답변 생성 중...")
    
    response = rag_chain.invoke({"input": query})
    
    # 6. 결과 및 출처 출력
    print("\n[답변]")
    print(response["answer"])
    
    print("\n[참고 자료 (출처)]")
    # 중복 제거를 위해 set 사용
    seen_pages = set()
    for i, doc in enumerate(response["context"]):
        page = doc.metadata.get("page", "N/A")
        # PDF 페이지는 0부터 시작하므로 +1
        display_page = int(page) + 1 if isinstance(page, int) else page
        
        if display_page not in seen_pages:
            seen_pages.add(display_page)
            print(f"- 페이지: {display_page}")
            # print(f"  내용 일부: {doc.page_content[:50].replace('\n', ' ')}...")

if __name__ == "__main__":
    citation_rag_demo()
