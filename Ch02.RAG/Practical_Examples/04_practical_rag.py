from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

def practical_rag_bot():
    print("=== Practical RAG Chatbot (Full Pipeline) ===")
    
    # 1. 문서 로드 (PDF)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "..", "Data", "2024_KB_부동산_보고서_최종.pdf")
    
    if not os.path.exists(pdf_path):
        print("PDF file not found.")
        return

    print("1. Loading Document...")
    loader = PyPDFLoader(pdf_path)
    # 시간 관계상 앞부분 20페이지만 로드
    documents = loader.load()[:20] 

    # 2. 텍스트 분할
    print("2. Splitting Text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 3. 벡터 저장소
    print("3. Indexing to VectorDB...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. RAG 체인 구성
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_prompt = (
        "You are a helpful assistant. Use the given context to answer the question. "
        "If the context doesn't have the answer, say you don't know. "
        "Answer in Korean. "
        "\n\nContext: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 5. 질문 답변 루프
    print("\n[Chatbot Started] 'quit' to exit.")
    while True:
        query = input("\n질문(Question): ")
        if query.lower() in ["quit", "exit"]:
            break
            
        print("Thinking...")
        result = rag_chain.invoke({"input": query})
        
        print(f"\n답변(Answer): {result['answer']}")
        
        # 출처 표시
        print("\n[Sources]")
        seen = set()
        for doc in result["context"]:
            page = doc.metadata.get("page", 0) + 1
            if page not in seen:
                print(f"- Page {page}")
                seen.add(page)

if __name__ == "__main__":
    practical_rag_bot()
