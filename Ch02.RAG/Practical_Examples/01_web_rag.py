import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def web_rag_chat(url, question):
    print(f"Loading content from: {url}")
    
    # 1. 웹 문서 로드
    # bs4를 사용하여 불필요한 헤더, 푸터 등을 제외하고 본문 내용만 추출하려고 시도합니다.
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "entry-content", "content", "article", "main")
            )
        ),
    )
    docs = loader.load()
    
    # 만약 위 클래스들로 내용을 못 찾으면 전체를 로드 (Fallback)
    if not docs or len(docs[0].page_content) < 100:
        print("Specific content tags not found, loading entire page body...")
        loader = WebBaseLoader(web_paths=(url,))
        docs = loader.load()

    print(f"Loaded document length: {len(docs[0].page_content)} characters")

    # 2. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks")

    # 3. 벡터 저장소 생성 및 임베딩
    # 메모리 내에서만 실행하므로 persist_directory를 지정하지 않거나 임시 디렉토리 사용 가능하지만, 
    # 여기서는 매번 새로 생성하는 방식으로 진행합니다.
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # 4. 프롬프트 및 RAG 체인 구성
    # LangChain Hub에서 검증된 RAG 프롬프트를 가져오거나 직접 정의합니다.
    # 여기서는 직접 정의하여 한국어로 답변하도록 유도합니다.
    from langchain_core.prompts import ChatPromptTemplate
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Answer in Korean.

    Question: {question} 

    Context: {context} 

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. 실행
    print(f"\nQuestion: {question}")
    print("Generating answer...")
    return rag_chain.invoke(question)

if __name__ == "__main__":
    # 예제 URL: LangChain 공식 문서 소개 페이지 (임의로 변경 가능)
    target_url = "https://python.langchain.com/docs/introduction/"
    target_question = "LangChain의 주요 특징은 무엇인가요?"
    
    try:
        answer = web_rag_chat(target_url, target_question)
        print(f"\nAnswer:\n{answer}")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("\nNote: This script requires 'beautifulsoup4' library. Install it with: pip install beautifulsoup4")
