import os
from dotenv import load_dotenv

load_dotenv()

os.getenv("OPENAI_API_KEY")

# 쿼리를 위한 로그 설정
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 문서 로더 설정
loaders = [TextLoader("./Data/How_to_invest_money.txt")]

docs = []
for loader in loaders :
    docs.extend(loader.load())

# 문서 생성을 위한 텍스트 분할기 정의
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap =200)

# 문서 분할
split_docs = recursive_splitter.split_documents(docs)

# OpenAiEmbeddings 인스턴스 생성
embeddings = OpenAIEmbeddings()

# Chroma vectorstore 생성
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)

from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# LLM 모델 설정 
llm = ChatOpenAI(model = "gpt-4o", temperature=0.2)

# MultiQueryRetriever 실행
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), # 기본 검색기 (벡터 데이터베이스)
    llm = llm
)

# 샘플 질문
question = "주식 투자를 처음 시작하려면 어떻게 해야되나요?"

# 결과 검색
unique_docs = retriever.invoke(question)
print(f"\n결과: {len(unique_docs)}개의 문서가 검색되었습니다.")

from langchain.chains import RetrievalQA

# RetrievalQA 체인 설정
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    retriever= retriever,
    return_source_documents = True
)
result = qa_chain.invoke({"query":question})

print("답변:",result["result"])
print("\n사용된 문서:")
for doc in result["source_documents"]:
    print(doc.page_content)