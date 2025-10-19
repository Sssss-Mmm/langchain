import os
from dotenv import load_dotenv

load_dotenv()

os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = ("./Data/투자설명서.pdf")

loader = PyPDFLoader(file_path)

doc_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap = 20)

docs = loader.load_and_split(doc_splitter)

from langchain_openai import OpenAIEmbeddings

# OpenAI의 임베딩 모델 사용
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

from langchain_community.vectorstores import FAISS

# FAISS DB 생성 후 저장
faiss_store = FAISS.from_documents(docs, embedding)
faiss_store.save_local("./DB")

# 저장된 DB 경로 지정 후 ,DB 로드
persist_directory = "./DB"
vectordb = FAISS.load_local(persist_directory,embeddings=embedding,allow_dangerous_deserialization=True)

# FAISS 리트리버 생성
faiss_retriever = vectordb.as_retriever(search_kwargs = {"k":2})

from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(temperature=0.2,model="gpt-4o"),
    chain_type = "stuff",
    retriever = faiss_retriever,
    return_source_documents=True # 답변에 사용된 source document도 보여주도록 설정
)

print(qa_chain.invoke("이 회사가 발행한 주식의 총 발행량이 어느정도야?"))