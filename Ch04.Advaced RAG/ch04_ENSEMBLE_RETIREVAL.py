import os
from dotenv import load_dotenv

load_dotenv()

os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = ("./Data/투자설명서.pdf")

loader = PyPDFLoader(file_path)

doc_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)

docs = loader.load_and_split(doc_splitter)

from langchain_community.retrievers import BM25Retriever
from kiwipiepy import Kiwi

# 한국어 토크나이저 설정
kiwi_tokenizer = Kiwi()

# 한국어 토크나이저 함수 정의
def kiwi_tokenize(text):
    return [token.form for token in kiwi_tokenizer.tokenize(text)]

bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=kiwi_tokenize)
bm25_retriever.k = 4


from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model = "text-embedding-3-large")

from langchain_community.vectorstores import FAISS

# FAISS DB 생성 후 저장
faiss_store = FAISS.from_documents(docs,embedding)
faiss_store.save_local("./DB")

persist_directory = "./DB"
vectordb = FAISS.load_local(persist_directory,embeddings=embedding, allow_dangerous_deserialization=True)

faiss_retriever = vectordb.as_retriever(search_kwargs = {"k":4})

from langchain.retrievers import EnsembleRetriever

# EnsembleRetriever 생성
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever,faiss_retriever], weights=[0.5,0.5]
)

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# 관련있는 문서를 수집 후 , Chatgpt로 최종 답변까지 수행하는 체인을 생성
qa_chain = RetrievalQA.from_chain_type(
    llm  = ChatOpenAI(temperature=0.2, model="gpt-4o"),
    chain_type = "stuff",
    retriever = ensemble_retriever,
    return_source_documents = False
)

print(qa_chain.invoke("이 회사가 발행한 주식의 총 발행량이 어느정도야?"))