import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = ("./Data/투자설명서.pdf")
loader = PyPDFLoader(file_path)

doc_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)

docs = loader.load_and_split(doc_splitter)

from langchain_community.retrievers import BM25Retriever
from kiwipiepy import Kiwi

kiwi_tokenizer = Kiwi()

def kiwi_tokenize(text):
    return [token.form for token in kiwi_tokenizer.tokenize(text)]

bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=kiwi_tokenize)
bm25_retriever.k = 2

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# 관련있는 문서를 수집 후 , Chatgpt로 최종 답변까지 수행하는 체인을 생성
qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(temperature=0.2, model="gpt-4o"),
    chain_type = "stuff",
    retriever = bm25_retriever,
    return_source_documents = True # 답변에 사용된 source document도 보여주도록 설정
)

print(qa_chain.invoke("이 회사가 발행한 주식의 총 발행량이 어느정도야?"))