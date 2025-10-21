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

kiwi_tokenizer = Kiwi()

def kiwi_tokenize(text):
    return [token.form for token in kiwi_tokenizer.tokenize(text)]

bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=kiwi_tokenize)
bm25_retriever.k = 4


from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model = "text-embedding-3-large")

from langchain_community.vectorstores import FAISS

faiss_store = FAISS.from_documents(docs,embedding)
faiss_store.save_local("./DB")

persist_directory = "./DB"
vectordb = FAISS.load_local(persist_directory,embeddings=embedding, allow_dangerous_deserialization=True)

faiss_retriever = vectordb.as_retriever(search_kwargs = {"k":4})

from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever,faiss_retriever], weights=[0.5,0.5]
)

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm  = ChatOpenAI(temperature=0.2, model="gpt-4o"),
    chain_type = "stuff",
    retriever = ensemble_retriever,
    return_source_documents = False
)

print(qa_chain.invoke("이 회사가 발행한 주식의 총 발행량이 어느정도야?"))