import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = ("./Data/투자설명서.pdf")

loader = PyPDFLoader(file_path)

doc_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 100)

docs = loader.load_and_split(doc_splitter)

from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model = "text-embedding-3-large")

from langchain_community.vectorstores import FAISS

faiss_store = FAISS.from_documents(docs, embedding)

persist_directory = "./DB"
faiss_store.save_local(persist_directory)

vectordb = faiss_store.load_local(persist_directory,embeddings=embedding,allow_dangerous_deserialization=True)

from pydantic import Field
from langchain.docstore.document import Document
from typing import List,Dict,Any,Tuple
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain.chains import RetrievalQA

crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# 크로스 인코더 기반 리트리버 정의
class Retriever_with_cross_encoder(BaseRetriever):
    vectorstore : Any = Field(description="초기 검색을 위한 벡터 저장소")
    crossencoder : Any = Field(description="재순위화를 위한 크로스 인코더 모델")
    k : int = Field(default=5,description="초기에 검색할 문서 수")
    rerank_top_k : int = Field(default=2 ,description="재순위화 후 최종적으로 반환할 문서 수")
    
    # Pydantic 설정
    class Config :
        arbitrary_types_allowed = True
    
    # 크로스 인코더를 사용한 문서 검색 메서드 재정의
    def get_relevant_documents(self, query:str) -> List[Document]:
        init_docs = self.vectorstore.similarity_search(query, k=self.k)

        pairs =[[query, doc.page_content] for doc in init_docs]

        scores = self.crossencoder.predict(pairs)

        scored_docs = sorted(zip(init_docs, scores),key=lambda x : x[1], reverse=True)

        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

# 크로스 인코더 기반 리트리버 생성
cross_encode_retriever = Retriever_with_cross_encoder(
    vectorstore= vectordb,
    crossencoder=crossencoder,
    k=4,
    rerank_top_k=2
)

llm =ChatOpenAI(temperature=0.2, model_name="gpt-4o")

# 관련있는 문서를 수집 후 , Chatgpt로 최종 답변까지 수행하는 체인을 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = cross_encode_retriever,
    return_source_documents = True
)

query = "이 회사의 2022년 영업손실이 정확히 얼마야?"
result = qa_chain({"query":query})

print(f"\n질문: {query}")
print(f"답변: {result['result']}")
print("\n답변 근거 문서:")

# 답변에 사용된 문서 출력
for i, doc in enumerate(result["source_documents"]):
    print(f"\nDocument {i+1}:")
    print(doc.page_content)