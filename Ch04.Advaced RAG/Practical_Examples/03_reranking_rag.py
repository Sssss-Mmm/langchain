import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain.docstore.document import Document
from pydantic import Field
from typing import List, Any

"""
이 코드는 Cross-Encoder를 활용한 재순위화(Reranking) RAG 시스템을 구현한 예제입니다.

주요 단계:
1. PDF 문서 로드 및 텍스트 분할: 투자설명서 PDF를 읽어 작은 청크 단위로 나눕니다.
2. 벡터 저장소 구축: OpenAI 임베딩을 사용하여 분할된 문서를 FAISS 벡터 DB에 저장합니다.
3. Custom Reranking Retriever 정의: 
   - 1차 검색: 벡터 유사도 기반으로 후보 문서들을 넓게 검색합니다(k=10).
   - 2차 재순위화: Cross-Encoder 모델을 사용하여 질문과 검색된 문서 간의 관련성 점수를 정밀하게 계산합니다.
   - 최종 선택: 점수가 높은 상위 문서들만 선별하여(rerank_top_k=3) 답변 생성에 사용함으로써 검색 정확도를 향상시킵니다.

"""

# 0. 환경 변수 로드
load_dotenv()

def find_data_file(filename):
    """
    Data 폴더에서 파일을 찾기 위한 견고한 경로 탐색 함수
    """
    possible_paths = [
        os.path.join(os.getcwd(), 'Data', filename),
        os.path.join(os.getcwd(), '../Data', filename),
        os.path.join(os.path.dirname(__file__), '../Data', filename),
        os.path.join(os.path.dirname(__file__), 'Data', filename),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"'{filename}' 파일을 찾을 수 없습니다.")

# 1. 문서 로드
try:
    file_path = find_data_file("투자설명서.pdf")
    print(f"Loading file from: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
except FileNotFoundError as e:
    print(e)
    exit()

# 2. 문서 분할 및 벡터 DB 저장
doc_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = doc_splitter.split_documents(docs)

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
faiss_store = FAISS.from_documents(split_docs, embedding)

# 3. Custom Reranking Retriever 정의
class RetrieverWithCrossEncoder(BaseRetriever):
    vectorstore: Any = Field(description="초기 검색을 위한 벡터 저장소")
    crossencoder: Any = Field(description="재순위화를 위한 크로스 인코더 모델")
    k: int = Field(default=10, description="초기에 검색할 문서 수 (넓게 검색)")
    rerank_top_k: int = Field(default=3, description="재순위화 후 최종적으로 반환할 문서 수")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        # 1차 검색 (Vector Search)
        init_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        # Cross-Encoder 입력을 위한 (Query, Doc) 쌍 생성
        pairs = [[query, doc.page_content] for doc in init_docs]
        
        # 점수 계산
        scores = self.crossencoder.predict(pairs)
        
        # 점수와 문서를 묶어서 정렬
        scored_docs = sorted(zip(init_docs, scores), key=lambda x: x[1], reverse=True)
        
        # 상위 K개 반환
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

# 4. Reranker 초기화 (HuggingFace Model)
# ms-marco-MiniLM-L-12-v2 모델은 비교적 가볍고 성능이 준수함
cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

reranking_retriever = RetrieverWithCrossEncoder(
    vectorstore=faiss_store,
    crossencoder=cross_encoder_model,
    k=10,            # 1차로 10개 검색
    rerank_top_k=3   # 그 중 3개만 최종 선택
)

# 5. QA 체인 연결
llm = ChatOpenAI(temperature=0, model="gpt-4o")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=reranking_retriever,
    return_source_documents=True
)

# 6. 실행 및 결과 확인
query = "이 회사의 2022년 영업손실이 정확히 얼마야?"
print(f"\n질문: {query}")
result = qa_chain.invoke(query)

print(f"답변: {result['result']}")
print("\n[Reranking 후 선택된 문서 Top 3]")
for i, doc in enumerate(result['source_documents']):
    print(f"{i+1}. {doc.page_content[:100]}...")
