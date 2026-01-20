import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from kiwipiepy import Kiwi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

"""
이 실습 코드는 키워드 기반 검색(BM25)과 의미 기반 벡터 검색(FAISS)을 결합한 하이브리드 검색(Hybrid Retrieval) 시스템을 구현합니다.

주요 단계:
1. 문서 로드 및 전처리: PyPDFLoader와 RecursiveCharacterTextSplitter를 사용합니다.
2. 키워드 검색: Kiwi 형태소 분석기를 활용하여 BM25Retriever의 한국어 검색 성능을 최적화합니다.
3. 벡터 검색: OpenAIEmbeddings와 FAISS를 사용하여 문맥적 유사성 기반 검색을 수행합니다.
4. 검색 통합: EnsembleRetriever를 사용하여 두 검색 방식의 결과를 가중치 기반으로 통합합니다.
5. 질의응답: RetrievalQA 체인을 통해 검색된 문서를 바탕으로 답변을 생성합니다.
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

# 2. 문서 분할
doc_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = doc_splitter.split_documents(docs)

# 3. BM25 검색기 (키워드 검색) 설정
kiwi = Kiwi()
def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

bm25_retriever = BM25Retriever.from_documents(split_docs, preprocess_func=kiwi_tokenize)
bm25_retriever.k = 3

# 4. Dense 검색기 (의미 검색) 설정
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
faiss_store = FAISS.from_documents(split_docs, embedding)
faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 3})

# 5. 앙상블 검색기 설정 (Hybrid Retrieval)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5] # BM25와 Dense 검색의 가중치를 5:5로 설정
)

# 6. QA 체인 생성
llm = ChatOpenAI(temperature=0, model="gpt-4o")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    return_source_documents=True
)

# 7. 실행 및 결과 출력
query = "이 회사가 발행한 주식의 총 발행량이 어느정도야?"
print(f"\n질문: {query}")
result = qa_chain.invoke(query)

print(f"답변: {result['result']}")
print("\n[참고 문서]")
for i, doc in enumerate(result['source_documents']):
    print(f"{i+1}. {doc.page_content[:100]}...")
