import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

"""
이 스크립트는 LangChain의 MultiQueryRetriever를 사용하여 '질문 확장(Query Expansion)' 기법을 구현하는 예제입니다.

주요 기능:
1. 사용자의 원본 질문을 LLM(GPT-4o)을 통해 다양한 관점의 여러 질문으로 확장합니다.
2. 확장된 모든 질문을 사용하여 벡터 저장소(Chroma)에서 관련 문서들을 검색합니다.
3. 중복된 문서를 제거하고 고유한 문서 세트를 확보하여 검색 누락을 방지하고 정확도를 높입니다.
4. 최종적으로 확장된 검색 결과를 바탕으로 RetrievalQA 체인을 통해 답변을 생성합니다.
"""
# 0. 환경 변수 및 로깅 설정
load_dotenv()
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

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
    file_path = find_data_file("How_to_invest_money.txt")
    print(f"Loading file from: {file_path}")
    loader = TextLoader(file_path)
    docs = loader.load()
except FileNotFoundError as e:
    print(e)
    exit()

# 2. 문서 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

# 3. 벡터 저장소 생성 (Chroma)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)

# 4. MultiQueryRetriever 설정
# LLM을 사용하여 사용자의 질문을 다양한 관점의 3가지 질문으로 변환하여 검색합니다.
llm = ChatOpenAI(model="gpt-4o", temperature=0)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 5. 실행 및 결과
question = "주식 투자를 처음 시작하려면 어떻게 해야되나요?"
print(f"\n원본 질문: {question}\n")
print("--- 생성된 쿼리 (로그 확인) ---")

# 검색 실행 (생성된 쿼리는 로깅으로 출력됨)
unique_docs = retriever.invoke(question)
print(f"\n검색된 독창적인 문서 개수: {len(unique_docs)}")

# QA 체인 연결
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain.invoke(question)
print(f"\n최종 답변: {result['result']}")
