import os
from dotenv import load_dotenv

load_dotenv()

os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import TextLoader

# 문서 로더 설정
loaders = [
    TextLoader("./Data/How_to_invest_money.txt")
]

docs = []

for loader in loaders :
    docs.extend(loader.load())

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 부모 문서 생성을 위한 텍스트 분할기
parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000)
# 자식 문서 생성을 위한 텍스트 분할기 (부모보다 작은 크기로 설정)
child_splitter = RecursiveCharacterTextSplitter(chunk_size = 200)

# 자식 문서 인덱싱을 위한 벡터 저장소
vectorstore = Chroma(
    collection_name="split_parents",embedding_function=OpenAIEmbeddings()
)

# 부모 문서 저장을 위한 저장
store = InMemoryStore()

# ParentDocumentRetriever 설정
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 문서 추가
retriever.add_documents(docs)

# 부모 문서 수 확인
print(f"Number of parent documents:{len(list(store.yield_keys()))}")

# 질문 정의
query = "What are the types of investments?"

# 연관문서 수집
retrieved_docs = retriever.get_relevant_documents(query)

# 첫 번째 연관문서 출력
print(f"Parent Document: {retrieved_docs[0].page_content}")

# 자식 문서 검색
sub_docs = vectorstore.similarity_search(query)
print(f"Child Document: {sub_docs[0].page_content}")