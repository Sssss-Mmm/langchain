import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# <Chroma>
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# PDF 파일 로드
loader = PyPDFLoader("./Data/2024_KB_부동산_보고서_최종.pdf")
pages = loader.load()

print("청크의 수: ",len(pages))

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)

print("분할된 청크의 수:",len(splits))

# 각 청크의 길이(문자 수)를 저장한 리스트 생성
chunk_lengths =[len(chunk.page_content) for chunk in splits]
max_length = max(chunk_lengths)
min_length = min(chunk_lengths)
avg_length = sum(chunk_lengths)/len(chunk_lengths)

print('청크 최대 길이: ',max_length)
print('청크 최소 길이: ',min_length)
print('청크 평균 길이: ',avg_length)

# 임베딩 모델 초기화
embedding_function = OpenAIEmbeddings()

# Chroma DB 생성 및 데이터 저장

persist_directory = "./directory"
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding_function,
    persist_directory=persist_directory
)

# print("문서의 수: ",vectordb._collection.count())

# similarity_search 메서드 사용

question = "수도권 주택 매매 전망"
top_three_docs = vectordb.similarity_search(question,k=2)

# for i ,doc in enumerate(top_three_docs,1):
#     print(f"문서{i}:")
#     print(f"내용: {doc.page_content[:150]}...")
#     print(f"메타데이터: {doc.metadata}")
#     print('--'*20)


# similarity_search_with_relevance_scores 메서드 사용

top_three_docs = vectordb.similarity_search_with_relevance_scores(question,k=2)

# for i ,doc in enumerate(top_three_docs,1):
#     print(f"문서{i}:")
#     print(f"유사 점수 {doc[1]}:")
#     print(f"내용: {doc[0].page_content[:150]}...")
#     print(f"메타데이터: {doc[0].metadata}")
#     print('--'*20)

# <FAISS>

from langchain_community.vectorstores import FAISS

# 파이스 db 생성

faiss_db = FAISS.from_documents(documents=splits,embedding=embedding_function)

print('문서의 수:',faiss_db.index.ntotal)

# 파이스 db 저장하기

faiss_directory = './directory'
faiss_db.save_local(faiss_directory)

# 파이스 db 불러오기

new_db_faiss = FAISS.load_local(faiss_directory,OpenAIEmbeddings(),allow_dangerous_deserialization=True)

# 검색할 질문 정의
question="수도권 주택 매매 전망"

# similarity_search 메서드 사용
docs = new_db_faiss.similarity_search(question)

for i, doc in enumerate(docs,1):
    print(f"문서{i}:")
    print(f"내용: {doc.page_content[:150]}")
    print(f"메타데이터: {doc.metadata}")
    print("--"*20)