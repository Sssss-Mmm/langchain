import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

#<RecursiveCharacterTextSplitter>
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# PyPDFLoader를 사용하여 PDF 파일 로드
loader = PyPDFLoader("./Data/2024_KB_부동산_보고서_최종.pdf")
pages = loader.load()

# PDF 파일의 모든 페이지에서 텍스트를 추출하여 총 글자 수 계산
print('총 글자 수:',len(''.join([i.page_content for i in pages])))

# RecursiveCharacterTextSplitter 초기화
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# 문서 분할
texts = text_splitter.split_documents(pages)
print('분할된 청크의 수:',len(texts))

print(texts[1])
print(texts[1].page_content)
print(texts[2].page_content)
print('1번 청크의 길이:',len(texts[1].page_content))
print('2번 청크의 길이:',len(texts[2].page_content))

# <SementicChunker>
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# SementicChunker 초기화
text_splitter = SemanticChunker(embeddings=OpenAIEmbeddings())

# 텍스트를 의미 단위로 분할
chunks= text_splitter.split_documents(pages)

# 분할된 청크 수
print('분할된 청크의 수:',len(chunks))
print(chunks[3])
print(chunks[4])
print(chunks[5])

# <백분위수 방식의 SementicChunker>
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)

# <표준 편차 방식의 SementicChunker>
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3,
)

# <사분위수 방식의 SemanticChunker>
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="interquartile",
    breakpoint_threshold_amount=1.5,
)
chunks = text_splitter.split_documents(pages)
print("분할된 청크의 수",len(chunks))