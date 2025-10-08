import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# <WebBaseLoader>
# 사용자 에이전트 설정
os.environ["USER_AGENT"]="MyApp/1.0 (Custom LangChain Application)"

from langchain_community.document_loaders import WebBaseLoader

# 단일 URL 초기화
loader = WebBaseLoader("https://docs.smith.langchain.com/")
# 다중 URL 초기화
load_multiple_pages = WebBaseLoader(
    ["https://python.langchain.com/docs/introduction/","https://langchain-ai.github.io/langgraph/"]
)

# 단일 문서 로드
single_doc = loader.load()

#문서의 메타데이터 확인
print(single_doc[0].metadata)

# 다중 문서 로드
docs = load_multiple_pages.load()
print(docs[0].page_content)


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
# <PyPDFLoader>
loader = PyPDFLoader("./Data/2024_KB_부동산_보고서_최종.pdf")
pages = loader.load_and_split()

print('청크의 수:',len(pages))
print(pages[10])
print(pages[10].metadata)
print(pages[10].page_content)

# <PyMuPDFLoader>

loader = PyMuPDFLoader("./Data/2024_KB_부동산_보고서_최종.pdf")
pages = loader.load_and_split()
print('청크의 수:',len(pages))
print(pages[10])
print(pages[10].page_content)

# <CSVLoader>
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredCSVLoader

# CSV 파일 로더 초기화
loader = CSVLoader("./Data/서울시_부동산_실거래가_정보.csv")

# CSV 파일 로드 및 행 분할
documents = loader.load()

print('청크의 수:',len(documents))
print(documents[5])

# <UnstructuredCSVLoader>

# CSV 파일 로더 초기화
loader = UnstructuredCSVLoader("./Data/서울시_부동산_실거래가_정보.csv", mode='elements')

# CSV 파일 로드
documents = loader.load()
print('청크의 수:',len(documents))

print(str(documents[0].metadata)[:500])
print(str(documents[0].page_content)[:500])
