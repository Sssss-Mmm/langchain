import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# 0. 환경 변수 로드
load_dotenv()

# 1. 문서 다운로드 및 준비
def download_pdf_if_not_exists(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print(f"{filename} already exists.")

pdf_url = "https://raw.githubusercontent.com/llama-index-tutorial/llama-index-tutorial/main/ch06/ict_usa_2024.pdf"
pdf_filename = "ict_usa_2024.pdf" # 현재 폴더에 저장
download_pdf_if_not_exists(pdf_url, pdf_filename)

# 2. 문서 인덱싱 (Vector Store 생성)
# 실전 예제에서는 매번 생성하지 않고 persists 하는 것이 좋지만, 
# 여기서는 간편한 실행을 위해 메모리/임시 생성 방식을 사용합니다.

print("Indexing document...")
loader = PyMuPDFLoader(pdf_filename)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    collection_name="usa_ict_policy" 
)
retriever = vectorstore.as_retriever()
print("Indexing complete.")

# 3. Retriever를 도구로 변환
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="usa_ict_search",
    description="Search for information about USA ICT policies and market trends. Use this tool for any questions regarding USA ICT."
)

tools = [retriever_tool]

# 4. 에이전트 생성
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 5. 실행
print("\n=== ReAct Agent with Retrieval Tool ===")
question = "미국의 ICT 정책 중, 디지털 인프라와 관련된 주요 내용을 요약해줘."
print(f"질문: {question}")

result = agent_executor.invoke({"input": question})
print(f"\n최종 답변: {result['output']}")
