import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# <2024 부동산 보고서 RAG 챗봇>

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

# PDF 파일 불러오기
loader =PyPDFLoader("./Data/2024_KB_부동산_보고서_최종.pdf")
documents = loader.load()

# 텍스트 분할 설정: 청크 크기와 겹침 설정
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 임베딩 생성 및 Chroma 데이터베이스 저장
embedding_function = OpenAIEmbeddings()
persist_directory = "./directory"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory=persist_directory
)

# 검색 및 재정렬

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

template = """당신은 KB 부동산 보고서 전문가 입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요 컨택스트: {context}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",template),
        ("placeholder","{chat_history}"),
        ("human","{question}")
    ]
)

# 템플릿 초기화
model = ChatOpenAI(model_name="gpt-4o",temperature=0)

# 문서 형식 변환 함수정의
def format_doc(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 체인 구성 : 검색한 문서를 프롬프트에 연결하고 모델을 통해 응답 생성
chain = (
    RunnablePassthrough.assign(
        context = lambda x : format_doc(retriever.invoke(x["question"]))
    )
    | prompt
    | model
    | StrOutputParser()
)

# 대화 기록을 유지하기 위한 메모리 설정
chat_history = ChatMessageHistory()
chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# 챗봇 실행 함수 정의

def chat_with_bot():
    session_id = "user_session",
    print("KB 부동산 보고서 챗봇입니다. 질문해 주세요. (종료하려면 'quit' 입력)")
    while True: 
        user_input= input("사용자: ")
        if user_input.lower()=="quit":
            break

        response = chain_with_memory.invoke(
            {"question":user_input},
            {"configurable":{"session_id":session_id}}
        )

        print("챗봇:",response)

# 메인 실행
if __name__ == "__main__":
    chat_with_bot()