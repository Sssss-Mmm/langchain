import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# <2024 부동산 보고서 RAG 챗봇>

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

# PDF 처리 함수
def process_pdf():
    loader = PyPDFLoader("./Data/2024_KB_부동산_보고서_최종.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# 벡터 스토어 초기화
def init_vectorstore():
    # 캐시: 세션에 이미 있으면 재생성하지 않음
    if "vectorstore" in st.session_state:
        return st.session_state["vectorstore"]
    chunks = process_pdf()
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    vs = Chroma.from_documents(chunks, embeddings)
    st.session_state["vectorstore"] = vs
    return vs

# 체인 초기화
def init_chain():
    if "chain" in st.session_state:
        return st.session_state["chain"]

    vectorstore = init_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해 주세요.

컨텍스트:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # 최신 버전은 보통 model 파라미터 사용
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    chain = RunnableWithMessageHistory(
        base_chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    st.session_state["chain"] = chain
    return chain

# Streamlit UI
def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    # 세션 상태 초기화 (없을 때만)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if user_prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # 체인 가져오기/초기화
        chain = init_chain()

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": user_prompt},
                    {"configurable": {"session_id": "streamlit_session"}}
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

# 터널링 및 실행
from pyngrok import ngrok

public_url = ngrok.connect(8501)  # Streamlit 기본 포트
print("앱 접속 URL:", public_url)

