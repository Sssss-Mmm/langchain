import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 문서 로더 설정
loaders = [TextLoader("./Data/How_to_invest_money.txt")]

docs = []
for loader in loaders :
    docs.extend(loader.load())

# 문서 생성을 위한 텍스트 분할기 정의
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

# 문서 분할
split_docs = recursive_splitter.split_documents(docs)

# OpenAiEmbeddings 인스턴스 설정
embeddings = OpenAIEmbeddings()

# Chroma vectorstore 생성
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)

# Chroma vectorstore 기반 리트리버 생성
retriever = vectorstore.as_retriever()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

# 1. 가상 문서 생성 체인
def create_virtual_doc_chain():
    system = " 당신은 고도로 숙련된 AI입니다."
    user = """
    주어진 질물 '{query}'에 대해 직접적으로 답변하는 가상의 문서를 생성하세요.
    문서의 크기는 {chunk_size} 글자 언저리여야 합니다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",system),
        ("human",user)
    ])
    llm = ChatOpenAI(model = "gpt-4o",temperature=0.2)
    return prompt | llm | StrOutputParser()

# 2. 문서 검색 체인
def create_retrieval_chain():
    return RunnableLambda(lambda x : retriever.get_relevant_documents(x['virtual_doc']))

# 유틸리티 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. 최종 응답 생성 체인
def create_final_response_chain():
    final_prompt = ChatPromptTemplate.from_template("""
    다음 정보와 질문을 바탕으로 답변해주세요:
    
    컨텍스트 : {context}
    
    질문 : {question}
    
    답변 :
    """)
    final_llm = ChatOpenAI(model="gpt-4o",temperature=0.2)
    return final_prompt | final_llm

def print_input_output(input_data, output_data, step_name):
    print(f"\n--- {step_name} ---")
    print(f"input: {input_data} ")
    print(f"output: {output_data} ")
    print("-" * 50)

def create_pipeline_with_logging():
    virtual_doc_chain = create_virtual_doc_chain()
    retrieval_chain = create_retrieval_chain()
    final_response_chain = create_final_response_chain()

    # 가상 문서 생성 단계
    def virtual_doc_step(x):
        result = {"virtual_doc": virtual_doc_chain.invoke({
            "query": x["question"],
            "chunk_size": 200
        })}
        print_input_output(x,result,"Virtual Doc Generation")
        return{**x,**result}
    
    # 문서 검색 단계
    def retrieval_step(x):
        result ={"retrieved_docs":retrieval_chain.invoke(x)}
        print_input_output(x,result,"Document Retrieval")
        return{**x,**result}
    
    # 컨텍스트 포맷팅 단계 
    def context_formatting_step(x):
        result ={"context":format_docs(x["retrieved_docs"])}
        print_input_output(x,result,"Context Formatting")
        return{**x,**result}    
    
    # 최종 응답 생성 단계
    def final_response_step(x):
        result = final_response_chain.invoke(x)
        print_input_output(x,result,"Final Response Generation")
        return result
    
    # 전체 파이프라인 구성
    pipeline = (
        RunnableLambda(virtual_doc_step)
        | RunnableLambda(retrieval_step)
        | RunnableLambda(context_formatting_step)
        | RunnableLambda(final_response_step)
    )

    return pipeline

# 파이프라인 객체 생성
pipeline = create_pipeline_with_logging()


# 예시 질문과 답변
question = "주식 시장의 변동성이 높을 때 투자 전략을 무엇인가요?"
response = pipeline.invoke({"question":question})
print(f"최종 답변: {response.content}")