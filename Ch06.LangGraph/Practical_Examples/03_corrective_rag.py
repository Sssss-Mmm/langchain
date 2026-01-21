import os
from dotenv import load_dotenv
from typing import List, Literal
from typing_extensions import TypedDict
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START

# 0. 환경 변수 로드
load_dotenv()
os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; MyLangchainBot/1.0; +https://example.com)"

# 1. RAG용 문서 로드 및 인덱싱 (Knowledge Base 구축)
urls = [
    "https://google.github.io/styleguide/pyguide.html",
    "https://google.github.io/styleguide/javaguide.html",
    "https://google.github.io/styleguide/jsguide.html",
]

print("--- 문서 로딩 및 인덱싱 중... ---")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Chroma DB (메모리 모드)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="crag-chroma",
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
print("--- 인덱싱 완료 ---")


# 2. 필요한 LLM 및 프롬프트 정의

# (1) 문서 평가용 LLM (Grader)
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="문서와 질문의 연관성 여부. '예' 또는 '아니오'")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_grader = """당신은 사용자의 질문에 대해 검색된 문서의 관련성을 평가하는 전문가입니다.
문서에 질문과 관련된 키워드나 의미가 담겨 있으면, 해당 문서를 ‘관련 있음’으로 평가하세요.
문서가 질문과 관련이 있는지 여부를 ‘예’ 또는 ‘아니오’로 표시해 주세요."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grader),
        ("human", "검색된 문서: \n\n {document} \n\n 사용자 질문: {question}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader

# (2) 답변 생성용 LLM (Generator)
system_generator = """당신은 질문에 답변하는 업무를 돕는 도우미입니다.
제공된 문맥을 바탕으로 질문에 답변하세요. 만약 답을 모르면 모른다고 말하세요.
세 문장을 넘지 않도록 답변을 간결하게 작성하세요."""

prompt_generator = ChatPromptTemplate.from_messages(
    [
        ("system", system_generator),
        ("human", "질문: {question}, \n 문맥: {context} \n 답변:")
    ]
)
rag_chain = prompt_generator | llm | StrOutputParser()

# (3) 질문 재작성용 LLM (Rewriter)
system_rewriter = """당신은 입력된 질문을 변형하여 웹 검색에 최적화된 형태로 만드는 질문생성기입니다.
입력된 질문을 보고 그 이면에 있는 의미나 의도를 파악해서 더 나은 질문을 만들어주세요."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewriter),
        ("human", "질문: \n\n {question} \n 더 나은 질문으로 바꿔주세요")
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

# (4) 웹 검색 도구
web_search_tool = TavilySearchResults(k=3)


# 3. LangGraph 상태 및 노드 정의

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str # "예" or "아니오"
    documents: List[str]

def retrieve(state):
    print("\n--- [Retrieve] 문서 검색 ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    print("--- [Generate] 답변 생성 ---")
    question = state["question"]
    documents = state["documents"]
    
    # 문서 객체를 문자열로 변환
    docs_txt = "\n\n".join(doc.page_content for doc in documents)
    generation = rag_chain.invoke({"context": docs_txt, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    print("--- [Grade] 문서 평가 ---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search = "아니오"
    
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "예":
            print(f"  - 관련 있음")
            filtered_docs.append(d)
        else:
            print(f"  - 관련 없음 (제거됨)")
            web_search = "예" # 하나라도 관련 없는게 나오면(혹은 부족하면) 웹 검색 트리거 (전략에 따라 다름)
            
    # 여기서는 '관련 없음'이 하나라도 있으면 웹 검색을 하는 전략 사용 (엄격한 전략)
    # 또는 filtered_docs가 비어있으면 웹 검색을 하는 전략도 가능
    if not filtered_docs:
        web_search = "예"
        
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    print("--- [Transform] 질문 재작성 ---")
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    print(f"  - 원본: {question}")
    print(f"  - 수정: {better_question}")
    return {"question": better_question}

def web_search_node(state):
    print("--- [Web Search] 웹 검색 ---")
    question = state["question"]
    documents = state["documents"]
    
    print(f"  - 검색어: {question}")
    docs = web_search_tool.invoke({"query": question})
    
    web_results = "\n".join([d["content"] for d in docs])
    # 검색 결과를 Document 형태로 변환하여 기존 문서 리스트에 추가
    from langchain_core.documents import Document
    web_results_doc = Document(page_content=web_results)
    documents.append(web_results_doc)
    
    return {"documents": documents, "question": question}

def decide_to_generate(state):
    print("--- [Decide] 경로 결정 ---")
    web_search = state["web_search"]
    
    if web_search == "예":
        print("  -> 질문 변환 및 웹 검색 필요")
        return "transform_query"
    else:
        print("  -> 바로 답변 생성")
        return "generate"

# 4. 그래프 조립
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# 5. 실행
if __name__ == "__main__":
    print("=== Corrective RAG (CRAG) 실행 ===")
    
    # Case 1: 문서에 정보가 있는 경우
    print("\n\n[Test 1] 파이썬 코드 스타일 가이드 관련 질문")
    inputs = {"question": "파이썬 코드 작성시 들여쓰기 규칙은?"}
    for output in app.stream(inputs):
        pass # 중간 과정 출력은 각 노드 함수에서 함
    
    print(f"\n[최종 답변] {output['generate']['generation']}")

    # Case 2: 문서에 정보가 없어서 웹 검색이 필요한 경우
    print("\n\n[Test 2] 문서에 없는 최신 정보 질문 (예: 한국 날씨)")
    inputs = {"question": "오늘 서울 날씨 어때?"} 
    for output in app.stream(inputs):
        pass
        
    print(f"\n[최종 답변] {output['generate']['generation']}")
