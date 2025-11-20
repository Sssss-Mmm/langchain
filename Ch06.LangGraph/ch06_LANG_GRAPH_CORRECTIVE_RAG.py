from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document

os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; MyLangchainBot/1.0; +https://example.com)"

# 크롤할 블로그의 url을 정의합니다.
urls = [
    "https://google.github.io/styleguide/pyguide.html",
    "https://google.github.io/styleguide/javaguide.html",
    "https://google.github.io/styleguide/jsguide.html",
]

# WebBaseLoader를 사용하여 주어진 URL 목록에서 문소를 크롤링합니다.
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 지정한 크기만큼 텍스트를 분할하는 텍스트 분할기를 설정
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250, chunk_overlap = 0
)

# 문서를 분할
doc_splits = text_splitter.split_documents(docs_list)

# Chroma 벡터 저장소에 문서의 분할된 조각을 저장
vectorstore = Chroma.from_documents(
    documents=docs_list,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings()
)

# 벡터 저장소에서 검색을 수행할 수 있는 검색기를 생성
retriever = vectorstore.as_retriever()

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 문서와 질문의 연관성을 평가하기 위한 데이터 모델을 정의합니다.
class GradeDocuments(BaseModel) :
    binary_score : str = Field(
        description= "문서와 질문의 연관성 여부. (예 or 아니오)"
    )

# 연관성 평가를 위한 LLM 을 정의
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# LLM이 사용자의 질문에 대해 문서의 연관성을 평가할 수 있도록 지시하는 프롬프트를 정의합니다.
system = """당신은  사용자의 질문에 대해 검색된 문서의 관련성을 평가하는 전문가입니다.
문서에 질문과 관련된 키워드나 의미가 담겨 있으면, 해당 문서를 ‘관련 있음’으로 평가하세요.
문서가 질문과 관련이 있는지 여부를 ‘예’ 또는 ‘아니오’로 표시해 주세요."""

# 시스템 메시지와 사용자의 질문 및 문서 내용을 포함한 템플릿을 만듭니다.
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "검색된 문서: \n\n {document} \n\n 사용자 질문: {question}"),
    ]
)

# 프롬프트와 구조화된 LLM 평가기를 결합하여 retrieval_grader 객체를 만듭니다.
retrieval_grader = grade_prompt | structured_llm_grader

question = "파이썬 코드 작성 가이드"
# 연관문서 검색
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
# 검색된 문서의 연관성 평가
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

from langchain_core.output_parsers import StrOutputParser

# LLM이 제공된 문맥을 바탕으로 답변할 수 있도록 지시하는 프롬프트를 정의합니다.
system = """당신은 질문에 답변하는 업무를 돕는 도우미입니다.
제공된 문맥을 바탕으로 질문에 답변하세요. 만약 답을 모르면 모른다고 말하세요.
세 문장을 넘지 않도록 답변을 간결하게 작성하세요."""

# 시스템 메세지와 사용자의 질문 및 문서 내용을 포함한 템플릿
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","질문: {question}, \n 문맥: {context} \n 답변:")
    ]
)

# 검색된 문서들을 한 문자열로 합쳐줍니다.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 프롬프트, LLM , 문자열 출력을 결합하여 RAG 체인을 생성
rag_chain = prompt | llm | StrOutputParser()

# 정의된 RAG 체인을 사용하여 질문과 문맥을 기반으로 답변을 생성
generation = rag_chain.invoke(
    {"context": format_docs(docs), "question": question}
)
# 생성된 답변 출력
print(generation)

# LLM이 입력된 질문을 웹검색에 적합한 형태로 변형하도록 지시하는 프롬프트를 정의합니다
system = """당신은 입력된 질문을 변형하여 웹 검색에 최적화된 형태로 만드는 질문생성기입니다.
입력된 질문을 보고 그 이면에 있는 의미나 의도를 파악해주세요."""

# 시스템 메시지와 사용자의 질문을 포함한 템플릿을 만듭니다.
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        (
            "human","질문: \n\n {question} \n 더 나은 질문으로 바꿔주세요"
        )
    ]
)


# 프롬프트, LLM, 문자열 출력을 결합하여 질문 변형 체인을 생성합니다.
question_rewriter = re_write_prompt | llm | StrOutputParser()

question = "C++ 깔끔하게 짜고싶다"
print(question_rewriter.invoke({"question": question}))



web_search_tool = TavilySearchResults(k=3)

# 그래프 상태 정의
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents : List[str]

# rag 체인을 사용하여 답변을 생성하는 노드
def retrieve(state):
    """
    문서를 검색합니다.
    
    Args:
        state (dict): 현재 그래프의 상태
        
    Returns:
        state (dict): 검색된 문서를 포함한 새로운 상태
    """
    print("---검색---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents":documents, "question":question}

# rag 체인을 사용하여 답변을 생성하는 노드
def generate(state):
    """
    답변을 생성합니다
    
    Args:
        state (dict): 현재 그래프의 상태
        
    Returns:
        state (dict): LLM이 생성한 답변을 포함한 새로운 상태
    """
    print("---생성---")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
# 문서의 연관성을 평가하는 노드
def grade_documents(state):
    """
    검색된 문서가 질문과 연관이 있는지 평가합니다.
    
    Args:
        state (dict): 현재 그래프의 상태
    
    Returns:
        state (dict): 연관이 잇다고 판단된 문서가 업데이트 된 상태"""
    
    print("---문서와 질문의 연관성 평가---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "아니오"
    for d in documents :
        score = retrieval_grader.invoke(
            {"question":question, "document":d.page_content}
        )
        grade = score.binary_score
        if grade == "예":
            print("---평가: 연관 문서---")
            filtered_docs.append(d)
        else :
            print("---평가: 연관 없는 문서---")
            web_search = "예"
            continue
    return {"documents": filtered_docs, "question":question, "web_search":web_search}
# 질문을 변환하는 노드
def transform_query(state):
    """
    질문을 더 적합한 형태로 변환합니다

    Args:
        state (dict): 현재 그래프의 상태

    Returns:
        state (dict): 변환된 질문이 업데이트된 상태
    """
    print("---질문 변환---")
    question = state["question"]
    documents = state["documents"]

    better_question = question_rewriter.invoke({"question":question})
    return {"documents":documents,"question":better_question}
# 웹 검색을 수행하는 노드
def web_search(state):
    """
    웹 검색을 수행합니다

    Args:
        state (dict): 현재 그래프의 상태

    Returns:
        state (dict): 웹 검색 결과가 업데이트된 상태
    """

    print("---웹 검색---")
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query":question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents":documents,"question":question}
# 결정 노드
def decide_to_generate(state):
    """
    답변을 생성할지, 질문을 재 생성할지 결정합니다.

    Args:
        state (dict): 현재 그래프의 상태

    Returns:
        str: 다음에 호출할 노드
    """

    print("---문서 검토---")
    web_search = state["web_search"]

    if web_search == "예":
        print(
            "---연관 없는 문서가 있음. 질문을 변환---"
        )
        return "transform_query"
    else:
        print("---연관 문서가 있음. 답변을 생성---")
        return "generate"
    
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# 노드 정의
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search_node", web_search)  # web search

# 그래프 정의
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

# 그래프 컴파일
app = workflow.compile()

from pprint import pprint

inputs = {"question": "구글의 코드 작성 가이드"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
        # pprint(value, indent=2, width=80, depth=None)

pprint(value["generation"])

inputs = {"question": "C++ 깔끔하게 짜고싶다"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
        # pprint(value, indent=2, width=80, depth=None)

pprint(value["generation"])