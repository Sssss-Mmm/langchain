from dotenv import load_dotenv

load_dotenv()


from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

# 크롤링할 URL을 지정
url = "https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel"
# 해당 페이지를 재귀적으로 크롤링
loader = RecursiveUrlLoader(
    url= url , max_depth=20, extractor=lambda x: Soup(x,"html.parser").text
)

# 지정된 URL에서 문서들을 크롤링하고, 이를 'docs' 변수에 저장
docs = loader.load()

# 크롤링된 문서들을 'source' 메타데이터를 기준으로 정렬
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))

# 모든 문서의 내용을 하나의 문자열로 연결
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# LLM이 LCEL 전문가로서 사용자의 질문에 답변하도록 지시하는 프롬프트를 정의
system = """
당신은 LCEL(LangChain expression language) 전문가인 코딩 어시스턴트입니다.
다음은 필요한 LCEL 문서 전문입니다:
-------
{context}
-------
위에 제공된 문서를 기반으로 사용자 질문에 답변하세요.
제공하는 코드는 실행 가능해야 하며, 필요한 모든 import 문과 변수들이 정의되어 있어야 합니다.
답변을 다음과 같은 구조로 작성하세요:
1. prefix : 문제와 접근 방식에 대한 설명
2. imports : 코드 블록 import 문
3. code : import 문을 제외한 코드 블록
4. description : 질문에 대한 코드 스키마

다음은 사용자 질문입니다:
"""

# 시스템 메시지와 사용자의 질문을 포함한 템플릿을 생성
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("placeholder","{messages}")
    ]
)

# 코드 출력을 구조화하기 위한 데이터 모델을 정의
class code(BaseModel):
    prefix: str = Field(description="문제와 접근 방식에 대한 설명")
    imports : str = Field(description="코드 블록 import 문")
    code : str = Field(description="import 문을 제외한 코드 블록")
    description : str = Field(description="질문에 대한 코드 스키마")

# 코드 생성을 위한 LLM 정의
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# 프롬프트, 구조화된 LLM 출력을 결합하여 RAG 체인 생성
code_gen_chain = code_gen_prompt | llm.with_structured_output(code)

question = "LCEL로 RAG 체인 어떻게 만들어?"
solution = code_gen_chain.invoke(
    {"context":concatenated_content,"messages":[("user",question)]}
)

print(solution)

from typing import List, TypedDict

class GraphState(TypedDict):
    error :str
    messages : List
    generation : str
    iterations : int

def generate(state: GraphState):
    """
    코드를 생성합니다

    Args:
        state (dict): 현재 그래프의 상태

    Returns:
        state (dict): 생성한 코드가 업데이트된 상태
    """

    print("---코드 생성---")

    messages =state["messages"]
    iterations = state["iterations"]
    error = state.get("error","no")

    if error == "yes":
        messages += [
            (
                "user",
                "다시 시도해보세요. 출력 결과를 prefix,imports, code block으로 구조화하기 위해 코드 도구를 호출하세요:"
            )
        ]
    
    code_solution = code_gen_chain.invoke(
        {"context": concatenated_content, "messages":messages}
    )
    messages += [
        (
            "assistant",
            f"{code_solution.prefix} \n imports: {code_solution.imports} \n Code: {code_solution.code}"
        )
    ]
    iterations = iterations + 1
    return {"generation":code_solution,"messages":messages, "iterations":iterations}
        

def code_check(state:GraphState):
    """
    코드 검사
    
    Args:
        state (dict): 현재 그래프의 상태
    
    Returns:
        state (dict): 에러 여부가 업데이트된 상태
    """

    print("---코드 검사---")

    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    imports = code_solution.imports
    code = code_solution.code

    try:
        exec(imports)
    except Exception as e :
        print("---imprt 체크: 실패---")
        error_message = [("user", f"당신의 코드는 import 테소트를 실패했습니다: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations":iterations,
            "error": "yes",
        }
    
    try:
        exec(imports + "\n" + code)
    except Exception as e :
        print("---code block 체크 : 실패---")
        error_message = [("user", f"당신의 코드는 실행 테스트를 실패했습니다: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations":iterations,
            "error": "yes",            
        }
    
    print("---에러 없음---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations":iterations,
        "error": "no",            
    }

def reflect(state: GraphState):
    """
    에러 반영

    Args:
        state (dict): 현재 그래프 상태

    Returns:
        state (dict): 생성된 코드가 추가된 상태
    """

    print("---코드 솔루션 생성---")

    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]

    reflections = code_gen_chain.invoke(
        {"context": concatenated_content, "messages": messages}
    )
    messages += [("assistant",f"여기 오류를 반영한 코드입니다:{reflections}")]
    return {"generation":code_solution,"messages":messages, "iterations":iterations}

flag = "do not reflect"  

def decide_to_finish(state: GraphState):
    """
    종료 여부를 결정합니다.

    Args:
        state (dict): 현재 그래프의 상태

    Returns:
        str: 다음에 호출할 노드
    """    
    error =state["error"]
    iterations = state["iterations"]

    if error  == "no" or iterations == 3:
        print("---종료---")
        return "end"
    else:
        print("---재시도---")
        if flag is True :
            return "reflect"
        else:
            return "generate"
        
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

workflow.add_node("generate",generate)
workflow.add_node("check_node",code_check)
workflow.add_node("reflect",reflect)

# 그래프 정의
workflow.add_edge(START,"generate")
workflow.add_edge("generate","check_node")
workflow.add_conditional_edges(
    "check_node",
    decide_to_finish,
    {
        "end":END,
        "reflect":"reflect",
        "generate":"generate"
    },
)

workflow.add_edge("reflect","generate")

# 그래프 컴파일
app = workflow.compile()

question = "문자열을 runnable 객체에 직접 전달하고 , 이를 사용하여 내 프롬프트에 필요한 입력을 구성하려면 어떻게 해야 하나요?"

print(app.invoke({"messages": [("user", question)], "iterations": 0}))