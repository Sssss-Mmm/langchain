from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent

def get_python_repl_agent():
    """
    Returns a configured Python REPL Agent executor.
    """
    python_repl_tool = PythonREPLTool()
    tools = [python_repl_tool]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_message = "당신은 파이썬 코드를 작성하고 실행하여 사용자의 질문을 해결하는 전문가입니다. 문제 해결을 위해 필요한 코드를 생성하고 실행 결과를 확인한 뒤 답변하세요."
    
    return create_react_agent(llm, tools, prompt=system_message)
