from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

def get_web_search_agent():
    """
    Returns a configured Web Search Agent executor.
    """
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # LangGraph ReAct Agent
    return create_react_agent(llm, tools)
