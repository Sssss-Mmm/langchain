import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import ChatOpenAI

# 0. 환경 변수 로드
load_dotenv()

# 1. LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. 도구(Tools) 로드
# "llm-math": LLM이 수학 계산을 할 수 있게 해주는 도구
# "tavily-search": 웹 검색 도구 (API Key 필요) -> 여기서는 예제로 math만 주로 활용
tools = load_tools(["llm-math"], llm=llm)

# 만약 Tavily API Key가 있다면 검색 도구도 추가 가능
# if os.getenv("TAVILY_API_KEY"):
#     tools.extend(load_tools(["tavily-search-results-json"], llm=llm))


# 3. ReAct Prompt 로드
# LangChain Hub에서 검증된 React 프롬프트를 가져옵니다.
prompt = hub.pull("hwchase17/react")

# 4. 에이전트 생성
agent = create_react_agent(llm, tools, prompt)

# 5. 에이전트 실행기(Executor) 생성
# verbose=True로 설정하면 Thought(생각) -> Action(행동) -> Observation(관찰) 과정을 볼 수 있습니다.
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
)

# 6. 실행 예제
print("=== Basic ReAct Agent (Math) ===")
question = "25의 제곱근에 10을 곱하고 5를 더하면 얼마야?"
print(f"\n질문: {question}")

try:
    result = agent_executor.invoke({"input": question})
    print(f"\n최종 답변: {result['output']}")
except Exception as e:
    print(f"오류 발생: {e}")
