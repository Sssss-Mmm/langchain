import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 0. 환경 변수 로드
load_dotenv()

# ==========================================
# 1. 도구 설정 (YouTube Script Loader)
# ==========================================
@tool
def get_youtube_transcript(url: str) -> str:
    """
    YouTube URL을 입력받아 해당 영상의 자막(Transcript)과 정보를 반환합니다.
    영상 요약이나 내용 분석 시 사용하세요.
    """
    try:
        # language=["ko", "en"] : 한국어 자막 우선, 없으면 영어 자막
        loader = YoutubeLoader.from_youtube_url(
            url, 
            add_video_info=True, 
            language=["ko", "en"]
        )
        docs = loader.load()
        
        if not docs:
            return "해당 영상에서 자막을 가져올 수 없습니다."
            
        # 첫 번째 문서의 내용과 메타데이터 추출
        doc = docs[0]
        title = doc.metadata.get("title", "Unknown Title")
        author = doc.metadata.get("author", "Unknown Author")
        content = doc.page_content
        
        return f"제목: {title}\n채널: {author}\n내용(자막):\n{content[:5000]}..." # 너무 길면 5000자에서 자름 (토큰 제한 고려)
        
    except Exception as e:
        return f"Error fetching transcript: {e}"

tools = [get_youtube_transcript]

# ==========================================
# 2. 에이전트 설정
# ==========================================

llm = ChatOpenAI(model="gpt-4o", temperature=0)

system_message = "당신은 유튜브 영상 내용을 분석하고 요약해주는 AI 비서입니다. 사용자가 제공한 URL의 자막을 읽고 질문에 답변하세요."

agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

def run_agent_and_print_process(query: str):
    print(f"\nUser: {query}")
    print("-" * 50)
    
    result = agent_executor.invoke({"messages": [("user", query)]})
    
    for msg in result['messages']:
        if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            for tool_call in msg.tool_calls:
                print(f"  ▶ [Tool Call] {tool_call['name']}: {tool_call['args']}")
        elif msg.type == "tool":
            content = str(msg.content)
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"  ▷ [Tool Result] {preview}") 
        elif msg.type == "ai" and not msg.tool_calls:
            print(f"\nAgent: {msg.content}")
    print("-" * 50)

# ==========================================
# 3. 실습 시나리오
# ==========================================
print("=== YouTube Analysis Agent ===")

# 시나리오 1: 랭체인 관련 튜토리얼 영상 요약 (예: LangChain 공식 설명)
# (URL은 예시이며 실제 유효한 URL로 교체해서 테스트 가능)
sample_url = "https://www.youtube.com/watch?v=aywZrzNaKjs" # "LangChain in 13 Minutes" (Quickstart)
run_agent_and_print_process(f"다음 유튜브 영상 내용을 3줄로 요약해줘: {sample_url}")

# 시나리오 2: 특정 정보 추출
run_agent_and_print_process(f"위 영상에서 언급된 'Chains'의 핵심 개념이 뭐야?")
