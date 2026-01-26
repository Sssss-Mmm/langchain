import os
import tempfile
import shutil
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def get_youtube_transcript(url: str) -> str:
    """
    YouTube URL을 입력받아 해당 영상의 자막(Transcript)과 정보를 반환합니다.
    
    1. 유튜브 기본 자막(한국어, 영어 및 자동번역)을 우선 시도합니다.
    2. 자막이 없을 경우, 영상의 오디오를 다운로드하여 Whisper AI로 받아쓰기(Transcription)를 수행합니다.
    
    영상 요약이나 내용 분석 시 사용하세요.
    """
    # 1. 자막 가져오기 시도
    try:
        loader = YoutubeLoader.from_youtube_url(
            url, 
            add_video_info=True, 
            language=["ko", "ko-KR", "en", "en-US"],
            translation="ko"
        )
        docs = loader.load()
        
        if docs:
            doc = docs[0]
            title = doc.metadata.get("title", "Unknown Title")
            author = doc.metadata.get("author", "Unknown Author")
            return f"제목: {title}\n채널: {author}\n[Source: 자막/자동번역]\n\n내용:\n{doc.page_content[:5000]}..."
            
    except Exception as e:
        print(f"자막 로드 실패, Whisper 전환 시도: {e}")
        pass # 자막 실패 시 Whisper로 넘어감

    # 2. Whisper로 오디오 트랜스크립션 시도
    temp_dir = tempfile.mkdtemp()
    try:
        print("자막이 없어 오디오 분석을 시작합니다... (시간이 소요될 수 있습니다)")
        # yt-dlp 기반 로더
        loader = YoutubeAudioLoader([url], temp_dir)
        # OpenAI Whisper 파서
        parser = OpenAIWhisperParser()
        
        # 다운로드 및 변환 수행
        docs = loader.load() # 여기서 다운로드 발생
        
        # 파싱 (Whisper API 호출)
        transcribed_docs = parser.parse(docs[0]) if docs else [] # YoutubeAudioLoader returns list of blobs? No, check usages.
        # Actually YoutubeAudioLoader returns blobs, we need GenericLoader to combine. 
        # But simpler: use GenericLoader
        from langchain_community.document_loaders.generic import GenericLoader
        
        loader_pipeline = GenericLoader(
            YoutubeAudioLoader([url], temp_dir),
            OpenAIWhisperParser()
        )
        docs = loader_pipeline.load()

        if not docs:
            return "오디오 분석에 실패했습니다."

        content = docs[0].page_content
        title = docs[0].metadata.get("title", "Unknown Title") # Metadata might be limited depending on loader
        
        return f"제목: {title}\n[Source: Whisper AI Audio Transcription]\n\n내용:\n{content[:5000]}..."

    except Exception as e:
        return f"분석 실패: {e}\n(자막도 없고 오디오 다운로드도 실패했습니다)"
    finally:
        # 임시 파일 정리
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def get_youtube_agent():
    """
    Returns a configured YouTube Analysis Agent executor.
    """
    tools = [get_youtube_transcript]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_message = "당신은 유튜브 영상 내용을 분석하고 요약해주는 AI 비서입니다. 사용자가 제공한 URL의 자막을 읽고 질문에 답변하세요. 자막이 없으면 오디오 분석 결과를 사용합니다."
    
    return create_react_agent(llm, tools, prompt=system_message)
