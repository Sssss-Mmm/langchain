from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import base64
import os
from dotenv import load_dotenv

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def visual_qa_chat():
    print("=== Visual QA (Chat with Image) ===")
    
    # 1. 이미지 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "Data")
    image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
    
    if not image_files:
        print("No images found.")
        return

    # 예제: 파레토 법칙 그래프나 표 같은 것이 있다면 좋음
    # 여기서는 랜덤으로 하나 선택하거나 특정 파일을 지정
    target_image = image_files[0] 
    image_path = os.path.join(data_dir, target_image)
    print(f"Selected Image: {target_image}")
    
    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. 모델 설정
    model = ChatOpenAI(model="gpt-4o", max_tokens=1024)

    # 3. 대화 루프 (메모리 직접 관리)
    # 이미지 컨텍스트를 유지하기 위해 메시지 리스트에 이미지를 포함한 초기 메시지를 저장해둡니다.
    print("[System] Chat session started. Type 'quit' to exit.")
    
    chat_history = [
        SystemMessage(content="당신은 이미지를 분석하고 사용자의 질문에 답하는 Visual AI 어시스턴트입니다."),
        HumanMessage(
            content=[
                {"type": "text", "text": "이 이미지를 참고자료로 제공합니다. 이후 질문에 대해 이 이미지를 바탕으로 답변해주세요."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        ),
        AIMessage(content="네, 이미지를 확인했습니다. 무엇이 궁금하신가요?")
    ]
    
    print(f"AI: {chat_history[-1].content}")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        # 사용자 메시지 추가
        chat_history.append(HumanMessage(content=user_input))
        
        # 모델 호출
        response = model.invoke(chat_history)
        
        # 응답 출력 및 히스토리 저장
        print(f"AI: {response.content}")
        chat_history.append(response)

if __name__ == "__main__":
    visual_qa_chat()
