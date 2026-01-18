from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import os
from dotenv import load_dotenv

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_describer():
    print("=== Image Describer (Vision Model) ===")
    
    # 1. 이미지 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Data 폴더에 있는 첫 번째 이미지를 사용해봅니다.
    data_dir = os.path.join(current_dir, "..", "Data")
    
    # jpg 파일 찾기
    image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
    if not image_files:
        print("No images found in Data directory.")
        return

    # 첫 번째 이미지 선택
    target_image = image_files[0]
    image_path = os.path.join(data_dir, target_image)
    print(f"Target Image: {target_image}")
    
    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    # 2. 모델 설정
    model = ChatOpenAI(model="gpt-4o", max_tokens=1024)

    # 3. 메시지 구성 (멀티모달)
    # 텍스트와 이미지를 함께 전달합니다.
    message = HumanMessage(
        content=[
            {"type": "text", "text": "이 이미지를 자세히 설명해주세요. 표나 그래프가 있다면 주요 수치나 트렌드를 언급해주세요."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )

    # 4. 실행
    print("Generating description...")
    response = model.invoke([message])
    
    print("\n[Description]")
    print(response.content)

if __name__ == "__main__":
    image_describer()
