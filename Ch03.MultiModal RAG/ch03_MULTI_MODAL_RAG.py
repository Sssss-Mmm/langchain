from dotenv import load_dotenv
import os
from unstructured.partition.pdf import partition_pdf
load_dotenv()

import nltk

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

# 파일 경로
fpath = "./Data"
fname = "sample.pdf"

# PDF에서 요소 추출
raw_pdf_elements = partition_pdf(
    filename=os.path.join(fpath,fname),
    extract_images_in_pdf=True,
    extract_image_block_types=["image","Table"],
    chunking_strategy="by_title",
    extract_image_block_output_dir=fpath,
)

# 텍스트, 테이블 추출
tables = []
texts = []

for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        tables.append(str(element)) # 테이블 요소 추가
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        texts.append(str(element)) # 텍스트 요소 추가

# print(tables[0])
# print(texts[0])

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# 프롬프트 설정
prompt_text= """당신은 표와 텍스트를 요약하여 검색할 수 있도록 돕는 역할을 맡은 어시스턴트입니다.
이 요약은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것싱비낟.
표 또는 텍스트에 대한 간결한 요약을 제공하여 검색에 최적화된 형태로 만들어 주세요. 표 또는 텍스트:{element}"""
prompt = ChatPromptTemplate.from_template(prompt_text)

# 텍스트 요약 체인
model = ChatOpenAI(temperature=0, model="gpt-4o")
summarize_chain = {"element":lambda x: x}| prompt | model | StrOutputParser()

# 제공된 텍스트에 대해 요약을 할 경우
text_summaries = summarize_chain.batch(texts,{"max_concurrency":5})
table_summaries = summarize_chain.batch(tables,{"max_concurrency":5})

# print(table_summaries[0])
# print(text_summaries[0])

import base64
import mimetypes
from PIL import Image

def encode_image(image_path) -> str:
    # 이미지 base64 인코딩
    with open(image_path,"rb") as image_file:
        img= Image.open(image_path)
        img.thumbnail((1024, 1024))
        mime_type,_ = mimetypes.guess_type(image_path)
        return base64.b64encode(image_file.read()).decode('UTF-8')
    
# 이미지의 base64 인코딩을 저장하는 리스트
img_base64_list = []

# 이미지를 읽어 base64 인코딩 후 저장
for img_file in sorted(os.listdir(fpath)):
    if img_file.endswith('.jpg'):
        img_path = os.path.join(fpath,img_file)
        base64_image = encode_image(img_path)
        img_base64_list.append(base64_image)

# print(len(img_base64_list))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

def image_summarize(img_base64: str) -> str:
    # GPT-4o 멀티모달 모델 선언
    chat = ChatOpenAI(model="gpt-4o", max_tokens=1024)

    prompt = (
        "당신은 이미지를 요약하여 검색을 위해 사용할 수 있도록 돕는 어시스턴트입니다.\n"
        "이 요약은 임베딩되어 원본 이미지를 검색하는 데 사용됩니다.\n"
        "이미지 검색에 최적화된 간결한 요약을 작성하세요."
    )

    # ✅ 올바른 메시지 포맷
    msg = chat.invoke([
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url":f"data:image/jpeg;base64,{img_base64}"
                    },
                },
            ]
        )
    ])
    return msg.content

# ✅ 사용 예시
image_summaries = []
for img_base64 in img_base64_list:
    summary = image_summarize(img_base64)
    image_summaries.append(summary)

print(image_summaries[0])