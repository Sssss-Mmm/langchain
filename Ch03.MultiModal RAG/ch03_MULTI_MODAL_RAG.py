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

# print(image_summaries[0])

from langchain.retrievers import MultiVectorRetriever
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 분할한 텍스트들을 색인할 벡터 저장조
vectorstore = Chroma(collection_name="multi_modal_rag",embedding_function=OpenAIEmbeddings())

# 원본 문서 저장을 위한 저장소 선언
docstore = InMemoryStore()
id_key = "doc_id"

# 검색기
retriever = MultiVectorRetriever(
    vectorstore = vectorstore,
    docstore= docstore,
    id_key=id_key
)

import uuid
# 원본 텍스트 데이터 저장
doc_ids = [str(uuid.uuid4()) for _ in texts]
retriever.docstore.mset(list(zip(doc_ids, texts)))

# 원본 테이블 데이터 저장
table_ids = [str(uuid.uuid4()) for _ in tables]
retriever.docstore.mset(list(zip(table_ids, tables)))

# 원본 이미지(base64) 데이터 저장
img_ids = [str(uuid.uuid4()) for _ in img_base64_list]
retriever.docstore.mset(list(zip(img_ids, img_base64_list)))

from langchain.schema.document import Document

# 텍스트 요약 벡터 저장
summary_texts = [
    Document(page_content=s,metadata={id_key: doc_ids[i]})
    for i , s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)

# 테이블 요약 벡터 저장
summary_tables = [
    Document(page_content=s,metadata={id_key: table_ids[i]})
    for i , s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)

# 이미지 요약 벡터 저장
summary_img = [
    Document(page_content=s,metadata={id_key: img_ids[i]})
    for i , s in enumerate(image_summaries)
]
retriever.vectorstore.add_documents(summary_img)

docs = retriever.invoke(
    "말라리아 군집 사례는 어떤가요?"
)

print(len(docs))

from base64 import b64decode

def split_image_text_types(docs):
    # 이미지와 텍스트 데이터를 분리
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {
        "images": b64,
        "texts": text
    }

docs_by_type = split_image_text_types(docs)

print(len(docs_by_type["images"]))

print(docs_by_type["images"][0][:100])
print(docs_by_type["texts"])

from IPython.display import display, HTML

def plt_img_base64(img_base64):
    # base64 이미지로 html 태그를 작성
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}'

    # html 태그를 기반으로 이미지를 표기
    display(HTML(image_html))

plt_img_base64(docs_by_type["images"][0])

from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

def prompt_func(dict):
    format_texts = "\n".join(dict["context"]["texts"])
    text = f"""
    다음 문맥에만 기반하여 질문에 답하세요. 문맥에는 텍스트, 표, 그리고 아래 이미지가 포함될 수 있습니다:
    질문: {dict["question"]}

    텍스트와 표:
    {format_texts}
    """

    prompt = [
        HumanMessage(
            content=[
                {"type":"text","text":text},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{dict['context']['images'][0]}"}}
            ]
        )
    ]

    return prompt

model =ChatOpenAI(temperature=0,model="gpt-4o",max_tokens=1024)

# RAG 파이프라인
chain = (
    {"context": retriever | RunnableLambda(split_image_text_types),"question":RunnablePassthrough()}
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)

print(chain.invoke("말라리아 군집 사례는 어떤가요?"))