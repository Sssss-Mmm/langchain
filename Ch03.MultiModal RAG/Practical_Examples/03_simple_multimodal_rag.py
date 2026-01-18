import os
import uuid
import base64
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from dotenv import load_dotenv
import shutil

load_dotenv()

# --- 헬퍼 함수 ---
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, model):
    """이미지 내용을 텍스트로 요약 (검색용)"""
    msg = model.invoke([
        HumanMessage(
            content=[
                {"type": "text", "text": "이 이미지를 검색에 최적화된 형태로 자세히 설명(요약)해주세요."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ]
        )
    ])
    return msg.content

def simple_multimodal_rag():
    print("=== Simple Multi-Modal RAG ===")
    
    # 0. 초기화
    persist_dir = "./mm_rag_db"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "Data")
    
    # jpg 2개만 샘플로 사용 (시간 절약)
    image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
    image_files = sorted(image_files)[:2]
    
    if not image_files:
        print("No images found.")
        return

    print(f"Indexing {len(image_files)} images...")
    
    # 1. 컴포넌트 설정
    # Vision 모델 (요약용)
    vision_model = ChatOpenAI(model="gpt-4o", max_tokens=1024)
    # 임베딩 모델
    embedding_model = OpenAIEmbeddings()
    # 벡터 저장소
    vectorstore = Chroma(
        collection_name="simple_mm_rag", 
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )
    # 원본 저장소 (Docstore) - 여기서는 메모리에 저장 (이미지 base64 등)
    store = InMemoryStore()
    id_key = "doc_id"
    
    # 멀티 벡터 검색기
    # 벡터 DB에는 '텍스트 요약'을 저장하고, 검색 시 연결된 '원본(이미지)'를 반환함
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # 2. 데이터 처리 및 인덱싱
    img_base64_list = []
    img_summaries = []
    doc_ids = []

    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        
        # 인코딩
        b64 = encode_image(img_path)
        img_base64_list.append(b64)
        
        # 요약 생성
        print(f"Summarizing {img_file}...")
        summary = image_summarize(b64, vision_model)
        img_summaries.append(summary)
        
        # ID 생성
        doc_ids.append(str(uuid.uuid4()))

    # 3. 데이터 저장
    
    # A. 원본(Base64 이미지)을 Docstore에 저장 (ID 매핑)
    # MultiVectorRetriever가 검색 후 이 ID로 원본을 찾아옴
    retriever.docstore.mset(list(zip(doc_ids, img_base64_list)))

    # B. 요약본을 Vectorstore에 저장 (ID를 메타데이터로 포함)
    summary_docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(img_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    
    print("Indexing Complete.")

    # 4. 검색 및 RAG 실행
    query = "이 문서들에 포함된 그래프나 도표의 내용은 무엇인가요?"
    print(f"\n[Query]: {query}")
    
    # 검색 실행 -> 관련된 이미지(Base64)가 반환됨
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} images.")
    
    if not retrieved_docs:
        print("Nothing retrieved.")
        return

    # 첫 번째 검색 결과로 답변 생성 (Multi-modal RAG)
    # 실제로는 여러 개를 context로 넣을 수 있음
    top_image_b64 = retrieved_docs[0]
    
    # 답변 생성 LLM
    qa_chain = ChatOpenAI(model="gpt-4o")
    
    msg = HumanMessage(
        content=[
            {"type": "text", "text": f"질문: {query}\n\n아래 제공된 이미지를 참고하여 답변하세요."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{top_image_b64}"}},
        ]
    )
    
    print("Generating Answer...")
    answer = qa_chain.invoke([msg])
    print(f"\n[Answer]:\n{answer.content}")

    # 정리
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

if __name__ == "__main__":
    simple_multimodal_rag()
