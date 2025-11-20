import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = ("./Data/투자설명서.pdf")

loader = PyPDFLoader(file_path)

doc_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
docs = loader.load_and_split(doc_splitter)

from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

from langchain_community.vectorstores import FAISS
from tqdm import tqdm
batch_size = 20
faiss_store = None

# 배치 단위로 FAISS 인덱스 생성 및 병합
for i in tqdm(range(0, len(docs), batch_size)):
    batch = docs[i:i + batch_size]
    temp_store = FAISS.from_documents(batch, embedding)

    if i == 0:
        faiss_store = temp_store
    else:
        faiss_store.merge_from(temp_store)

persist_directory = "./DB"
faiss_store.save_local(persist_directory)

# 저장된 DB 경로 지정 후 ,DB 로드
vectordb = FAISS.load_local(persist_directory,embeddings=embedding,allow_dangerous_deserialization=True)

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from textwrap import dedent
from langchain_core.output_parsers import JsonOutputParser

# 재순위화 점수용 Pydantic 모델 정의
class RelevanceScore(BaseModel):
    relevance_score: float = Field(description="문서가 쿼리와 얼마나 관련이 있는지 나타내는 점수.")

# 재순위화 함수 정의
def reranking_documents(query:str, docs: List[Document], top_n: int = 2)-> List[Document]:
    parser = JsonOutputParser(pydantic_object=RelevanceScore)
    human_message_prompt = PromptTemplate(
        template = """
        1점부터 10점까지 점수를 매겨 다음 문서가 질문이 얼마나 관련이 있는지 평가해주세요. 단순히 키워드가 일치하는 것이 아니라 쿼리의 구체적인 맥락과 의도를 고려하세요.
        {format_instructions}
        question: {query}
        document: {doc}
        relevance_score:""",
        input_variables=["query","doc"],
        partial_variables ={"format_instructions": parser.get_format_instructions()}
    )

    llm = ChatOpenAI(temperature=0, model_name = "gpt-4o", max_tokens = 3000)
    chain = human_message_prompt | llm | parser
    scored_docs = []

    for doc in docs :
        input_data = {"query":query, "doc":doc.page_content}
        try:
            score = chain.invoke(input_data)['relevance_score']
            score = float(score)
        except Exception as e :
            print(f"오류 발생 : {str(e)}")
            default_score = 5 
            print(f"기본 점수 {default_score}점을 사용합니다.")
            score = default_score
        scored_docs.append((doc, score))

    reranked_docs =sorted(scored_docs,key=lambda x : x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]

query = "이 회사의 2022년 영업손실이 정확히 얼마야?"
init_docs = vectordb.similarity_search(query, k=4)
reranked_docs = reranking_documents(query,init_docs)

print(f"Query: {query}\n\n")

print("Top init documents:")
for i, doc in enumerate(init_docs):
    print(f"\nDocument {i+1}:")
    print(doc.page_content)

print("\n\nTop reranked documents:")
for i, doc in enumerate(reranked_docs):
    print(f"\nDocument {i+1}:")
    print(doc.page_content)


from langchain_core.retrievers import BaseRetriever
from langchain.chains import RetrievalQA

# 커스텀 리트리버 정의
class CustomRetriever(BaseRetriever):
    vectorstore : Any = Field(description="Retrieval을 위한 벡터스토어")
    # Pydantic 설정
    class Config: 
        arbitrary_types_allowed = True
    # 재순위화된 문서 반환 메서드 재정의
    def get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query,k=4)
        return reranking_documents(query,initial_docs,top_n=num_docs)
    
# 커스텀 리트리버 인스턴스 생성    
custom_retriever = CustomRetriever(vectorstore=vectordb)

llm = ChatOpenAI(temperature=0.2, model_name = "gpt-4o")

# 관련있는 문서를 수집 후 , Chatgpt로 최종 답변까지 수행하는 체인을 생성
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = custom_retriever,
    return_source_documents = True
)

print(qa_chain.invoke("이 회사의 2022년 영업 손실이 정확히 얼마야?"))