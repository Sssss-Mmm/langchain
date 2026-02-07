"""
Terms Analyst Agent - 약관 분석 에이전트
PDF 문서 기반 질의응답 (RAG)
"""
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from config.settings import OPENAI_MODEL


def get_terms_analyst_chain(vector_store: VectorStore):
    """
    약관 분석을 위한 RAG 체인을 생성합니다.
    
    Args:
        vector_store: 문서가 저장된 벡터 저장소
        
    Returns:
        실행 가능한 RAG 체인
    """
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,  # 사실 기반 응답을 위해 0으로 설정
    )
    
    # 검색기 설정
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # 프롬프트 템플릿
    system_prompt = (
        "당신은 금융 약관 분석 전문가입니다. "
        "제공된 context를 기반으로 질문에 정확하게 답변하세요. "
        "만약 context에 정보가 없다면, '제공된 문서에서 관련 내용을 찾을 수 없습니다'라고 답변하세요. "
        "답변 시, 근거가 되는 조항이나 내용을 구체적으로 언급하세요."
        "\n\nContext:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # 문서 결합 체인
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # RAG 체인
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain
