from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# 2. 문제 생성 체인 (Question Generation)
gen_system = """당신은 Python 코딩 튜터입니다.
주어진 '주제(Topic)', '난이도(Difficulty)', '문제 유형(Question Type)'에 맞는 코딩 퀴즈를 하나 만들어주세요.

문제 유형별지침:
- conceptual: 주제에 대한 핵심 개념을 묻는 서술형/단답형 문제
- coding: 간단한 기능을 구현하는 코딩 과제 (함수 작성 등)
- debugging: 버그가 있는 코드를 제시하고 원인을 찾거나 수정하게 하는 문제

문제는 간결해야 하며, 정답 코드를 포함하지 마세요. 사용자가 직접 풀 수 있도록 유도하세요."""

gen_prompt = ChatPromptTemplate.from_messages([
    ("system", gen_system),
    ("human", "주제: {topic}\n난이도: {difficulty}\n문제 유형: {question_type}\n\n새로운 문제를 하나 출제해줘.")
])

# 출력은 문제 텍스트 자체
generate_question_chain = gen_prompt | llm | StrOutputParser()


# 3. 답변 평가 체인 (Evaluation)
# 정확한 평가를 위해 JSON 구조나 명확한 키워드 출력을 유도할 수 있지만,
# 여기서는 간단히 "Pass" 또는 "Fail" 키워드를 첫 줄에 포함하도록 지시합니다.
eval_system = """당신은 엄격하지만 친절한 코딩 튜터입니다.
사용자의 답변을 평가하고, 정답 여부를 판단하세요.

응답 형식:
첫 줄: [PASS] 또는 [FAIL]
두 번째 줄부터: 상세한 피드백 또는 해설

사용자가 정답을 맞췄다면 [PASS], 틀렸거나 부족하다면 [FAIL]을 출력하세요.
"""

eval_prompt = ChatPromptTemplate.from_messages([
    ("system", eval_system),
    ("human", """
[현재 문제]
{current_question}

[사용자 답변]
{user_answer}

이 답변이 정답인가요? 평가해주세요.
""")
])

evaluate_answer_chain = eval_prompt | llm | StrOutputParser()


# 4. 힌트 제공 체인 (Hint Generation)
hint_system = """당신은 코딩 튜터입니다. 사용자가 문제를 틀렸습니다.
정답을 직접 알려주지 말고, 스스로 생각할 수 있도록 유도하는 '힌트'를 주세요.
이전 시도 횟수를 고려하여, 많이 틀렸다면 조금 더 구체적인 힌트를 주세요.
"""

hint_prompt = ChatPromptTemplate.from_messages([
    ("system", hint_system),
    ("human", """
[현재 문제]
{current_question}

[지금까지 시도 횟수]
{attempt_count}

사용자가 어려워하고 있습니다. 힌트를 하나 주세요.
""")
])
provide_hint_chain = hint_prompt | llm | StrOutputParser()

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 5.1 벡터 스토어 로드
DB_PATH = os.path.join(os.path.dirname(__file__), "VectorStore")
# DB가 존재하는지 확인
if os.path.exists(DB_PATH):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    # 주제와 연관된 내용을 찾기 위해 검색기 설정 (랜덤성을 위해 k를 늘림)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
else:
    vectorstore = None
    retriever = None

import random

def get_random_documents(k=3, subject=None):
    """
    벡터 스토어에서 랜덤하게 k개의 문서를 추출합니다.
    subject가 주어지면 해당 과목의 문서만 추출합니다.
    """
    if vectorstore is None:
        return []
    
    # docstore의 모든 문서 ID 가져오기
    doc_ids = list(vectorstore.docstore._dict.keys())
    
    if subject and subject != "Random":
        # 메타데이터 필터링
        filtered_ids = []
        for doc_id in doc_ids:
            doc = vectorstore.docstore.search(doc_id)
            # 메타데이터의 subject 필드 확인 (부분 일치 허용)
            # ingest.py에서 "1.소프트웨어 설계" 등으로 저장함
            # subject 인자는 "1.소프트웨어 설계" 와 같이 들어올 예정
            doc_subject = doc.metadata.get('subject', 'Unknown')
            if subject in doc_subject:
                filtered_ids.append(doc_id)
        
        if not filtered_ids:
            # 해당 과목 문서가 없으면 fallback (전체에서)
            print(f"[Warning] No documents found for subject: {subject}. Falling back to all docs.")
            filtered_ids = doc_ids
    else:
        filtered_ids = doc_ids
        
    if not filtered_ids:
        return []
        
    selected_ids = random.sample(filtered_ids, min(len(filtered_ids), k))
    return [vectorstore.docstore.search(doc_id) for doc_id in selected_ids]

# 5.2 출력 데이터 구조 정의
class ExamQuestion(BaseModel):
    question: str = Field(description="The question text")
    options: list[str] = Field(description="List of 4 options, e.g. ['1. option A', '2. option B', ...]")
    answer: str = Field(description="The correct answer, e.g. '1' or '2'")
    explanation: str = Field(description="Detailed explanation of the answer")

parser = JsonOutputParser(pydantic_object=ExamQuestion)

# 5.3 프롬프트 및 체인 정의
rag_system = """당신은 정보처리기사 필기 시험 문제 출제자입니다.
제공된 [Context]를 바탕으로, 실제 시험에 나올법한 5지 선다형 객관식 문제를 하나 만들어주세요.
반드시 JSON 형식으로 출력해야 하며, 다음 키를 포함하세요: 'question', 'options', 'answer', 'explanation'.
Context 내용이 부족하면, 해당 주제(Topic)에 관한 일반적인 지식을 활용하여 문제를 만드세요."""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system),
    ("human", """
[Context]
{context}

[Topic]
{topic}

위 내용을 바탕으로 객관식 문제를 하나 출제해줘.
FORMAT:
{format_instructions}
""")
])

# RAG 체인 호출 시: {"context": retrieved_docs, "topic": input_topic, "format_instructions": parser.get_format_instructions()}
rag_question_chain = rag_prompt | llm | parser
