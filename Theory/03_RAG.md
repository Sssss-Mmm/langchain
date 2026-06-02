# 3단계: RAG (Retrieval-Augmented Generation)

---

## LLM의 근본적인 한계

LLM은 학습 데이터 기준으로만 답합니다.

```
문제 1: 최신 정보 모름
  "오늘 삼성전자 주가가 얼마야?" → 모름 (학습 시점 이후)

문제 2: 내부 정보 모름
  "우리 회사 2024년 매출은?" → 모름 (학습 안 됨)

문제 3: 환각(Hallucination)
  모르는 것을 그럴듯하게 지어냄
```

이걸 해결하는 게 **RAG**입니다.

---

## RAG = 검색 + 생성

> "답변하기 전에 먼저 관련 문서를 찾아서 참고해라"

```
[일반 LLM]
사용자 질문 → LLM → 답변 (기억에 의존)

[RAG]
사용자 질문 → 문서 검색 → 관련 문서 + 질문 → LLM → 답변
```

사람으로 치면, 시험을 "오픈북"으로 보는 것과 같습니다.

---

## RAG 전체 파이프라인

### 1단계: 문서 준비 (오프라인)

```
PDF/Word/웹페이지
    ↓
[문서 로더]         ← LangChain의 DocumentLoader
    ↓
텍스트 추출
    ↓
[텍스트 분할기]     ← chunk_size=500, overlap=50
    ↓
작은 청크(chunk)들
    ↓
[임베딩 모델]
    ↓
벡터들 → [벡터 DB 저장]
```

### 2단계: 질문 답변 (실시간)

```
사용자 질문
    ↓
[임베딩 모델]
    ↓
질문 벡터
    ↓
[벡터 DB 검색] → 유사한 청크 Top K개 반환
    ↓
[프롬프트 구성]
  "다음 문서를 참고해서 답해줘:
   [검색된 문서들]
   질문: [사용자 질문]"
    ↓
[LLM]
    ↓
최종 답변
```

---

## 청크(Chunk)란?

문서를 LLM이 처리하기 좋은 크기로 잘라낸 조각입니다.

```
[긴 PDF 문서]
"... 1페이지 내용 ... 2페이지 내용 ... 3페이지 내용 ..."
                ↓ 분할
청크1: "1페이지 내용..."       (500자)
청크2: "...1페이지 끝 + 2페이지 시작..."  (500자, overlap으로 문맥 유지)
청크3: "2페이지 내용..."       (500자)
```

**overlap(겹침)** 을 두는 이유: 청크 경계에서 문맥이 끊기지 않도록.

---

## Fine-tuning vs RAG

둘 다 LLM에게 새로운 지식을 주는 방법이지만 방식이 다릅니다.

| | RAG | Fine-tuning |
|--|-----|-------------|
| **방식** | 검색해서 참고 | 모델 자체를 재학습 |
| **비용** | 낮음 | 매우 높음 |
| **최신화** | 쉬움 (DB만 업데이트) | 어려움 (재학습 필요) |
| **적합한 경우** | 자주 바뀌는 정보 | 말투/스타일 변경 |
| **현재 프로젝트** | Ch02~Ch04 | Ch08 |

**결론:** 대부분의 실무에서는 RAG를 먼저 시도합니다.

---

## LangChain에서 RAG 코드 구조

```python
# 1. 문서 로드
loader = PDFLoader("document.pdf")
docs = loader.load()

# 2. 청크 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)
chunks = splitter.split_documents(docs)

# 3. 벡터 DB 저장
vectorstore = Chroma.from_documents(chunks, embedding_model)

# 4. 검색기 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. RAG 체인 구성
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 6. 질문
answer = chain.invoke("질문 내용")
```

---

## 정리

| 개념 | 한줄 요약 |
|------|-----------|
| RAG | 검색 후 생성 — LLM이 문서를 참고해서 답변 |
| 청크 | 문서를 적당한 크기로 자른 조각 |
| Retriever | 질문과 유사한 청크를 찾아주는 검색기 |
| Fine-tuning | 모델 자체를 재학습 (RAG와 다름) |

---

## 이전/다음 단계

← [2단계: 벡터 임베딩 & 유사도 검색](./02_Vector_Embedding.md)

→ [4단계: Agent & Tool Use](./04_Agent.md)
