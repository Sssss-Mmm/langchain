# LangChain 실습 가이드

이 리포지토리는 LangChain 프레임워크를 활용하여 LLM 애플리케이션을 개발하는 방법을 학습하고 실습하기 위한 코드 모음입니다. 기초적인 개념부터 고급 RAG 기법, LangGraph, Agent 구현까지 단계별로 구성되어 있습니다.

## 📂 목차 및 구조

각 디렉토리는 주제별로 구분되어 있으며, 파이썬 스크립트 형태의 실습 예제들을 포함하고 있습니다.

### [Ch01. Langchain Basics](./Ch01.Langchain%20Basics)
LangChain의 핵심 구성 요소를 학습합니다.
- **LCEL (LangChain Expression Language)**: 체인 구성의 기초
- **Prompts**: 프롬프트 템플릿 활용
- **Memory**: 대화 맥락 유지
- **Output Parsers**: LLM 출력의 구조화

### [Ch02. RAG (Retrieval-Augmented Generation)](./Ch02.RAG)
외부 데이터를 활용하여 LLM의 답변 능력을 증강시키는 RAG의 기본 파이프라인을 구축합니다.
- Document Loading & Text Splitting
- Embedding & Vector Database
- Basic RAG Chatbot 구현

### [Ch03. MultiModal RAG](./Ch03.MultiModal%20RAG)
텍스트뿐만 아니라 이미지 등 다양한 모달리티를 활용한 RAG를 다룹니다.

### [Ch04. Advanced RAG](./Ch04.Advaced%20RAG)
RAG의 성능을 극대화하기 위한 고급 기법들을 다룹니다.
- **Retrieval**: BM25, Ensemble, Dense Retrieval
- **Query Transformation**: HyDE, MultiQuery
- **Reranking**: Cross Encoder, LLM Reranker
- **Advanced Strategy**: Parent-Child Chunking, Self-RAG

### [Ch06. LangGraph](./Ch06.LangGraph)
LangChain의 그래프 기반 오케스트레이션 도구인 LangGraph를 학습합니다.
- 순환(Cyclic) 그래프 구조 및 상태 관리
- Code Assist Chatbot
- Corrective RAG 구현

### [Ch07. Agent](./Ch07.Agent)
스스로 사고하고 도구를 사용하여 문제를 해결하는 에이전트를 구현합니다.
- ReAct Agent
- 실제 정책 문서(PDF) 기반의 질의응답 에이전트

### [Ch08. Fine-tuning for RAG](./Ch08.Fine-tuning%20for%20RAG)
RAG 성능 향상을 위한 모델 파인튜닝 방법을 다룹니다.

---

## 🚀 시작하기

### 환경 설정
본 프로젝트는 Python 환경에서 실행됩니다. 가상 환경 사용을 권장합니다.

1. **리포지토리 클론**
   ```bash
   git clone <repository-url>
   cd langchain
   ```

2. **가상 환경 생성 및 활성화** (선택 사항)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   # venv\Scripts\activate  # Windows
   ```

3. **의존성 설치**
   필요한 패키지는 각 챕터의 코드 실행 시 확인하거나, 통합된 `requirements.txt`가 있다면 설치합니다.
   (일반적으로 `pip install langchain langchain-openai chromadb faiss-cpu` 등이 필요할 수 있습니다.)

4. **API 키 설정**
   OpenAI 등을 사용하기 위해 환경 변수 설정이 필요할 수 있습니다. `.env` 파일을 생성하거나 환경 변수에 키를 등록하세요.

### 실행 방법
각 실습 파일은 개별적으로 실행할 수 있습니다.
```bash
python "Ch01.Langchain Basics/ch01_LCEL.py"
```
