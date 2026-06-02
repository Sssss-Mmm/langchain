# 4단계: Agent & Tool Use

---

## LLM만으로는 부족한 것들

```
"지금 날씨 알려줘"     → LLM은 인터넷 검색 불가
"이 파일 계산해줘"     → LLM은 파일 접근 불가
"코드 실행해서 결과줘" → LLM은 코드 실행 불가
```

LLM은 텍스트 생성만 할 수 있습니다. **Agent**는 이 한계를 도구(Tool)로 돌파합니다.

---

## Agent란?

> "목표를 받으면, 스스로 계획하고, 도구를 골라 사용하고, 결과를 보고, 다음 행동을 결정하는 LLM"

```
[일반 LLM]
질문 → 답변 (1회성)

[Agent]
목표 → 생각 → 도구 선택 → 실행 → 결과 확인 → 생각 → ... → 최종 답변
```

---

## ReAct 패턴: Agent의 사고 방식

가장 유명한 Agent 패턴입니다. **Re**asoning + **Act**ion의 합성어.

```
목표: "서울 날씨를 찾아서 우산이 필요한지 알려줘"

Thought: 날씨 정보가 필요하다. 날씨 검색 도구를 써야겠다.
Action: weather_search("서울")
Observation: "서울 현재 날씨: 비, 강수확률 90%"

Thought: 비가 온다. 우산이 필요하다는 답을 줄 수 있다.
Final Answer: "오늘 서울은 비가 오니 우산을 꼭 챙기세요!"
```

Thought → Action → Observation 을 목표 달성까지 반복합니다.

---

## Tool이란?

Agent가 사용할 수 있는 외부 기능들입니다.

```python
tools = [
    검색_도구,      # 인터넷 검색
    계산기_도구,    # 수식 계산
    파일읽기_도구,  # 파일 접근
    코드실행_도구,  # Python 실행
    DB조회_도구,    # 데이터베이스 쿼리
]
```

LLM이 상황에 따라 **어떤 도구를 쓸지 스스로 결정**합니다.

---

## LangChain Agent 구조

```
사용자 목표
    ↓
[Agent]
  ├── LLM (두뇌 역할)
  ├── Tools (손발 역할)
  └── Memory (단기 기억)
    ↓
[Tool 실행]
    ↓
[결과 관찰]
    ↓
[반복 or 종료]
    ↓
최종 답변
```

```python
from langchain.agents import create_react_agent, AgentExecutor

# 도구 정의
tools = [search_tool, calculator_tool]

# Agent 생성
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# 실행기
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
result = executor.invoke({"input": "서울 날씨 알려줘"})
```

---

## Multi-Agent: Agent들의 협업

복잡한 작업은 여러 Agent가 나눠서 처리합니다.

```
[오케스트레이터 Agent]
  ├── [검색 Agent]   → 정보 수집 담당
  ├── [분석 Agent]   → 데이터 분석 담당
  └── [작성 Agent]   → 보고서 작성 담당
```

현재 `AI_Portfolio_Dashboard` 프로젝트의 구조가 이것입니다:

```
Projects/AI_Portfolio_Dashboard/agents/
  ├── market_researcher.py   ← 시장 조사 Agent
  ├── portfolio_analyst.py   ← 포트폴리오 분석 Agent
  ├── risk_assessor.py       ← 리스크 평가 Agent
  └── terms_analyst.py       ← 용어 분석 Agent
```

---

## LangGraph: Agent 흐름 제어

LangChain의 Agent는 단순하고, **LangGraph**는 복잡한 흐름을 그래프로 표현합니다.

```
[LangChain Agent]  단순한 Thought→Action 루프
[LangGraph]        조건 분기, 병렬 실행, 상태 관리 가능

예시:
  검색 결과가 충분하면 → 답변 생성
  검색 결과가 부족하면 → 재검색
  오류 발생하면 → 대체 도구 사용
```

현재 프로젝트 `Ch06.LangGraph` 가 이 내용입니다.

---

## 전체 로드맵과 연결

```
Transformer    → LLM이 텍스트를 이해하는 원리
     ↓
벡터 임베딩    → 텍스트를 의미 있는 숫자로 변환
     ↓
RAG            → 문서를 검색해서 LLM에게 제공
     ↓
Agent          → LLM이 도구를 사용해 스스로 문제 해결
```

이 4가지가 현대 AI 애플리케이션의 핵심 기반입니다.

---

## 정리

| 개념 | 한줄 요약 |
|------|-----------|
| Agent | 스스로 계획하고 도구를 써서 목표를 달성하는 LLM |
| Tool | Agent가 사용할 수 있는 외부 기능 |
| ReAct | Thought → Action → Observation 반복 패턴 |
| Multi-Agent | 여러 Agent가 역할을 나눠 협업 |
| LangGraph | Agent 흐름을 그래프로 제어 |

---

## 이전 단계

← [3단계: RAG](./03_RAG.md)
