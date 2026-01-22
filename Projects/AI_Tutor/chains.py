from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# 2. 문제 생성 체인 (Question Generation)
gen_system = """당신은 Python 코딩 튜터입니다.
주어진 '주제(Topic)'와 '난이도(Difficulty)'에 맞는 코딩 퀴즈를 하나 만들어주세요.
문제는 간결해야 하며, 정답 코드를 포함하지 마세요. 개념을 묻거나 짧은 코딩 과제를 내세요."""

gen_prompt = ChatPromptTemplate.from_messages([
    ("system", gen_system),
    ("human", "주제: {topic}\n난이도: {difficulty}\n\n새로운 문제를 하나 출제해줘.")
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
