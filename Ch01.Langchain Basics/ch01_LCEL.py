from  langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 오픈AI 대규모 언어 모델 설정
model = ChatOpenAI(model="gpt-4o")
# 프롬프트 템플릿 정의 : 주어진 주제에 대한 설명 요청
prompt = ChatPromptTemplate.from_template("주제 {topic}에 대해 짧은 설명을 해줘")
#출력 파서 정의 : AI 메세지의 출력 내용을 추출
parser = StrOutputParser()

# 프롬프트, 모델, 출력 파서를 체인으로 연결
chain = prompt | model | parser
print(chain.invoke({"topic": "더블딥"}))

# <배치(Batch) 출력>

# 주어진 주제 리스트를 배치로 출력
print(chain.batch([{"topic": "더블딥"}, {"topic": "인플레이션"}])) 

# <스트림(Stream) 출력>'

# 주어진 주제에 대해 스트림 방식으로 출력
for token in chain.stream({"topic": "더블딥"}):
    print(token, end="", flush=True)

# <구성된 체인을 다른 러너블과 결합하기>

# '이 대답을 영어로 번역해주세요'라는 질문을 생성하는 프롬프트 템플릿을 정의

analysis_prompt = ChatPromptTemplate.from_template("이 대답을 영어로 번역해주세요: {answer}")

# 이전에 정의된 체인(chain)을 사용하여 대답을 생성하고, 그 대답을 영어로 번역하도록 프롬프트에 전달한 후, 모델을 통해 결과를 생성하여 최종적으로 문자열로 파싱하는 체인 구성
composed_chain = {"answer": chain} | analysis_prompt | model | parser
print(composed_chain.invoke({"topic": "더블딥"}))

# <람다 함수를 사용한 체인을 통해 구성하기>

# 람다 함수를 사용한 체인 구성
composed_chain_with_lambda = (
    # 이전에 정의된 체인(chain)을 사용하여 데이터를 받아온다
    chain
    # 입력된 데이터를 "answer" 키로 변환하는 람다함수를 적용
    | (lambda input: {"answer": input})
    # "answer" 키의 값을 사용하여 번역 프롬프트에 전달
    | analysis_prompt
    # 모델을 통해 번역된 결과를 생성
    | model
    # 최종적으로 문자열로 파싱
    | parser
)
# "더블딥"에 대한 설명을 영어로 번역하여 출력
print(composed_chain_with_lambda.invoke({"topic": "더블딥"}))

# <`.pipe()`를 통해 체인 구성하기>

# (방법1) 여러 작업을 순차적으로 .pipe를 통해 연결하여 체인 구성하기
composed_chain_with_pipe = (
  # 이전에 정의된 체인(chain)으로 입력된 데이터를 받아옴
  chain
  # 입력된 데이터를 "answer" 키로 변환하는 람다 함수 적용
  .pipe(lambda input: {"answer": input})
  # analysis_prompt를 체인에 연결하여 설명을 영어로 번역하는 작업 추가
  .pipe(analysis_prompt)
  # 모델을 사용해 응답 생성
  .pipe(model)
  # 생성된 응답을 문자열로 파싱
  .pipe(parser)
)
# "더블딥"이라는 주제로 체인을 실행하여 답변 생성
print(composed_chain_with_pipe.invoke({"topic": "더블딥"}))

# (방법2) 좀 더 간단하게 연결하기
composed_chain_with_pipe = chain.pipe(lambda input: {"answer": input}, analysis_prompt, model, parser)
print(composed_chain_with_pipe.invoke({"topic": "더블딥"}))
#<'RunnableParallel'을 이용한 체인 구성>
from langchain_core.runnables import RunnableParallel

#한국어 설명 생성 프롬프트 체인
korean_template = ChatPromptTemplate.from_template("주제 {topic}에 대해 짧은 한국어 설명을 해줘")

korean_chain = korean_template | model | parser
#영어 설명 생성 프롬프트 체인
eng_template = ChatPromptTemplate.from_template("주제 {topic}에 대해 짧은 영어 설명을 해줘")
eng_chain = eng_template | model | parser

#병렬 실행을 위한 RunnableParallel 체인 구성
parallel_chain = RunnableParallel(kor=korean_chain, eng=eng_chain)
# 주제에 대한 한국어 및 영어 설명을 병렬로 생성
result = parallel_chain.invoke({"topic": "더블딥"})
print("한국어",result["kor"])
print("영어",result["eng"])