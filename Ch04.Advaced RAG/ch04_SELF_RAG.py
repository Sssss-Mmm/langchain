import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = ("./Data/투자설명서.pdf")

loader = PyPDFLoader(file_path)
doc_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap =100)
docs = loader.load_and_split(doc_splitter)

from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

from langchain_community.vectorstores import FAISS

faiss_store = FAISS.from_documents(docs, embedding)

persist_directory = "./DB"
faiss_store.save_local(persist_directory)

vectordb = FAISS.load_local(persist_directory,embeddings=embedding,allow_dangerous_deserialization=True)

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from typing import Literal

class RetrievalResponse(BaseModel):
    Reasoning: str = Field(description="검색의 필요유무를 추론하는 과정(2~3문장 이내)")
    Retrieve: Literal['Yes','No'] = Field(description="검색 필요유무")

retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template = """
주어진 질문에 대해, 외부 문서를 참고하는 것이 더 나은 응답을 생성하는 데 도움이 되는지 판단해주세요. 추론과정을 작성한 뒤, "Yes" 또는 "No"로 답하세요

다음 기준을 참고하세요:
1. 사실적 정보나 복잡한 주제에 대한 상세한 설명을 요구하는 질문의 경우, 검색이 도움이 될 수 있습니다.
2. 개인적인 의견, 창의적인 과제, 또는 간단한 계산의 경우, 일반적으로 검색이 필요하지 않습니다.
3. 잘 알려진 사실에 대해서도, 검색은 때때로 추가적인 맥락이나 검증을 제공할 수 있습니다.

질문: {query}
"""
)
llm = ChatOpenAI(model="gpt-4o",max_tokens=2000, temperature = 0.2)
retrieval_chain = retrieval_prompt | llm.with_structured_output(RetrievalResponse)

class RelevanceResponse(BaseModel):
    Reasoning : str = Field(description="연관문서의 관련성 평가 추론과정(2~3문장 이내)")
    ISREL : Literal['Relevant','Irrelevant']  = Field(description="관련성 평가 결과")

relevance_prompt = PromptTemplate(
    input_variables=["query","context"],
    template="""
당신은 제공된 연관문서가 주어진 질문과 관련이 있는지, 그리고 질문에 답하는 데 유용한 정보를 제공하는지 판단하는 것입니다.
만약 연관문서가 이 요구사항을 충족한다면 "Relevant"로 응답하고, 그렇지 않다면 "Irrelevant"로 응답하세요.

다음 예시들을 참고하세요:

예시 1:
질문: 지구의 자전은 무엇을 야기하나요?
연관문서: 자전은 낮과 밤의 순환을 야기하며, 이는 또한 온도와 습도의 상응하는 순환을 만듭니다. 지구가 자전함에 따라 해수면은 하루에 두 번 상승하고 하강합니다.
Reasoning: 이 관련문서는 지구의 자전이 낮과 밤의 순환을 야기한다고 명시적으로 언급하고 있어, 질문에 직접적으로 관련이 있습니다.
ISREL: Relevant

예시 2:
질문: 미국 하원의원 출마를 위한 나이 제한은 어떻게 되나요?
연관문서: 헌법은 미국 상원 의원직을 위한 세 가지 자격 요건을 설정합니다: 나이(최소 30세), 미국 시민권(최소 9년), 그리고 선거 시점에 해당 상원의원이 대표하는 주의 거주자여야 합니다.
Reasoning: 이 관련문서는 미국 하원이 아닌 상원 의원직에 대한 나이 제한을 논의하고 있어, 주어진 질문과 직접적인 관련이 없습니다.
ISREL: [Irrelevant]

위의 예시들을 참고하여, 다음 질문과 연관문서에 대해 평가해주세요.

질문: {query}
연관문서: {context}
"""
)

llm = ChatOpenAI(model="gpt-4o",max_tokens= 2000, temperature=0.2)
relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceResponse)

class GenerationResponse(BaseModel):
  response: str = Field(description="질문과 연관문서를 바탕으로 생성된 답변")

# 답변 생성 단계 프롬프트 템플릿
generation_prompt = PromptTemplate(
input_variables=["query", "context"],
template="질문 '{query}'와 연관문서 '{context}'를 기반으로 답변을 만들어주세요."
)
# 사용할 LLM 설정
llm = ChatOpenAI(model="gpt-4o", max_tokens=2000, temperature=0.2)
generation_chain = generation_prompt | llm.with_structured_output(GenerationResponse)

class SupportResponse(BaseModel):
    Reasoning: str = Field(description="답변이 연관문서에 충분히 근거하는지 여부를 추론하는 과정(2~3문장 이내)")
    ISSUP: Literal['Fully supported', 'Partially supported', 'No support'] = Field(description="답변이 연관문서에 충분히 근거하는지에 대한 평가결과")

support_prompt = PromptTemplate(
    input_variables=["query", "response", "context"],
    template="""
당신은 주어진 답변이 연관문서의 정보에 얼마나 근거하고 있는지 평가하는 것입니다. 다음 척도를 사용하여 평가해주세요:

1. Fully supported - 답변의 모든 정보가 연관문서에 의해 뒷받침되거나, 연관문서에서 직접 추출된 경우입니다. 이는 답변과 연관문서의 일부가 거의 동일한 극단적인 경우에만 해당합니다.
2. Partially supported - 답변이 어느 정도 연관문서에 의해 뒷받침되지만, 연관문서에서 다루지 않는 주요 정보가 답변에 포함된 경우입니다. 예를 들어, 질문이 두 가지 개념에 대해 물었는데 연관문서가 그 중 하나만 다루고 있다면 이에 해당합니다.
3. No support - 답변이 연관문서를 완전히 무시하거나, 관련이 없거나, 또는 연관문서와 모순되는 경우입니다. 연관문서가 질문과 무관한 경우에도 이에 해당할 수 있습니다.

주의: 답변이 사실인지 아닌지를 판단하기 위해 외부 정보나 지식을 사용하지 마세요. 오직 답변이 연관문서에 의해 뒷받침되는지만 확인하세요. 답변이 질문을 잘 따르고 있는지는 판단하지 않습니다.

다음 예시를 참고하세요:
질문: 자연어 처리에서 단어 임베딩의 사용에 대해 설명해주세요.
답변: 단어 임베딩은 감성 분석, 텍스트 분류, 다음 단어 예측, 동의어와 유추 관계 이해 등의 작업에 유용합니다.
연관문서: 단어 임베딩은 자연어 처리(NLP)에서 어휘의 단어나 구를 실수 벡터에 매핑하는 언어 모델링 및 특징 학습 기술의 총칭입니다. 단어와 구 임베딩은 기본 입력 표현으로 사용될 때 구문 분석, 감성 분석, 다음 토큰 예측, 유추 감지 등의 NLP 작업에서 성능 향상을 보여주었습니다.
Reasoning: 답변에서 언급된 단어 임베딩의 모든 응용 분야(감성 분석, 텍스트 분류, 다음 단어 예측, 동의어와 유추 관계 이해)가 연관문서에서 직접적으로 언급되거나 유추될 수 있습니다. 따라서 답변은 연관문서에 의해 완전히 뒷받침됩니다.
ISSUP: Fully supported

위의 예시를 참고하여, 주어진 질문, 답변, 연관문서에 대한 당신의 평가를 제시해주세요:

질문: {query}
답변: {response}
연관문서: {context}
"""
)
# 각 단계에 대한 LLMChain 생성
support_chain = support_prompt | llm.with_structured_output(SupportResponse)

class UtilityResponse(BaseModel):
    Reasoning: str = Field(description="응답의 유용성 평가 추론과정")
    ISUSE: Literal[1, 2, 3, 4, 5] = Field(description="응답의 유용성 평가결과")

utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="""
주어진 질문과 답변에 대해, 그 응답이 얼마나 도움이 되고 유익한 답변인지 1점(최저)부터 5점(최고)까지 평가해주세요. 이 점수를 'Utility_score'라고 부릅니다.

평가 기준은 다음과 같습니다:
5: 답변이 완벽하고 매우 상세하며 정보가 풍부하여 질문의 정보 요구를 완전히 충족시킵니다.
4: 답변이 대체로 질문의 요구를 충족시키지만, 더 자세한 정보 제공, 응답 구조 개선, 또는 일관성 향상 등의 약간의 개선이 가능합니다.
3: 답변이 수용 가능하지만, 사용자의 요구를 만족시키기 위해 주요한 추가 정보나 개선이 필요합니다.
2: 답변이 주요 요청을 다루고는 있지만, 불완전하거나 질문과 완전히 관련이 없습니다.
1: 답변이 거의 주제와 관련이 없거나 완전히 무관합니다.

다음 예시들을 참고하세요:

예시 1:
질문: 2023년 현재 영국의 총리는 누구인가요?
답변: Boris Johnson은 2019년부터 2022년까지 영국의 총리였습니다.
Reasoning: 이 응답은 2019년부터 2022년까지의 영국 총리에 대해 사실적으로 정확한 진술을 제공하지만, 질문은 2023년 현재의 총리를 묻고 있습니다. 따라서 질문에 직접적으로 답하지 않아 유용성이 2점입니다.
ISUSE: 2

예시 2:
질문: 여행 목적지인 도쿄, 일본에 대한 설명을 바탕으로 10개의 관광 명소를 추천하고 각각에 대해 자세히 설명해주세요.
답변: 도쿄는 흥미진진한 관광 명소로 가득한 활기찬 도시입니다. 꼭 봐야 할 명소로는 도쿄 스카이트리, 도쿄 디즈니랜드, 센소지 사원, 메이지 신궁, 츠키지 어시장, 하라주쿠, 신주쿠 교엔 등이 있습니다.
Reasoning: 이 응답은 각 명소에 대한 설명을 제공하지 않았고, 명소의 수도 10개보다 적습니다. 질문에 부분적으로 답변하고 있지만, 지시사항을 엄격히 따르지 않았습니다.
ISUSE: 3

위의 예시들을 참고하여, 주어진 질문과 응답에 대한 당신의 평가를 제시해주세요:

질문: {query}
응답: {response}
"""
)

# 사용할 LLM 설정
llm = ChatOpenAI(model="gpt-4o", max_tokens=2000, temperature=0.2)
# 각 단계에 대한 LLMChain 생성
utility_chain = utility_prompt | llm.with_structured_output(UtilityResponse)

class SelfRAG:
    def __init__(self, vectorstore, retrieval_chain, relevance_chain, generation_chain, support_chain, utility_chain, top_k):
        self.vectorstore = vectorstore
        self.retrieval_chain = retrieval_chain
        self.relevance_chain = relevance_chain
        self.generation_chain = generation_chain
        self.support_chain = support_chain
        self.utility_chain = utility_chain
        self.top_k = top_k

    def determine_retrieval(self, query):
        print("\n1단계: 검색 필요 여부 결정 중...")
        input_data = {"query": query}
        retrieval_decision_response = self.retrieval_chain.invoke(input_data)
        reasoning = retrieval_decision_response.Reasoning
        retrieve_token = retrieval_decision_response.Retrieve
        print(f"검색 결정 추론과정: {reasoning}")
        print(f"검색 결정: {retrieve_token}")
        return retrieve_token

    def retrieve_documents(self, query):
        print("\n2단계: 관련 문서 검색 중...")
        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        contexts = [doc.page_content for doc in docs]
        print(f"{len(contexts)}개의 문서를 검색했습니다")
        return contexts

    def evaluate_relevance(self, query, contexts):
        print("\n3단계: 문서의 관련성 평가 중...")
        relevant_contexts = []
        for i, context in enumerate(contexts):
            input_data = {"query": query, "context": context}
            relevance_response = self.relevance_chain.invoke(input_data)
            relevance_reasoning = relevance_response.Reasoning
            relevance_token = relevance_response.ISREL
            print(f"문서 {i+1} 관련성 추론과정: {relevance_reasoning}")
            print(f"문서 {i+1} 관련성: {relevance_token}")
            if relevance_token == 'Relevant':
                relevant_contexts.append(context)
        print(f"관련된 컨텍스트 수: {len(relevant_contexts)}")
        return relevant_contexts

    def generate_responses(self, query, relevant_contexts):
        print("\n4단계: 관련 컨텍스트로 응답 생성 중...")
        responses = []
        for i, context in enumerate(relevant_contexts):
            print(f"컨텍스트 {i+1}에 대한 응답 생성 중...")
            input_data = {"query": query, "context": context}
            response = self.generation_chain.invoke(input_data).response
            responses.append(response)
        return responses

    def generate_without_retrieval(self, query):
        input_data = {"query": query, "context": "관련된 컨텍스트를 찾지 못했습니다."}
        response = self.generation_chain.invoke(input_data).response
        return response

    def assess_and_evaluate(self, query, responses, relevant_contexts):
        assessed_responses = []
        for i, (response, context) in enumerate(zip(responses, relevant_contexts)):
            # 5단계: 지원 평가
            print(f"\n5단계: 응답 {i+1}의 지원 평가 중...")
            input_data = {"query":query, "response": response, "context": context}
            support_response = self.support_chain.invoke(input_data)
            support_reasoning = support_response.Reasoning
            support_token = support_response.ISSUP
            print(f"지원 평가 추론과정: {support_reasoning}")
            print(f"지원 평가: {support_token}")

            # 6단계: 유용성 평가
            print(f"\n6단계: 응답 {i+1}의 유용성 평가 중...")
            input_data = {"query": query, "response": response}
            utility_response = self.utility_chain.invoke(input_data)
            utility_reasoning = utility_response.Reasoning
            utility_token = int(utility_response.ISUSE)
            print(f"유용성 점수 평가과정: {utility_reasoning}")
            print(f"유용성 점수: {utility_token}")
            assessed_responses.append((response, support_token, utility_token))
        return assessed_responses

    def select_best_response(self, responses):
        print("\n최고의 응답 선택 중...")

        # 1. fully supported 항목이 있는지 확인
        fully_supported = [r for r in responses if r[1] == 'Fully supported']
        if fully_supported:
            best_response = max(fully_supported, key=lambda x: x[2])
            print(f"선택된 응답의 지원 상태: {best_response[1]}, 유용성 점수: {best_response[2]}")
            return best_response

        # 2. fully supported가 없으면 Partially supported 항목 확인
        partially_supported = [r for r in responses if r[1] == 'Partially supported']
        if partially_supported:
            best_response = max(partially_supported, key=lambda x: x[2])
            print(f"선택된 응답의 지원 상태: {best_response[1]}, 유용성 점수: {best_response[2]}")
            return best_response

        # 3. 둘 다 없는 경우,  유용성점수 기준으로 선택
        best_response = max(responses, key=lambda x: x[2])
        print(f"선택된 응답의 지원 상태: {best_response[1]}, 유용성 점수: {best_response[2]}")
        return best_response


    def process_query(self, query):
        print(f"\n쿼리 처리 중: {query}")

        # 1단계: 검색이 필요한지 결정
        retrieval_decision = self.determine_retrieval(query)

        if retrieval_decision == 'Yes':
            # 2단계: 관련 문서 검색
            contexts = self.retrieve_documents(query)

            # 3단계: 검색된 문서의 관련성 평가
            relevant_contexts = self.evaluate_relevance(query, contexts)

            if not relevant_contexts:
                # 관련된 컨텍스트가 없으면 검색 없이 생성
                print("관련된 컨텍스트를 찾지 못했습니다. 검색 없이 생성합니다...")
                return self.generate_without_retrieval(query)

            # 4단계: 관련 컨텍스트를 사용하여 응답 생성
            responses = self.generate_responses(query, relevant_contexts)

            # 5단계 및 6단계: 지원 평가 및 유용성 평가
            assessed_responses = self.assess_and_evaluate(query, responses, relevant_contexts)

            # 최고의 응답 선택
            best_response = self.select_best_response(assessed_responses)
            return best_response[0]
        else:
            # 검색 없이 생성
            print("검색 없이 생성합니다...")
            return self.generate_without_retrieval(query)
        
self_rag_instance = SelfRAG(
    vectorstore = vectordb,
    retrieval_chain = retrieval_chain,
    relevance_chain = relevance_chain,
    generation_chain = generation_chain,
    support_chain = support_chain,
    utility_chain = utility_chain,
    top_k=4
)

# 쿼리 처리
query = "이 회사의 바이오 의약품 라이센스 아웃 수익을 알려줘"
response = self_rag_instance.process_query(query)

print("\n최종 응답:")
print(response)