from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# 허깅페이스 허브에서 데이터셋 로드
dataset = load_dataset("iamjoon/klue-mrc-ko-rag-dataset", split="train")

# system_message 정의
system_message = """당신은 검색 결과를 바탕으로 질문에 답변해야 합니다.

다음의 지시사항을 따르십시오.
1. 질문과 검색 결과를 바탕으로 답변하십시오.
2. 검색 결과에 없는 내용을 답변하려고 하지 마십시오.
3. 질문에 대한 답이 검색 결과에 없다면 검색 결과에는 "해당 질문~에 대한 내용이 없습니다."라고 답변하십시오.
4. 답변할 때 특정 문서를 참고하여 문장 또는 문단을 작성했다면 뒤에 출처는 이중 리스트로 해당 문서 번호를 남기십시오. 예를 들어 특정 문장이나 문단을 1번 문서에서 인용했다면 뒤에 [[ref1]]이라고 기재하십시오.
5. 예를 들어 특정 문장이나 문단을 1번 문서와 5번 문서에서 동시에 인용했다면 뒤에 [[ref1]], [[ref5]]라고 기재하십시오.
6. 최대한 다수의 문서를 인용하여 답변하십시오.

검색 결과:
-----
{search_result}"""

# 원본 데이터의 type별 분포 출력
print("원본 데이터의 type 분포:")
for type_name in set(dataset['type']):
    print(f"{type_name}: {dataset['type'].count(type_name)}")

# train/test 분할 비율 설정
test_ratio = 0.8

train_data = []
test_data = []

# type별로 순화하면서 train/test 데이터 분할
for type_name in set(dataset['type']):
    # 현재 type에 해당하는 데이터의 인덱스만 추출
    curr_type_data = [i for i in range(len(dataset)) if dataset[i]['type']== type_name]

    # test_ratio에 따라 test 데이터 개수 계산
    test_size = int(len(curr_type_data) * test_ratio)

    # 현재 type의 데이터를 test_ratio 비율로 분할하여 추가
    test_data.extend(curr_type_data[:test_size])
    train_data.extend(curr_type_data[test_size:])

# OpenAi format으로 데이터를 변환하기 위한 함수
def format_data(sample):
    # 검색 결과를 문서1, 문서2 ... 형태로 포매팅
    search_result = "\n------\n".join([f"문서{idx + 1}: {result}" for idx, result in enumerate(sample["search_result"])])

    # OpenAI format으로 변환
    return {
        "messages": [
            {
                "role": "system",
                "content": system_message.format(search_result=search_result)
            },
            {
                "role": "user",
                "content": sample["question"]
            },
            {
                "role":"assistant",
                "content": sample["answer"]
            }
        ]
    }

# 분활된 데이터를 OpenAI format으로 변환
train_dataset = [format_data(dataset[i]) for i in train_data]
test_dataset = [format_data(dataset[i]) for i in test_data]

# 최종 데이터셋 크기 출력
print(f"\n전체 데이터 분할 결과: Train {len(train_dataset)}개, Test {len(test_dataset)}개")


# 분할된 데이터의 type별 분포 출력
print("\n학습 데이터의 type 분포:")
for type_name in set(dataset['type']):
    count = sum(1 for i in train_data if dataset[i]['type']==type_name)
    print(f"{type_name}:{count}")

print("\n테스트 데이터의 type 분포:")
for type_name in set(dataset['type']):
    count = sum(1 for i in test_data if dataset[i]['type'] == type_name)
    print(f"{type_name}:{count}")

# print(train_dataset[345]["messages"])

# 리스트 형태에서 다시 Dataset 객체로 변경
print(type(train_dataset))
print(type(test_dataset))
train_dataset = Dataset.from_list(train_dataset)
test_dataset = Dataset.from_list(test_dataset)
print(type(train_dataset))
print(type(test_dataset))

# 허깅페이스 모델 이름
model_id = "Qwen/Qwen2-7B-Instruct" 

# 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = "auto",
    torch_dtype = torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 템플릿 적용
text = tokenizer.apply_chat_template(
    train_dataset[0]["messages"], tokenize=False , add_generation_prompt=False
)
print(text)