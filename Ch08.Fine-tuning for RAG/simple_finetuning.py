
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ==========================================
# [실행 전 필수 설치 라이브러리]
# 이 코드를 실행하려면 아래 명령어로 무거운 라이브러리들을 설치해야 합니다.
# 터미널에 입력하세요:
# uv add torch transformers peft trl bitsandbytes datasets list-ds accelerate
# ==========================================

def train():
    # 1. 모델과 파라미터 설정
    # 학습 시간이 오래 걸리지 않도록 가벼운 모델(TinyLlama 1.1B)을 선택했습니다.
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    new_model_name = "TinyLlama-Kor-Tuned"
    
    print(f"Loading model: {model_name}...")

    # 4비트 양자화 설정 (GPU 메모리 절약)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # 패딩 토큰 설정
    tokenizer.padding_side = "right"

    # 2. 데이터셋 준비
    # 실습용으로 공개된 한국어 데이터셋을 조금만 가져옵니다.
    print("Loading dataset...")
    # 예: 한국어 질문-답변 데이터셋 (Beomi/KoAlpaca-v1.1a 등)
    # 여기서는 데모를 위해 유명한 영문 데이터셋인 guanaco를 예시로 듭니다.
    # 실제로는 'kor_dataset' 등을 로드하여 포맷을 맞추면 됩니다.
    dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train[:100]") # 100개만 사용

    # 3. LoRA (PEFT) 설정
    # 모델 전체를 학습하면 너무 무거우므로, 일부 파라미터만 학습하는 LoRA 방식을 씁니다.
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64, # Rank: 클수록 많이 학습하지만 메모리 많이 먹음
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. 학습 파라미터 설정 (Hyperparameters)
    training_params = SFTConfig(
        output_dir="./results",
        num_train_epochs=1, # 1회 학습
        per_device_train_batch_size=4, # 배치 크기
        gradient_accumulation_steps=1, # 
        logging_steps=10,
        learning_rate=2e-4,
        max_seq_length=512,
        fp16=True, # GPU 가속 사용
        group_by_length=True,
        packing=False, # True면 여러 문장을 하나로 묶음
        report_to="none" # wandb 같은 곳에 리포트 안 함
    )

    # 5. Trainer 생성 및 학습 시작
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text", # 데이터셋에서 텍스트가 들어있는 컬럼명 (guanaco는 'text'임)
        tokenizer=tokenizer,
        args=training_params,
    )

    print("Starting training...")
    trainer.train()
    
    # 6. 저장
    print(f"Saving model to {new_model_name}...")
    trainer.model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)
    print("Training Done!")

if __name__ == "__main__":
    # 라이브러리가 없어서 실행시 에러가 날 수 있으므로 try-except 처리
    try:
        train()
    except ImportError as e:
        print("\n[Error] 필수 라이브러리가 설치되지 않았습니다.")
        print(f"상세 에러: {e}")
        print("터미널에 다음 명령어를 실행하여 설치해주세요:")
        print("uv add torch transformers peft trl bitsandbytes datasets accelerate")
    except Exception as e:
        print(f"\n[Error] 실행 중 오류가 발생했습니다: {e}")
