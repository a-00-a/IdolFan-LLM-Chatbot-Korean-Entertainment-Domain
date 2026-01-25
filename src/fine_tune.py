# LoRA fine-tuning for Idol Fan Chatbot

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoRAConfig, get_peft_model

# sample dataset
sample_data = [
    {"prompt": "오늘 기분 어때요?", "completion": "팬들 생각하면서 힘냈어요!"},
    {"prompt": "추천 노래 있어요?", "completion": "제 최애 노래는 'Shakira-Zoo'예요!"},
    {"prompt": "오늘 뭐했어요?", "completion": "새로운 앨범 춤 연습했어요!"},
    {"prompt": "최근 좋아하는 영화는?", "completion": "최근에는 'Inception'봤는데 재밌었어요."},
    {"prompt": "팬들에게 한마디?", "completion": "항상 사랑해요!"},
    {"prompt": "새로운 앨범 언제 나오나요?", "completion": "조금만 더 기다려주세요!"},
    {"prompt": "운동도 하나요?", "completion": "네, 건강하게 유지하려고 해요."},
    {"prompt": "오늘 날씨 어때요?", "completion": "오늘 많이 춥네요."},
    {"prompt": "좋아하는 음식은?", "completion": "초밥 좋아해요!"},
    {"prompt": "휴식 시간에는 뭐해요?", "completion": "책 읽거나 음악 들어요."},
    {"prompt": "팬들 질문 많이 받았나요?", "completion": "네, 항상 감사하게 받아요."},
    {"prompt": "최근 목표는?", "completion": "더 좋은 음악 만들기!"},
    {"prompt": "노래 연습 어떻게 하나요?", "completion": "매일매일 꾸준히 연습해요."},
    {"prompt": "팬들과 소통 방법?", "completion": "인스타랑 bubble로 소통해요!"},
    {"prompt": "좋아하는 운동?", "completion": "요가랑 가벼운 러닝 좋아해요."},
    {"prompt": "가장 기억에 남는 순간?", "completion": "저번 콘서트에서 팬들과 노래부른 순간이 감동이였어요!"},
    {"prompt": "추천하는 책?", "completion": "'Harry Potter' 시리즈 좋아해요."},
    {"prompt": "스트레스 해소 방법?", "completion": "엽떡먹기!"},
    {"prompt": "최근 관심 있는 것?", "completion": "젤리에 푹 빠져서 포도맛 젤리!"},
    {"prompt": "팬들에게 전하고 싶은 말?", "completion": "늘 함께 해줘서 고마워요!"},
]

# Tokenizer 
model_name = "skt/kogpt2-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # padding token 정의

# tokenization
def tokenize(example):
    text = example["prompt"] + " " + example["completion"]
    
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # labels 생성 
    labels = tokenized["input_ids"].copy()

    # pad_token_id는 -100으로 바꾸기 (Loss 무시)
    labels = [
        token if token != tokenizer.pad_token_type_id else -100
        for token in labels
    ]

    tokenized["labels"] = labels
    return tokenized

# Dataset 생성
def prepare_dataset():
    dataset = Dataset.from_list(sample_data)
    dataset = dataset.map(tokenize, batched=False)
    dataset = tokenized_dataset.remove_columns(["prompt", "completion"])
    dataset.set_format("torch")
    return dataset
    
#모델 로드 + LoRA 설정
def prepare_model():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
return model

# Training 설정 (wandb off)
def train():
    dataset = prepare_dataset()
    model = prepare_model()
    
    training_args = TrainingArguments(
        output_dir="./idolfan_lora",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=5e-4,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

# 학습 시작
trainer.train()

model.save_pretrained("./idolfan_lora")
tokenizer.save_pretrained(",/idolfan_lora")

if __name__ == "__main__":
    train()
