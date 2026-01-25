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
import json
from datasets import Dataset

data_path = "./data/sample_data.json"
with open(data_path, "r", encoding="utf-8") as f:
    sample_data = json.load(f)

dataset = Data.from_list(sample_data)

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
