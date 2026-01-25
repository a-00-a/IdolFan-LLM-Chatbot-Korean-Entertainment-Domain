# 01_DatasetPipeline.ipynb
from datasets import Dataset
from transformers import AutoTokenizer

# 예시 데이터
sample_data = [
    {"prompt": "오늘 하루 어땠어요?", "completion": "팬들 생각하면서 힘냈어요!"},
    {"prompt": "추천 노래 있어요?", "completion": "제 최애 노래는 'Shakira-Zoo'에요!"}
]

def prepare_dataset(data_list, model_name="skt/kogpt2-base-v2"):
    dataset = Dataset.from_list(data_list)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Add this line to set the padding token

    def tokenize(batch):
        # batch 단위로 처리
        return tokenizer(
            batch["prompt"],
            batch["completion"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["prompt", "completion"])
    tokenized_dataset.set_format("torch")

    return tokenized_dataset, tokenizer

tokenized_dataset, tokenizer = prepare_dataset(sample_data)
print(tokenized_dataset)
