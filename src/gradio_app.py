# gradio_app.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# 모델 로드
model_name_or_path = "./idolfan_lora"  # fine_tuen.py에서 저장한 체크포인트
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 챗봇 함수
def chatbot(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio 앱 실행
if __name__ == "__main__":
    gr.Interface(
        fn=chatbot,
        inputs="text",
        outputs="text",
        title="Idol Fan Chatbot",
        description="LoRA fine-tuned idol-style chatbot"
    ).launch()
