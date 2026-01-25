## 00_EnvironmentSetup.ipynb

# 라이브러리 설치
!pip install transformers datasets torch peft gradio --quiet

# GPU 확인
import torch

def check_gpu():
    available = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if available else "None"
    print(f"GPU available: {available}, GPU name: {name}")
    return available, name

check_gpu()
