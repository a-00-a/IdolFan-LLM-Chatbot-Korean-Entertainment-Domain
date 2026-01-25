# IdolFan LLM Chatbot

A small Korean LLM fine-tuned to emulate an idol's personality and respond to fan questions.
Covers dataset preparation, LoRA fine-tuning, and a simple Gradio chatbot demo.

## Project Structure

## Dataset
- **Source**: Idol SNS posts, interviews, fan Q&A
- **Structure**: '{"prompt": fan_question, "completion": idol_answer}'
- **Location**: './data/sample_data.json'
- Sample entries:

'''json
[
  {"prompt": "오늘 기분 어때요?", "completion": "팬들 생각하면서 힘냈어요!"}
  {"prompt": "추천 노래 있어요?", "completion": "제 최애 노래는 'Shakira-Zoo'예요!"}
]

## Model & Training
- Base model: skt/kogpt2-base-v2 (small Korean GPT-2)
- Fine-tuning: LoRA / PEFT
- Dataset:20 prompt-completion pairs
- Training script: src/fine_tune.py
- Note: Only 1 epoch for demonstration purposes

## Gradio Demo
- Run the chatbot interface: python src/gradioapp.py
- Enter a fan question in Korean -> get idol-style response

## Notes
- Current version is a demo / work-in-progress
- Future improvements: larger dataset, RAG integration, deployment on cloud

## Project workflow
1. **Environment Setup**
- Libraries: transformers. torch, datasets, gradio, peft
- GPU check
- [Open 00 Colab Notebook](notebooks/00_EnvironmentSetup.ipynb)

2. **Dataset pipline**
- Collect and preprocess idol SNS posts, interviews, and fan Q&A
- Structure: fan question -> idol-style answer
- [Open 01 Colab Notebook](notebooks/01_DatasetPipeline.ipynb)

3. **Fine-tuning prep**
- [Open 02 Colab Notebook](notebooks/02_FineTuningPrep.ipynb)

4. **Fine-tuning with Gradio app**
- [Open 03 Colab Notebook](notebooks/03_FineTuning_Gradio.ipynb)
