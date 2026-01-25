# IdolFan LLM Chatbot: Korean Entertainment Domain

## Overview
Fine-tuned open-source LLM to emulate a specific idol's personality and speech style for fan interactions in Korean.

## Motivation
Fans want to interact with idols in natural dialogue. This project demonstrates end-to-end LLM application in entertainment content.

## Dataset
-Source: Idol SNS posts, interviews, fan Q&A
-Design: Fan questions -> Idol-style responses
-Preprocessing: Text cleaning, tokenization, formatting

## Model & Training
-Base model: small Korean-capable LLM
-Fine-tuning: LoRA / PEFT
-Training setup

## Evaluation
-Baseline vs fine-tuned qualitative comparison
-Example prompts and outputs

## Deployment
-Gradio interface
-Korean language interface examples

## Notes on LLM Trends
-Recent LLM research insights
-Applicability to fan-oriented chat services

## 📂 Day 1 : 환경 설정
- GPU 확인 및 환경 세팅
- transformer, torch, datasets import 및 버전 확인
- Colab Notebook: [Day1] (https://colab.research.google.com/github/a-00-a/LLM_Practice/blob/main/day1_environment_setup.ipynb)

## 📂 Day 2 : 데이터셋 파이프라인
- Hugging Face Dataset 로딩
- 텍스트 정제 및 토크나이징
- Colab Notebook: [Day2] (https://colab.research.google.com/github/a-00-a/LLM_Practice/blob/main/day2_dataset_pipeline.ipynb) 
