# IdolFan LLM Chatbot: Korean Entertainment Domain

## Overview
Fine-tuned open-source LLM to emulate a specific idol's personality and speech style for fan interactions in Korean.
Covers end-to-end workflow: dataset design, model fine-tuning, evaluation, and deployment.

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

## Project workflow
1. **Environment Setup**
- Libraries: transformers. torch, datasets, gradio, peft
- GPU check
- [Open Day1 Colab Notebook](notebooks/00_EnvironmentSetup.ipynb)

2. **Dataset pipline**
- Collect and preprocess idol SNS posts, interviews, and fan Q&A
- Structure: fan question -> idol-style answer
- [Open Day2 Colab Notebook](notebooks/01_DatasetPipeline.ipynb)

## Sample Dataset
- Located in 'data/fan_qa_samples.json'
- JSON format: '{"prompt": fan_question, "completion": idol_answer}'

## Outputs
- Screenshots and sample outputs in 'outputs/'
