import gradio as gr

def launch_chatbot(model, tokenizer, max_length=50):
    def chatbot_response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text",
                         title="IdolFan Chatbot",
                         description="팬 질문에 맞춰 아이돌 말투로 대답하는 챗봇")
    iface.launch()
  
