from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss, os, torch
import gradio as gr

# 모델 로드
model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# 외부 문서 로딩
docs_folder = "data/fan_docs"
docs = [open(os.path.join(docs_folder,f), "r", encoding="utf-8").read() for f in os.listdir(docs_folder)]

# FAISS 벡터 DB
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
doc_embeddings = embed_model.encode(docs).astype("float32")
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

def rag_answer(question, top_k=2, max_length=50):
    q_emb = embed_model.encode([question]).astype("float32")
    D, I = index.search(q_emb, top_k)
    context = " ".join([docs[i] for i in I[0]])
    input_text = f"Context: {context}\n질문: {question}\n답변:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=rag_answer, inputs="text", outputs="text",
                     title="IdolFan RAG Chatbot",
                     description="팬 질문에 맞춰 외부 문서 기반으로 답변하는 RAG 챗봇")
iface.launch()
