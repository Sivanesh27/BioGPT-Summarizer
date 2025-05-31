from transformers import pipeline
import streamlit as st
from transformers import AutoTokenizer

@st.cache_resource(show_spinner=False)
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

def chunk_text(text, tokenizer, max_tokens=1000):
    inputs = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = inputs['input_ids'][0]
    
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def summarize_text(text):
    summarizer = get_summarizer()
    tokenizer = get_tokenizer()
    text_chunks = chunk_text(text, tokenizer)

    summary = ""
    progress_bar = st.progress(0)
    for i, chunk in enumerate(text_chunks):
        try:
            result = summarizer(chunk, max_length=100, min_length=20, do_sample=False)
            summary += result[0]['summary_text'] + " "
        except Exception as e:
            summary += f"[Error summarizing chunk {i+1}] "
            st.warning(f"⚠️ Skipped chunk {i+1} due to error: {e}")
        progress_bar.progress((i + 1) / len(text_chunks))
    progress_bar.empty()
    return summary.strip()

