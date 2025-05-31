from transformers import pipeline
import streamlit as st

@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summarizer = get_summarizer()

    max_chunk = 1000
    text_chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    summary = ""
    for chunk in text_chunks:
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + " "
    return summary.strip()

