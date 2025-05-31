from transformers import pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_summarizer():
    # Use smaller, faster model for quicker summaries
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    summarizer = get_summarizer()
    max_chunk = 4000  # bigger chunks, fewer calls
    text_chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    summary = ""
    progress_bar = st.progress(0)
    for i, chunk in enumerate(text_chunks):
        result = summarizer(chunk, max_length=100, min_length=20, do_sample=False)
        summary += result[0]['summary_text'] + " "
        progress_bar.progress((i + 1) / len(text_chunks))
    progress_bar.empty()
    return summary.strip()

