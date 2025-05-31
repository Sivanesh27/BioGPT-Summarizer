from transformers import pipeline, AutoTokenizer
import streamlit as st
import re

@st.cache_resource(show_spinner=False)
def get_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

def chunk_text(text, tokenizer, max_tokens=1000):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def clean_text(text):
    # Remove duplicate sentences and unnecessary whitespace
    sentences = list(set(re.split(r'(?<=[.!?]) +', text)))
    sentences = [s.strip().capitalize() for s in sentences if len(s.split()) > 6]
    return " ".join(sentences)

def postprocess_summary(raw_summary):
    text = clean_text(raw_summary)
    # Capitalize first letter and fix spacing
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_final_summary(text):
    summarizer = get_model()
    tokenizer = get_tokenizer()
    chunks = chunk_text(text, tokenizer)

    summaries = []
    progress = st.progress(0)

    for i, chunk in enumerate(chunks):
        try:
            result = summarizer(chunk, max_length=160, min_length=40, do_sample=False)
            summaries.append(result[0]["summary_text"])
        except Exception as e:
            summaries.append("[Skipped faulty chunk]")
            st.warning(f"⚠️ Error summarizing chunk {i+1}: {e}")
        progress.progress((i + 1) / len(chunks))
    progress.empty()

    # Meta-summary to improve cohesion
    combined_summary = " ".join(summaries)
    try:
        final = summarizer(combined_summary, max_length=200, min_length=60, do_sample=False)
        return postprocess_summary(final[0]["summary_text"])
    except Exception as e:
        st.warning("⚠️ Meta-summary failed. Showing combined summary instead.")
        return postprocess_summary(combined_summary)

