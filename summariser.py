# summariser.py
from transformers import pipeline, AutoTokenizer
import streamlit as st
import re

@st.cache_resource(show_spinner=False)
def get_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def split_sections(text):
    headers = ["title", "background", "objective", "methods", "methodology", "study design", "participants", "results", "discussion", "interpretation", "conclusion"]
    section_dict = {}
    current_header = "title"
    section_dict[current_header] = []

    for line in text.split("\n"):
        line_clean = line.strip().lower()
        if any(h in line_clean for h in headers):
            current_header = next(h for h in headers if h in line_clean)
            section_dict[current_header] = []
        section_dict[current_header].append(line.strip())

    return {k: " ".join(v) for k, v in section_dict.items() if v}

def summarize_chunk(text, model, max_len=512):
    try:
        return model(text, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
    except Exception as e:
        return f"[Error summarizing section: {e}]"

def generate_final_summary(full_text):
    tokenizer = get_tokenizer()
    model = get_model()

    sections = split_sections(full_text)
    full_summary = ""
    progress = st.progress(0)

    for i, (title, content) in enumerate(sections.items()):
        if not content.strip():
            continue

        tokens = tokenizer(content, return_tensors="pt", truncation=False)["input_ids"][0]
        if len(tokens) > 1024:
            chunks = [tokens[j:j+900] for j in range(0, len(tokens), 900)]
            chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
            chunk_summaries = [summarize_chunk(chunk, model) for chunk in chunk_texts]
            summary_text = " ".join(chunk_summaries)
        else:
            summary_text = summarize_chunk(content, model)

        full_summary += f"\n**{title.capitalize()}**\n{summary_text}\n"
        progress.progress((i + 1) / len(sections))

    progress.empty()
    return full_summary.strip()

