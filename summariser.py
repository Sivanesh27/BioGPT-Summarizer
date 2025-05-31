from transformers import pipeline, AutoTokenizer
import streamlit as st
import re

@st.cache_resource(show_spinner=False)
def get_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

def split_by_sections(text):
    """Naive section splitter for scientific papers"""
    section_titles = [
        "abstract", "background", "introduction", "methods", "materials and methods",
        "results", "discussion", "conclusion"
    ]
    sections = {}
    current = "preamble"
    sections[current] = []

    for line in text.splitlines():
        clean_line = line.strip().lower()
        if clean_line in section_titles:
            current = clean_line
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line.strip())

    return {k: "\n".join(v).strip() for k, v in sections.items() if v and len(" ".join(v)) > 100}

def summarize_section(title, content, model):
    try:
        result = model(content, max_length=200, min_length=60, do_sample=False)
        summary = result[0]["summary_text"]
        return f"### {title.capitalize()}\n{summary.strip()}\n"
    except Exception as e:
        return f"### {title.capitalize()}\n[Error summarizing this section: {e}]\n"

def generate_final_summary(text):
    tokenizer = get_tokenizer()
    summarizer = get_model()
    sections = split_by_sections(text)

    structured_summary = ""
    total = len(sections)
    progress = st.progress(0)

    for i, (section_title, section_content) in enumerate(sections.items()):
        # Token-aware chunking for long sections
        tokens = tokenizer(section_content, return_tensors="pt", truncation=False)["input_ids"][0]
        if len(tokens) > 1024:
            chunks = [tokens[j:j+900] for j in range(0, len(tokens), 900)]
            chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
            combined_summary = " ".join(
                summarizer(c, max_length=160, min_length=40, do_sample=False)[0]["summary_text"]
                for c in chunk_texts
            )
            summary = summarize_section(section_title, combined_summary, summarizer)
        else:
            summary = summarize_section(section_title, section_content, summarizer)

        structured_summary += summary + "\n"
        progress.progress((i + 1) / total)

    progress.empty()
    return structured_summary.strip()
st.caption("⚠️ This structured summary is generated using AI and may omit technical details. Always verify with the full paper before citing.")


