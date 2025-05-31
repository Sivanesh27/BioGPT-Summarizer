from transformers import pipeline, AutoTokenizer
import streamlit as st
import re

@st.cache_resource(show_spinner=False)
def get_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def clean_text(text):
    # Remove weird unicode artifacts, extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()

def chunk_text(text, tokenizer, max_tokens=1024):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        if len(chunk) > 10:  # Skip tiny junk chunks
            chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

def summarize_text(text, summarizer):
    try:
        if len(text.strip()) < 30:
            return ""
        return summarizer(text, max_length=250, min_length=60, do_sample=False)[0]["summary_text"]
    except Exception as e:
        return f"[Error summarizing section: {e}]"

def split_sections(text):
    section_pattern = re.compile(r'^\s*(abstract|introduction|background|methods?|materials?|results?|discussion|conclusion|references?)\s*$', re.I)
    sections = {}
    current_section = "Full Text"
    sections[current_section] = []

    for line in text.splitlines():
        if section_pattern.match(line.strip()):
            current_section = line.strip().title()
            sections[current_section] = []
        else:
            sections[current_section].append(line.strip())

    return {
        k: clean_text(" ".join(v)) for k, v in sections.items()
        if len(" ".join(v)) > 100
    }

def format_structured_summary(sections):
    output = "✅ **Detailed Summary**\n\n"
    for title, summary in sections.items():
        if summary and "Error summarizing" not in summary:
            output += f"### {title}\n{summary.strip()}\n\n"
    return output.strip()

def generate_final_summary(full_text):
    tokenizer = get_tokenizer()
    summarizer = get_model()
    full_text = clean_text(full_text)
    sections = split_sections(full_text)

    if not sections:
        st.warning("⚠️ No clear sections detected. Summarizing full document...")
        chunks = chunk_text(full_text, tokenizer)
        summaries = [summarize_text(chunk, summarizer) for chunk in chunks]
        combined = " ".join([s for s in summaries if s])
        return format_structured_summary({"Full Summary": combined})

    summary_results = {}
    progress = st.progress(0)
    for i, (title, content) in enumerate(sections.items()):
        chunks = chunk_text(content, tokenizer)
        summaries = [summarize_text(chunk, summarizer) for chunk in chunks]
        joined_summary = " ".join([s for s in summaries if s])
        summary_results[title] = joined_summary
        progress.progress((i + 1) / len(sections))
    progress.empty()

    return format_structured_summary(summary_results)

