from transformers import pipeline, AutoTokenizer
import streamlit as st
import re

@st.cache_resource(show_spinner=False)
def get_model():
    # Using bart-large-cnn for summarization
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def clean_text(text):
    # Remove multiple spaces, newlines, tabs
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove weird footer/header patterns (e.g. page numbers, repeated phrases)
    text = re.sub(r'(?i)(confidential|copyright|all rights reserved|www\.\S+)', '', text)
    return text.strip()

def split_sections(text):
    """
    Split the paper text into sections based on common academic section headers.
    This uses regex to find lines that look like section headers.
    """
    # Typical academic paper section headers - can extend as needed
    section_headers = [
        "abstract", "introduction", "background", "objective", "aim", "methods", "methodology",
        "materials and methods", "study design", "participants", "results", "findings",
        "discussion", "interpretation", "conclusion", "summary", "references", "acknowledgments"
    ]

    # Regex to match section headers (line starting with section name, possibly numbered)
    pattern = re.compile(
        r"^(?:\d{0,2}\.?\s*)?(" + "|".join(section_headers) + r")\s*[:\-]?\s*$",
        re.IGNORECASE | re.MULTILINE
    )

    # Find all headers and their positions
    matches = list(pattern.finditer(text))
    sections = {}

    if not matches:
        # No headers found — treat whole text as one section "Full Text"
        return {"Full Text": clean_text(text)}

    for i, match in enumerate(matches):
        header = match.group(1).strip().capitalize()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections[header] = clean_text(section_text)

    return sections

def summarize_chunk(text, model, max_len=512):
    """
    Summarizes a chunk of text.
    Model max_length is set to 512 tokens max here.
    """
    try:
        # Huggingface pipeline expects str, truncate long inputs if needed
        max_input_len = 1024
        if len(text.split()) > max_input_len:
            # Truncate to max_input_len words (approximate)
            text = " ".join(text.split()[:max_input_len])
        summary = model(text, max_length=150, min_length=80, do_sample=False)[0]['summary_text']
        return summary.strip()
    except Exception as e:
        return f"[Error summarizing section: {e}]"

def generate_final_summary(full_text):
    tokenizer = get_tokenizer()
    model = get_model()

    sections = split_sections(full_text)
    full_summary = f"### Paper Summary\n\n"

    progress = st.progress(0)
    n_sections = len(sections)
    for i, (section, content) in enumerate(sections.items()):
        if not content or len(content) < 50:
            continue

        # For long sections, split into ~900 token chunks for summarization
        words = content.split()
        chunk_size = 900
        summaries = []
        for j in range(0, len(words), chunk_size):
            chunk = " ".join(words[j:j+chunk_size])
            summary = summarize_chunk(chunk, model)
            summaries.append(summary)

        combined_summary = " ".join(summaries)

        full_summary += f"**{section}**\n\n{combined_summary}\n\n"
        progress.progress((i + 1) / n_sections)

    progress.empty()
    return full_summary.strip()

    progress.empty()
    return full_summary.strip()

