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
    # Remove excessive newlines and whitespace
    return re.sub(r'\n+', '\n', text).strip()

def split_sections(text):
    """
    Split the full text into sections based on typical research paper headers.
    Returns a dictionary: {Section Title: Section Text}
    """
    # Common headers (can add more as needed)
    headers = [
        "title", "abstract", "background", "objective", "objectives",
        "methods", "methodology", "study design", "participants",
        "results", "discussion", "interpretation", "conclusion", "references"
    ]
    pattern = re.compile(r'^\s*(%s)\s*[:\n]' % '|'.join(headers), re.IGNORECASE | re.MULTILINE)

    matches = list(pattern.finditer(text))
    sections = {}

    if not matches:
        # If no headers found, put all in one section
        return {"Full Text": clean_text(text)}

    for i, match in enumerate(matches):
        header = match.group(1).capitalize()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections[header] = clean_text(section_text)

    # Debug print - comment out in production
    # print(f"Detected sections: {list(sections.keys())}")
    # for sec, cont in sections.items():
    #     print(f"Section: {sec}, Length: {len(cont)} chars")

    return sections

def summarize_chunk(text, model):
    if not text or len(text.strip()) < 20:
        return "[Skipped empty or too short chunk]"

    try:
        # Limit input length to 1024 tokens approx by word count (~750 words)
        words = text.split()
        if len(words) > 750:
            text = " ".join(words[:750])

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
            # Skip empty or very short sections
            continue

        words = content.split()
        chunk_size = 700  # safe chunk size to avoid token overflow
        summaries = []

        for j in range(0, len(words), chunk_size):
            chunk = " ".join(words[j:j + chunk_size])
            summary = summarize_chunk(chunk, model)
            summaries.append(summary)

        combined_summary = " ".join(summaries)

        full_summary += f"**{section}**\n\n{combined_summary}\n\n"
        progress.progress((i + 1) / n_sections)

    progress.empty()
    return full_summary.strip()

