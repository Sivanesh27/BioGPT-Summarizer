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
    Improved regex to split on lines that look like headers.
    Headers can be uppercase or capitalized, followed by optional colon or newline.
    """
    pattern = re.compile(
        r'^\s*([A-Z][A-Za-z\s]{2,50})(?=\s*$|\n|:)', re.MULTILINE
    )

    matches = list(pattern.finditer(text))
    sections = {}

    if not matches:
        # No headers detected, return whole text as one section
        return {"Full Text": clean_text(text)}

    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections[header] = clean_text(section_text)

    # If no sections found (very rare), fallback
    if not sections:
        return {"Full Text": clean_text(text)}

    return sections

def summarize_chunk(text, model):
    if not text or len(text.strip()) < 20:
        return "[Skipped empty or too short chunk]"

    try:
        words = text.split()
        if len(words) > 750:
            text = " ".join(words[:750])

        summary = model(text, max_length=150, min_length=80, do_sample=False)[0]['summary_text']
        return summary.strip()
    except Exception as e:
        return f"[Error summarizing section: {str(e)}]"

def generate_final_summary(full_text):
    tokenizer = get_tokenizer()
    model = get_model()

    sections = split_sections(full_text)
    full_summary = f"### Paper Summary\n\n"

    progress = st.progress(0)
    n_sections = len(sections)

    for i, (section, content) in enumerate(sections.items()):
        if not content or len(content.strip()) < 50:
            full_summary += f"**{section}**\n\n[Skipped empty or too short section]\n\n"
            progress.progress((i + 1) / n_sections)
            continue

        words = content.split()
        chunk_size = 700
        summaries = []

        for j in range(0, len(words), chunk_size):
            chunk = " ".join(words[j:j + chunk_size])
            # Skip too short chunks to avoid errors
            if len(chunk.strip()) < 20:
                continue
            summary = summarize_chunk(chunk, model)
            summaries.append(summary)

        combined_summary = " ".join(summaries).strip()

        if not combined_summary:
            combined_summary = "[No summary could be generated for this section]"

        full_summary += f"**{section}**\n\n{combined_summary}\n\n"
        progress.progress((i + 1) / n_sections)

    progress.empty()
    return full_summary.strip()
