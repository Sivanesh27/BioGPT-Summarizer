from transformers import pipeline, AutoTokenizer
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def chunk_text(text, tokenizer, max_tokens=1024):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    if len(inputs) <= max_tokens:
        return [text]
    chunks = [inputs[i:i+max_tokens] for i in range(0, len(inputs), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarize_text(text, summarizer):
    try:
        if not text.strip():
            return "[Skipped empty section]"
        return summarizer(text, max_length=250, min_length=60, do_sample=False)[0]["summary_text"]
    except Exception as e:
        return f"[Error summarizing section: {e}]"

def split_sections(text):
    # Improved section detection (case insensitive, flexible spacing)
    import re
    section_pattern = re.compile(r'^\s*(abstract|introduction|background|methods?|results?|discussion|conclusion|references?)\s*$', re.I)
    sections = {}
    current_section = "Full Text"
    sections[current_section] = []

    for line in text.splitlines():
        if section_pattern.match(line.strip()):
            current_section = line.strip().title()
            sections[current_section] = []
        else:
            sections[current_section].append(line.strip())

    # Filter short content sections
    return {
        k: "\n".join(v).strip() for k, v in sections.items()
        if len(" ".join(v)) > 100
    }

def format_structured_summary(sections):
    output = "✅ **Detailed Summary**\n\n"
    for title, summary in sections.items():
        output += f"### {title}\n{summary.strip()}\n\n"
    return output.strip()

def generate_final_summary(full_text):
    tokenizer = get_tokenizer()
    summarizer = get_model()
    sections = split_sections(full_text)

    # If no sections detected or all are too short, summarize the full text
    if not sections:
        st.warning("⚠️ No clear sections detected. Summarizing full document...")
        chunks = chunk_text(full_text, tokenizer)
        summary = "\n\n".join(summarize_text(chunk, summarizer) for chunk in chunks if chunk.strip())
        return format_structured_summary({"Full Summary": summary})

    # Summarize section by section
    summary_results = {}
    progress = st.progress(0)
    total = len(sections)

    for i, (title, content) in enumerate(sections.items()):
        chunked = chunk_text(content, tokenizer)
        summaries = [summarize_text(chunk, summarizer) for chunk in chunked if chunk.strip()]
        summary_results[title] = " ".join(summaries)
        progress.progress((i + 1) / total)

    progress.empty()
    return format_structured_summary(summary_results)
