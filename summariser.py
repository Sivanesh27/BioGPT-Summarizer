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
    chunks = [inputs[i:i+max_tokens] for i in range(0, len(inputs), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def format_structured_summary(sections):
    output = "✅ **Detailed Summary**\n\n"
    for title, summary in sections.items():
        output += f"### {title}\n{summary.strip()}\n\n"
    return output.strip()

def split_sections(text):
    titles = [
        "title", "abstract", "background", "introduction", "methods",
        "materials and methods", "results", "discussion", "conclusion", "references"
    ]
    sections = {}
    current = "Title"
    sections[current] = []

    for line in text.splitlines():
        clean = line.strip()
        if clean.lower() in titles:
            current = clean.capitalize()
            sections[current] = []
        else:
            sections.setdefault(current, []).append(clean)

    return {k: "\n".join(v).strip() for k, v in sections.items() if len(" ".join(v)) > 100}

def summarize_text(text, summarizer):
    try:
        return summarizer(text, max_length=250, min_length=60, do_sample=False)[0]["summary_text"]
    except Exception as e:
        return f"[Error summarizing section: {e}]"

def generate_final_summary(full_text):
    tokenizer = get_tokenizer()
    summarizer = get_model()
    sections = split_sections(full_text)

    summary_results = {}
    progress = st.progress(0)
    total = len(sections)

    for i, (title, content) in enumerate(sections.items()):
        chunked = chunk_text(content, tokenizer)
        combined = " ".join(summarize_text(chunk, summarizer) for chunk in chunked)
        summary_results[title] = combined
        progress.progress((i + 1) / total)

    progress.empty()
    return format_structured_summary(summary_results)



