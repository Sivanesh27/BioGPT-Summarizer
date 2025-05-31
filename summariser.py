from transformers import pipeline, AutoTokenizer
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

def split_by_sections(text):
    """Split paper into main scientific sections with rough matching."""
    section_titles = [
        "abstract",
        "background",
        "introduction",
        "methods",
        "materials and methods",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
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

    # Filter sections: keep only those with significant text
    return {
        k: "\n".join(v).strip()
        for k, v in sections.items()
        if v and len(" ".join(v)) > 100
    }

def summarize_section(title, content, model, tokenizer):
    # Truncate content to max 1000 tokens to speed up summarization
    tokens = tokenizer(content, return_tensors="pt", truncation=True, max_length=1000)[
        "input_ids"
    ][0]
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Use shorter summary lengths to speed up generation
    try:
        result = model(
            truncated_text,
            max_length=120,
            min_length=40,
            do_sample=False,
            truncation=True,
        )
        summary = result[0]["summary_text"]
        return f"**{title.capitalize()}:**\n{summary.strip()}\n"
    except Exception as e:
        return f"**{title.capitalize()}:**\n[Error summarizing this section: {e}]\n"

def generate_final_summary(text):
    tokenizer = get_tokenizer()
    summarizer = get_model()
    sections = split_by_sections(text)

    # Extract title as the first non-empty line in the text (usually paper title)
    title = ""
    for line in text.splitlines():
        if line.strip():
            title = line.strip()
            break

    # Define the sections we want to summarize in detail
    important_sections = [
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "background",
        "materials and methods",
        "conclusions",
    ]

    summary_text = f"# {title}\n\n"

    progress = st.progress(0)
    total = len(important_sections)

    for i, section in enumerate(important_sections):
        content = sections.get(section)
        if content:
            sec_summary = summarize_section(section, content, summarizer, tokenizer)
            summary_text += sec_summary + "\n"
        progress.progress((i + 1) / total)

    progress.empty()

    summary_text += (
        "⚠️ *This structured summary is AI-generated. Verify with the full paper before citing.*"
    )

    return summary_text.strip()



