from transformers import pipeline, AutoTokenizer
import streamlit as st

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

def generate_final_summary(text):
    tokenizer = get_tokenizer()
    summarizer = get_model()
    sections = split_by_sections(text)

    # Try to extract title: first non-empty line
    title = ""
    for line in text.splitlines():
        if line.strip():
            title = line.strip()
            break

    structured_summary = "✅ Detailed Summary\n"
    structured_summary += f"Title:\n{title}\n\n"

    # Map original sections to nicer headings
    section_name_map = {
        "abstract": "Background",
        "background": "Background",
        "introduction": "Background",
        "methods": "Study Design and Methodology",
        "materials and methods": "Study Design and Methodology",
        "results": "Key Results",
        "discussion": "Interpretation",
        "conclusion": "Conclusion",
        # You can add "objective" or others if needed
    }

    total = len(sections)
    progress = st.progress(0)

    for i, (section_title, section_content) in enumerate(sections.items()):
        display_title = section_name_map.get(section_title, section_title.capitalize())

        tokens = tokenizer(section_content, return_tensors="pt", truncation=False)["input_ids"][0]
        if len(tokens) > 1024:
            # chunk and summarize each chunk, then combine
            chunks = [tokens[j:j+900] for j in range(0, len(tokens), 900)]
            chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
            combined_summary = " ".join(
                summarizer(c, max_length=160, min_length=40, do_sample=False)[0]["summary_text"]
                for c in chunk_texts
            )
            summary = combined_summary.strip()
        else:
            summary = summarizer(section_content, max_length=200, min_length=60, do_sample=False)[0]["summary_text"].strip()

        structured_summary += f"{display_title}:\n{summary}\n\n"
        progress.progress((i + 1) / total)

    progress.empty()
    return structured_summary.strip()

st.caption("⚠️ This structured summary is generated using AI and may omit technical details. Always verify with the full paper before citing.")


