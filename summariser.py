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
    # Remove excessive whitespace, newlines, and broken links
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def split_sections(text):
    """
    Split text into sections using likely headers.
    Headers are lines with initial capitalization or all-caps followed by a colon or newline.
    """
    pattern = re.compile(r'^\s*([A-Z][A-Za-z\s]{2,50})(?=\s*$|\n|:)', re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections = {}

    if not matches:
        return {"Full Text": clean_text(text)}

    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections[header] = clean_text(section_text)

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

def remove_duplicates(text):
    # Remove repeated lines or phrases
    lines = text.split('. ')
    seen = set()
    unique_lines = []
    for line in lines:
        line_clean = line.strip().lower()
        if line_clean and line_clean not in seen:
            seen.add(line_clean)
            unique_lines.append(line.strip())
    return '. '.join(unique_lines)

def generate_final_summary(full_text):
    tokenizer = get_tokenizer()
    model = get_model()

    # Clean up text early
    full_text = clean_text(full_text)

    # Filter out junk text before sectioning
    garbage_patterns = [
        r"http\S+", r"www\.\S+", r"PMC.*?\n", r"Learn more", r"©.*?\n"
    ]
    for pattern in garbage_patterns:
        full_text = re.sub(pattern, '', full_text, flags=re.IGNORECASE)

    sections = split_sections(full_text)

    # Exclude sections that shouldn't be summarized
    excluded_sections = {
        "Keywords", "References", "Footnotes", "Conflict of interest",
        "Acknowledgment", "Acknowledgement", "Foundation Project",
        "Notes Comments", "Learn more", "Background", "Funding", "Sources", "Support"
    }
    sections = {k: v for k, v in sections.items() if k.strip() not in excluded_sections}

    if not sections:
        return "⚠️ Could not extract valid sections to summarize."

    full_summary = "### 📄 Detailed Summary\n\n"
    short_summary_parts = []

    progress = st.progress(0)
    n_sections = len(sections)

    for i, (section, content) in enumerate(sections.items()):
        try:
            if not content or len(content.strip()) < 50:
                full_summary += f"**{section}**\n\n[Skipped empty or too short section]\n\n"
                progress.progress((i + 1) / n_sections)
                continue

            words = content.split()
            chunk_size = 700
            summaries = []

            for j in range(0, len(words), chunk_size):
                chunk = " ".join(words[j:j + chunk_size])
                if len(chunk.strip()) < 20:
                    continue
                summary = summarize_chunk(chunk, model)
                summaries.append(summary)

            combined_summary = " ".join(summaries).strip()
            combined_summary = remove_duplicates(combined_summary)

            if not combined_summary:
                combined_summary = "[No summary could be generated for this section]"

            full_summary += f"**{section}**\n\n{combined_summary}\n\n"
            short_summary_parts.append(combined_summary)

        except Exception as e:
            full_summary += f"**{section}**\n\n[Error summarizing section: {str(e)}]\n\n"
        
        progress.progress((i + 1) / n_sections)

    progress.empty()

    # Generate concise summary using all cleaned section summaries
    try:
        short_summary_input = " ".join(short_summary_parts)
        short_summary = summarize_chunk(short_summary_input, model)
        short_summary = remove_duplicates(short_summary)
    except Exception:
        short_summary = "[Could not generate concise summary]"

    return f"### 🧠 Concise Summary\n\n{short_summary}\n\n---\n\n" + full_summary.strip()

