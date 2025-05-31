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

