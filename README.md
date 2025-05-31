# BioGPT: Research Paper Summarizer

> A free and open-source research paper summarizer built with Hugging Face Transformers and Streamlit — No API keys required!

## Overview

**BioGPT** is a web app that takes research PDFs (or direct `.pdf` links) and generates **clean, structured summaries** for each section like Abstract, Introduction, Methods, and Conclusion.

Unlike typical summarizers that just chop up the full text, BioGPT:
- Extracts sections intelligently,
- Breaks long content into manageable chunks,
- Summarizes each chunk with `facebook/bart-large-cnn`,
- Removes duplicates & clutter,
- And outputs highly readable, concise text.

Ideal for researchers, students, and professionals who want to **grasp papers faster**.

---

##  Key Features

- **Upload PDFs** or paste `.pdf` **URLs** (bioRxiv, medRxiv, etc.)
- **Hugging Face summarization** using `facebook/bart-large-cnn`
- Section-wise extraction: `Abstract`, `Introduction`, `Results`, etc.
- Duplicate removal and text cleanup
- Deployed using **Streamlit Cloud**
- Fully free — No API keys or signup needed!

---

## Demo

🔗 **Live App:** https://biogpt-summarizer-rxqvfymykdg6wo2uz7nkpp.streamlit.app/

---

## 🛠Tech Stack

- `Python 3.10+`
- `Streamlit`
- `Hugging Face Transformers`
- `pdfplumber`
- `torch`
- `re`, `textwrap`, `os`, `requests`, `tempfile`

---

## How It Works

1. **PDF Extraction:**  
   Uses `pdfplumber` to extract raw text from uploaded files or URLs.

2. **Section Detection:**  
   Uses regex-based heuristics to split text into logical sections (abstract, intro, etc.).

3. **Chunking:**  
   Each section is broken into 700–750-word chunks to fit model input limits.

4. **Summarization:**  
   Each chunk is passed through `facebook/bart-large-cnn` and outputs are aggregated.

5. **Post-processing:**  
   Final summaries are cleaned, stripped of duplicate lines, and shown to the user.

---


