import streamlit as st
import pdfplumber
import requests
from io import BytesIO
from summariser import generate_final_summary

st.set_page_config(page_title="BioGPT Free Summarizer", page_icon="🧠")

st.title("🧠 BioGPT: Free Research Paper Summarizer")
st.markdown(
    """
Upload a PDF or paste a direct .pdf link to summarize **any** biomedical or scientific research paper — no API key needed!

**Note:** Best with text-based PDFs. Scanned PDFs may not extract well.
"""
)

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
url = st.text_input("🌐 Or paste a direct .pdf URL (must end with .pdf)")

full_text = ""

if uploaded_file:
    try:
        with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        st.error(f"❌ Failed to read PDF: {e}")

elif url and url.strip().lower().endswith(".pdf"):
    try:
        response = requests.get(url.strip())
        if response.status_code == 200:
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
        else:
            st.error("❌ Failed to download PDF. Check the URL.")
    except Exception as e:
        st.error(f"❌ Error downloading PDF: {e}")

elif url and not url.endswith(".pdf"):
    st.warning("⚠️ Please provide a direct PDF link (ends with .pdf).")

if full_text.strip():
    if len(full_text.strip()) < 1000:
        st.warning(
            "⚠️ The extracted text is very short. This may be due to scanned PDF or poor formatting."
        )

    st.subheader("📑 Extracted Text Preview (first 2000 characters)")
    st.text_area("Preview of extracted paper text:", full_text[:2000], height=300)

    if st.button("🧠 Summarize"):
        with st.spinner("Generating detailed summary... This may take a minute."):
            summary = generate_final_summary(full_text)
            st.subheader("✅ Detailed Summary")
            st.markdown(summary)
            st.caption("⚠️ This AI-generated summary may omit technical details. Always verify with the original paper.")
else:
    st.info("Please upload a PDF or paste a direct .pdf URL to begin.")

