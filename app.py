import streamlit as st
import pdfplumber
import requests
from io import BytesIO
from summariser import generate_final_summary

# IMPORTANT: set_page_config must be the very first Streamlit command
st.set_page_config(page_title="BioGPT Fast Summarizer", page_icon="🧠")

st.title("🧠 BioGPT: Fast Research Paper Summarizer")
st.markdown(
    "Upload a PDF or paste a direct .pdf link to summarize a biomedical research paper — no API key needed!"
)

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
url = st.text_input("🌐 Or paste a direct .pdf URL (e.g., from bioRxiv)")

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

elif url and url.endswith(".pdf"):
    try:
        response = requests.get(url)
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
            "⚠️ The extracted text is very short. This may be due to a scanned PDF or poor formatting."
        )

    st.subheader("📑 Extracted Text Preview")
    st.text_area("First part of the paper:", full_text[:2000], height=300)

    if st.button("🧠 Summarize"):
        with st.spinner("Generating summary... This may take a few seconds."):
            summary = generate_final_summary(full_text)
            st.subheader("✅ Detailed Summary")
            st.markdown(summary)
            st.caption(
                "⚠️ This is an AI-generated summary. Please verify with the original article before citing."
            )
else:
    st.info("Please upload a PDF or paste a direct .pdf URL to begin.")
