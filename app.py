import streamlit as st
import pdfplumber
import requests
from io import BytesIO
from summarise import summarize_text

st.set_page_config(page_title="BioGPT Free Summarizer", page_icon="🧠")

st.title("🧠 BioGPT: Free Research Paper Summarizer with HuggingFace")
st.markdown("Upload a PDF or paste a direct .pdf link to summarize a biomedical research paper — no API key needed!")

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
url = st.text_input("🌐 Or paste a direct .pdf URL (e.g., from bioRxiv)")

pdf = None
full_text = ""

if uploaded_file:
    try:
        with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
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
                        full_text += text
        else:
            st.error("❌ Failed to download PDF. Check the URL.")
    except Exception as e:
        st.error(f"❌ Error downloading PDF: {e}")

elif url and not url.endswith(".pdf"):
    st.warning("⚠️ Please provide a direct PDF link (ends with .pdf).")

if full_text:
    st.subheader("📑 Extracted Text Preview")
    st.text_area("First part of the paper:", full_text[:2000], height=300)

    if st.button("🧠 Summarize"):
        with st.spinner("Generating summary..."):
            summary = summarize_text(full_text)
            st.subheader("✅ Summary")
            st.success(summary)
else:
    st.info("Please upload a PDF or paste a direct .pdf URL to begin.")
