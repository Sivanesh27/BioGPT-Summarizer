import streamlit as st
import fitz  # PyMuPDF
import requests
from io import BytesIO
from summarizer import summarize_text

st.set_page_config(page_title="BioGPT Summarizer", page_icon="🧠")

st.title("🧠 BioGPT: Health Research Paper Summarizer")
st.markdown("Upload a PDF or paste a direct .pdf link to summarize a biomedical research paper.")

# Upload or URL input
uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
url = st.text_input("🌐 Or paste a direct .pdf URL (e.g., from bioRxiv)")

# Extract PDF text
pdf = None
full_text = ""

if uploaded_file:
    source = uploaded_file.read()
    pdf = fitz.open(stream=source, filetype="pdf")

elif url and url.endswith(".pdf"):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            pdf = fitz.open(stream=BytesIO(response.content), filetype="pdf")
        else:
            st.error("❌ Failed to download PDF. Check the URL.")
    except Exception as e:
        st.error(f"❌ Error downloading PDF: {e}")

elif url and not url.endswith(".pdf"):
    st.warning("⚠️ Please provide a direct PDF link (e.g., ends with .pdf).")

# Process PDF
if pdf:
    with st.spinner("🔍 Extracting text from PDF..."):
        for page in pdf:
            full_text += page.get_text()

    st.subheader("📑 Extracted Text Preview")
    st.text_area("First part of the paper:", full_text[:2000], height=300)

    if st.button("🧠 Summarize"):
        with st.spinner("Generating summary using GPT..."):
            try:
                summary = summarize_text(full_text[:3500])  # Limit input size
                st.subheader("✅ Summary")
                st.success(summary)
            except Exception as e:
                st.error(f"❌ Failed to summarize: {e}")
else:
    st.info("Please upload a PDF or paste a direct .pdf URL to begin.")
