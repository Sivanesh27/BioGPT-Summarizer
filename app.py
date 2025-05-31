import streamlit as st
import pdfplumber
import requests
from io import BytesIO
from summariser import generate_final_summary

# This line MUST be first!
st.set_page_config(page_title="BioGPT Free Summarizer", page_icon="🧠")

st.title("🧠 BioGPT: Free Research Paper Summarizer")
st.markdown("Upload a PDF or paste a direct .pdf link to summarize a biomedical research paper — no API key needed!")

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
url = st.text_input("🌐 Or paste a direct .pdf URL (must end with `.pdf`)")

full_text = ""

# PDF from upload
if uploaded_file:
    try:
        with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        st.error(f"❌ Failed to read PDF: {e}")

# PDF from URL
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
    st.warning("⚠️ Please enter a direct link that ends with `.pdf`.")

# Summary logic
if full_text.strip():
    st.subheader("📑 Extracted Text Preview")
    st.text_area("First part of the paper:", full_text[:2000], height=300)

    if st.button("🧠 Generate Detailed Summary"):
        with st.spinner("Summarizing..."):
            summary = generate_final_summary(full_text)
            st.subheader("✅ Detailed Summary")
            st.markdown(summary)
            st.caption("⚠️ This AI summary may omit technical details. Always verify with the original paper.")
else:
    st.info("Upload a PDF or paste a valid .pdf link to get started.")

