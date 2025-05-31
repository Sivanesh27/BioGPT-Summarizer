
import openai

import streamlit as st
openai.api_key = st.secrets["OPENAI_API_KEY"]

def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo" for cheaper
        messages=[
            {"role": "system", "content": "Summarize this biomedical research paper in clear, plain English."},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0]['message']['content']
