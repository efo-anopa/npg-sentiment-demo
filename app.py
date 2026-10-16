import streamlit as st
import requests

st.set_page_config(page_title="Lightweight Sentiment App", layout="centered")
st.title("⚡ Ultra-Lightweight Sentiment Analyzer")

# 1. Configuration (Uses the standard DistilBERT sentiment model hosted by HF)
API_URL = "https://huggingface.co"

# Securely grab your free token from Streamlit Secrets or prompt the user
# Get a token at: hf.co/settings/tokens
HF_TOKEN = st.secrets.get("HF_TOKEN", "") 

def query_sentiment(text):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# 2. UI Layout
user_input = st.text_area("Enter text to evaluate:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter text first.")
    elif not HF_TOKEN:
        st.error("Please configure your HF_TOKEN in Streamlit secrets!")
    else:
        with st.spinner("Calling API..."):
            try:
                # The API returns a nested list: [[{'label': '...', 'score': ...}]]
                api_result = query_sentiment(user_input)
                top_prediction = api_result[0][0]
                
                label = top_prediction['label']
                score = top_prediction['score']
                
                if label == "POSITIVE":
                    st.success(f"**Positive** (Confidence: {score:.2%})")
                else:
                    st.error(f"**Negative** (Confidence: {score:.2%})")
            except Exception as e:
                st.error("API is waking up or limit reached. Please try again in a few seconds.")
