from transformers import pipeline
import streamlit as st

# App UI
st.title("YOUR PERSONAL SENTIMENT ANALYZER")
st.subheader("It can tell whether the sentiment is positive or negative.")
st.write("Loading model...")

# Load model (do this once to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment = load_model()

# Chat input
reply = st.chat_input("Type something that has either a positive or negative sentiment...")

# Only run the sentiment analysis if input is provided
if reply:
    result = sentiment(reply)
    st.write("Sentiment Analysis Result:", result)
    st.write(f"the senitment is {result}")