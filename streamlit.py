# app.py
import streamlit as st
import requests

st.title('Data Summarizer')

# Text input
text = st.text_area("Enter text to summarize:")

# Button to trigger summary
if st.button('Summarize'):
    # Assuming there's a backend API to summarize text
    response = requests.post('http://backend-api/summarize', json={'text': text})
    if response.status_code == 200:
        summary = response.json()['summary']
        st.write(summary)
    else:
        st.error('Error in API call.')
