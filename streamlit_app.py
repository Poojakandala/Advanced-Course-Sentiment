import streamlit as st
import requests

st.set_page_config(page_title="Course Review AI", page_icon="🎓")

st.title("🎓 Course Review Sentiment AI")
st.markdown("Analyze student feedback using **Deep Learning (BERT-BiLSTM)**.")

user_input = st.text_area("Enter course review here:", placeholder="The course was challenging but very rewarding...")

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner('AI is analyzing...'):
            # Calling your FastAPI backend
            response = requests.post("http://localhost:8000/predict", json={"text": user_input})
            result = response.json()
            
            sentiment = result['sentiment']
            
            if sentiment == "Positive":
                st.success(f"Result: {sentiment} 😊")
            elif sentiment == "Neutral":
                st.warning(f"Result: {sentiment} 😐")
            else:
                st.error(f"Result: {sentiment} ☹️")
    else:
        st.write("Please enter some text first.")
