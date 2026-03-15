import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 FAQ Chatbot")

with open("faq.json") as file:
    data = json.load(file)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

user_question = st.text_input("Ask a question")

if user_question:
    query_vec = vectorizer.transform([user_question])
    similarity = cosine_similarity(query_vec, X)
    index = similarity.argmax()
    st.write("Answer:", answers[index])
