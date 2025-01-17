import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Upload Data and Details")

bot_name = st.text_input("Enter your chatbot's name")
domain = st.text_input("Enter domain of the dataset")
specifications = st.text_area("Enter specifications", placeholder="Type your specifications here...", height=150)
primary_column = st.text_input("Enter primary column name")
uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if st.button("Send"):
    st.session_state.bot_name = bot_name
    st.session_state.domain = domain
    st.session_state.specifications = specifications
    df = pd.read_csv(uploaded_file)
    df['usage_embeddings'] = df[primary_column].apply(lambda x: model.encode(x))
    df.to_pickle('data_with_embeddings.pkl')
    st.success("Your custom chatbot has been created")