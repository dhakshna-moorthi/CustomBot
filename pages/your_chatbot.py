import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

LLMFOUNDRY_TOKEN = os.getenv("LLMFOUNDRY_TOKEN")
BASE_URL = os.getenv("BASE_URL")

client = OpenAI(
    api_key=f"{LLMFOUNDRY_TOKEN}:custom-bot",
    base_url=BASE_URL,
) 

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_pickle("data_with_embeddings.pkl")

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

if "context" not in st.session_state:
    st.session_state.context = [ {'role':'system', 'content':"""
You are {st.session_state.bot_name}, a friendly and efficient virtual assistant for {st.session_state.domain} domain. \

your specifications are as follows: {st.session_state.specifications}
                                  
For every user query (tagged as user_query), a relevant context (tagged as domain_context) from an internal {st.session_state.domain} \
database will be provided. Respond strictly based on this context unless the user query is has no {st.session_state.domain} context. \
If the provided context does not relate to the user's query, \
Explicitly state: "The internal {st.session_state.domain} database does not have sufficient information to answer your query." \
Then, offer general advice or suggestions for managing the condition, ensuring the information is helpful and accurate. \
                                 
Respond in a warm and approachable manner, and ensure all interactions prioritize customer safety and accuracy. \
"""} ]
    
st.title(st.session_state.bot_name)
    
if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("",placeholder="Type your query here...")

if st.button("Send"):
    if user_input.strip():
        query_embedding = model.encode(user_input)

        df['similarity'] = df['usage_embeddings'].apply(lambda x: cosine_similarity([x], [query_embedding])[0][0])
        closest_match = df.sort_values(by='similarity', ascending=False).iloc[0]
        
        content =  "domain_context: " + str(closest_match) + ", user_query: " + user_input

        st.session_state.context.append({'role': 'user', 'content': content})
        
        response = get_completion_from_messages(st.session_state.context)
        st.session_state.context.append({'role': 'assistant', 'content': response})
        
        st.session_state.conversation.append((f"**You:** {user_input}", f"**{st.session_state.bot_name}:** {response}"))
    else:
        st.warning("Please enter a message before sending.")

if st.session_state.conversation:
    for user_msg, bot_msg in st.session_state.conversation:
        st.markdown(user_msg)
        st.markdown(bot_msg)