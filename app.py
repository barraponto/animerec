import streamlit as st
from langchain_chroma import Chroma

from chains.ingest import IngestChain
from chains.recommend import RecommendChain
from chains.vectorstore import vectorstore
from settings import Settings


@st.cache_resource
def init_settings():
    return Settings()


@st.cache_resource
def init_vectorstore(settings: Settings):
    ingest_chain = IngestChain(settings, vectorstore)
    return ingest_chain.vectorstore


@st.cache_resource
def init_recommender(settings: Settings, _vectorstore: Chroma):
    return RecommendChain(settings, _vectorstore)


st.title("Anime Recommendation System")

with st.spinner("Loading data..."):
    settings = init_settings()
    vectorstore = init_vectorstore(settings)
    recommender = init_recommender(settings, vectorstore)

prompt = st.chat_input("Tell us what you're looking for in an anime.")

if prompt:
    with st.spinner("Thinking..."):
        messages = recommender.ask(prompt)
        for message in messages:
            print(message)
            st.chat_message(message.type).write(message.content)
