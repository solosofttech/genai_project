import os
import streamlit as st
from bl.about import About
from dotenv import load_dotenv
from bl.ingestion import Ingestion
from bl.chatollama import ChatOllama
from bl.chatgroq import ChatGroqModel
from streamlit_option_menu import option_menu

load_dotenv()

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Ingestion", "Chat Using Groq (Faiss)", "Chat Ollama (Chroma)", "About Me"],
        icons=["cloud-upload", "hourglass", "info-circle", "fire"],
        menu_icon="cast",
        default_index=0,
    )

# Display content based on the selected option
if selected == "Ingestion":
    Ingestion.ingestion()    

elif selected == "Chat Using Groq (Faiss)":
    ChatGroqModel.chatbot()

elif selected == "Chat Ollama (Chroma)":
    ChatOllama.chat_interface()

elif selected == "About Me":
    About.initialize()