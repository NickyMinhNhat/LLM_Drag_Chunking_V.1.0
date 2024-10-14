# client.py

import streamlit as st
import requests
import pandas as pd
import uuid
from io import BytesIO

# Server URL
SERVER_URL = "http://localhost:5002"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "data_saved_success" not in st.session_state:
    st.session_state.data_saved_success = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Language Selection and Settings
def sidebar_settings():
    st.sidebar.title("Settings")
    
    # Language selection
    language_choice = st.sidebar.radio("Select Language:", ["English", "Vietnamese"])
    language_map = {"English": "en", "Vietnamese": "vi"}
    language = language_map[language_choice]
    
    # Chunk size
    chunk_size = st.sidebar.number_input(
        "Chunk Size", min_value=50, max_value=1000, value=200, step=50,
        help="Set the size of each chunk in terms of tokens."
    )
    
    # Number of documents retrieval
    number_docs_retrieval = st.sidebar.number_input(
        "Number of Documents Retrieval", min_value=1, max_value=50, value=10, step=1,
        help="Set the number of documents to retrieve."
    )
    
    return language, chunk_size, number_docs_retrieval

# Function to upload and process file
def upload_and_process_file(language, chunk_size, number_docs_retrieval):
    uploaded_file = st.file_uploader("Upload CSV, JSON, PDF, or DOCX file", type=["csv", "json", "pdf", "docx"])
    
    if uploaded_file:
        st.write(f"Uploaded File: {uploaded_file.name}")
        
        # Prepare form data
        files = {"file": (uploaded_file.name, uploaded_file, 'multipart/form-data')}
        data = {
            "language": language,
            "chunk_size": chunk_size,
            "number_docs_retrieval": number_docs_retrieval
        }

        if st.button("Upload and Process File"):
            with st.spinner("Uploading and processing file..."):
                try:
                    response = requests.post(f"{SERVER_URL}/upload", files=files, data=data)
                    if response.status_code == 200:
                        res = response.json()
                        st.session_state.session_id = res['session_id']
                        st.success(res['message'])
                    else:
                        st.error(response.json().get('error', 'An error occurred'))
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Function to initialize session
def initialize_session():
    if st.session_state.session_id and not st.session_state.data_saved_success:
        if st.button("Initialize"):
            with st.spinner("Initializing..."):
                try:
                    payload = {"session_id": st.session_state.session_id}
                    response = requests.post(f"{SERVER_URL}/initialize", json=payload)
                    if response.status_code == 200:
                        st.success(response.json()['message'])
                    else:
                        st.error(response.json().get('error', 'Initialization failed'))
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Function to perform chunking
def perform_chunking():
    if st.session_state.session_id and not st.session_state.data_saved_success:
        st.subheader("Chunking Options")
        
        chunk_options = ["No Chunking", "RecursiveTokenChunker", "SemanticChunker", "AgenticChunker"]
        chunk_option = st.selectbox("Select Chunking Method:", chunk_options)
        
        # Choose embedding option if SemanticChunker is selected
        if chunk_option == "SemanticChunker":
            embedding_option = st.selectbox(
                "Choose the embedding method for Semantic Chunker:",
                ["TF-IDF", "Sentence-Transformers"]
            )
        else:
            embedding_option = "TF-IDF"  # Default
        
        # Gemini API Key for AgenticChunker
        if chunk_option == "AgenticChunker":
            gemini_api_key = st.text_input("Enter your Gemini API Key:", type="password")
        else:
            gemini_api_key = None
        
        if st.button("Perform Chunking"):
            with st.spinner("Chunking data..."):
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "chunk_option": chunk_option,
                        "embedding_option": embedding_option,
                        "gemini_api_key": gemini_api_key
                    }
                    response = requests.post(f"{SERVER_URL}/chunk", json=payload)
                    if response.status_code == 200:
                        res = response.json()
                        st.success(f"Chunking completed with {res['num_chunks']} chunks")
                    else:
                        st.error(response.json().get('error', 'Chunking failed'))
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Function to save data to Chroma
def save_data_to_chroma():
    if st.session_state.session_id and not st.session_state.data_saved_success:
        if st.button("Save Data to Chroma"):
            with st.spinner("Saving data to Chroma..."):
                try:
                    payload = {"session_id": st.session_state.session_id}
                    response = requests.post(f"{SERVER_URL}/save", json=payload)
                    if response.status_code == 200:
                        st.success(response.json()['message'])
                        st.session_state.data_saved_success = True
                    else:
                        st.error(response.json().get('error', 'Saving data failed'))
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Function to display setup completion
def display_setup_completion():
    if st.session_state.data_saved_success:
        st.success("✅ **Data Saved Successfully!**")

# Function to handle interactive chatbot
def interactive_chatbot():
    if st.session_state.data_saved_success:
        st.header("Interactive Chatbot")
        
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Bot:** {message['content']}")
        
        # Chat input fields
        prompt = st.text_input("You:", key="chat_input")
        gemini_api_key_chat = st.text_input("Enter your Gemini API Key for Chat:", type="password")
        
        if st.button("Send") and prompt:
            with st.spinner("Generating response..."):
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "prompt": prompt,
                        "gemini_api_key": gemini_api_key_chat
                    }
                    response = requests.post(f"{SERVER_URL}/chat", json=payload)
                    if response.status_code == 200:
                        res = response.json()
                        st.session_state.chat_history = res['chat_history']
                    else:
                        st.error(response.json().get('error', 'Chat failed'))
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Option to refresh chat history
        if st.button("Refresh Chat"):
            with st.spinner("Refreshing chat history..."):
                try:
                    params = {"session_id": st.session_state.session_id}
                    response = requests.get(f"{SERVER_URL}/get_chat_history", params=params)
                    if response.status_code == 200:
                        res = response.json()
                        st.session_state.chat_history = res['chat_history']
                    else:
                        st.error(response.json().get('error', 'Failed to fetch chat history'))
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Main function to orchestrate the Streamlit app
def main():
    st.title("Drag and Drop RAG")
    st.image("https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/678dadd0-603b-11ef-b0a7-998b84b38d43-ProtonX_logo_horizontally__1_.png")
    
    # Sidebar settings
    language, chunk_size, number_docs_retrieval = sidebar_settings()
    
    # Upload and process file
    upload_and_process_file(language, chunk_size, number_docs_retrieval)
    
    # Initialize session
    initialize_session()
    
    # Perform chunking
    perform_chunking()
    
    # Save data to Chroma
    save_data_to_chroma()
    
    # Display setup completion
    display_setup_completion()
    
    # Interactive chatbot
    interactive_chatbot()

if __name__ == "__main__":
    main()
