# server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import uuid
import os
import io
import time
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
import pdfplumber
from docx import Document

# Import your custom modules
from chunking import RecursiveTokenChunker, LLMAgenticChunker, ProtonxSemanticChunker
from utils import process_batch, divide_dataframe, get_search_result

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Chroma client
chroma_client = chromadb.PersistentClient("db")

# In-memory storage (for simplicity; consider persistent storage for production)
session_data = {}

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        uploaded_file = request.files['file']
        language = request.form.get('language', 'en')
        chunk_size = int(request.form.get('chunk_size', 200))
        number_docs_retrieval = int(request.form.get('number_docs_retrieval', 10))
        embedding_model_name = request.form.get('embedding_model')

        # Read file based on type
        if uploaded_file.filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.filename.endswith(".json"):
            json_data = json.load(uploaded_file)
            df = pd.json_normalize(json_data)
        elif uploaded_file.filename.endswith(".pdf"):
            pdf_text = []
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page in pdf.pages:
                    pdf_text.append(page.extract_text())
            df = pd.DataFrame({"content": pdf_text})
        elif uploaded_file.filename.endswith((".docx", ".doc")):
            doc = Document(io.BytesIO(uploaded_file.read()))
            docx_text = [para.text for para in doc.paragraphs if para.text]
            df = pd.DataFrame({"content": docx_text})
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Generate unique document IDs
        doc_ids = [str(uuid.uuid4()) for _ in range(len(df))]
        df['doc_id'] = doc_ids

        # Save DataFrame to session
        session_id = str(uuid.uuid4())
        session_data[session_id] = {
            "df": df.to_dict(orient='records'),
            "language": language,
            "chunk_size": chunk_size,
            "number_docs_retrieval": number_docs_retrieval,
            "embedding_model_name": embedding_model_name,
            "collection": None,
            "embedding_model": None,
            "chat_history": []
        }

        return jsonify({"message": "File uploaded successfully", "session_id": session_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/initialize', methods=['POST'])
def initialize():
    try:
        data = request.json
        session_id = data['session_id']
        language = session_data[session_id]['language']
        embedding_model_name = data.get('embedding_model')

        # Load embedding model
        if language == 'en':
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        elif language == 'vi':
            embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
        else:
            return jsonify({"error": "Unsupported language"}), 400

        session_data[session_id]['embedding_model'] = embedding_model

        # Initialize Chroma collection
        if session_data[session_id]['collection'] is None:
            collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "A collection for RAG system"},
            )
            session_data[session_id]['collection'] = collection

        return jsonify({"message": "Initialization successful"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chunk', methods=['POST'])
def chunk_data():
    try:
        data = request.json
        session_id = data['session_id']
        chunk_option = data.get('chunk_option', 'No Chunking')
        embedding_option = data.get('embedding_option', 'TF-IDF')
        gemini_api_key = data.get('gemini_api_key', None)

        df = pd.DataFrame(session_data[session_id]['df'])
        index_column = data.get('index_column', df.columns[0])

        gemini_model = None
        if chunk_option == "AgenticChunker":
            if not gemini_api_key:
                return jsonify({"error": "Gemini API key required for AgenticChunker"}), 400
            genai.configure(api_key=gemini_api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-pro')

        chunk_records = []

        for index, row in df.iterrows():
            selected_column_value = row[index_column]
            if not (isinstance(selected_column_value, str) and len(selected_column_value) > 0):
                continue

            chunks = []
            if chunk_option == "No Chunking":
                chunks = [selected_column_value]
            elif chunk_option == "RecursiveTokenChunker":
                chunker = RecursiveTokenChunker(chunk_size=session_data[session_id]['chunk_size'])
                chunks = chunker.split_text(selected_column_value)
            elif chunk_option == "SemanticChunker":
                if embedding_option == "TF-IDF":
                    chunker = ProtonxSemanticChunker(embedding_type="tfidf")
                else:
                    chunker = ProtonxSemanticChunker(embedding_type="transformers", model="all-MiniLM-L6-v2")
                chunks = chunker.split_text(selected_column_value)
            elif chunk_option == "AgenticChunker":
                chunker = LLMAgenticChunker(organisation="google", model_name="gemini-1.5-pro", api_key=gemini_api_key)
                chunks = chunker.split_text(selected_column_value)

            for chunk in chunks:
                chunk_record = {**row.to_dict(), 'chunk': chunk}
                chunk_record = {
                    'chunk': chunk_record['chunk'],
                    'doc_id': chunk_record['doc_id'],
                    # Add other necessary fields if any
                }
                chunk_records.append(chunk_record)

        chunks_df = pd.DataFrame(chunk_records)
        session_data[session_id]['chunks_df'] = chunks_df.to_dict(orient='records')

        return jsonify({"message": "Chunking completed", "num_chunks": len(chunks_df)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save', methods=['POST'])
def save_data():
    try:
        data = request.json
        session_id = data['session_id']
        collection = session_data[session_id]['collection']
        embedding_model = session_data[session_id]['embedding_model']
        chunks_df = pd.DataFrame(session_data[session_id]['chunks_df'])
        batch_size = 256

        df_batches = divide_dataframe(chunks_df, batch_size)
        num_batches = len(df_batches)

        for i, batch_df in enumerate(df_batches):
            if batch_df.empty:
                continue
            process_batch(batch_df, embedding_model, collection)
            # Optionally, implement progress tracking

        return jsonify({"message": "Data saved to Chroma successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        session_id = data['session_id']
        prompt = data['prompt']
        gemini_api_key = data.get('gemini_api_key')

        if 'gemini_model' not in session_data[session_id]:
            if not gemini_api_key:
                return jsonify({"error": "Gemini API key required"}), 400
            genai.configure(api_key=gemini_api_key)
            session_data[session_id]['gemini_model'] = genai.GenerativeModel('gemini-1.5-pro')

        collection = session_data[session_id]['collection']
        embedding_model = session_data[session_id]['embedding_model']
        number_docs_retrieval = session_data[session_id]['number_docs_retrieval']
        chunks_df = pd.DataFrame(session_data[session_id]['chunks_df'])

        # Retrieve relevant documents
        metadatas, retrieved_data = get_search_result(
            embedding_model, 
            prompt, 
            collection, 
            ["chunk"],  # Assuming 'chunk' is the column to answer from
            number_docs_retrieval
        )

        if metadatas:
            flattened_metadatas = [item for sublist in metadatas for item in sublist]
            metadata_df = pd.DataFrame(flattened_metadatas)
            # Optionally, return metadata or store it

        enhanced_prompt = f"""You are a good salesperson. The prompt of the customer is: "{prompt}". Answer it based on the following retrieved data: \n{retrieved_data}"""

        # Generate response using Gemini
        response = session_data[session_id]['gemini_model'].generate_content(enhanced_prompt)
        content = response.candidates[0].content.parts[0].text

        # Update chat history
        if "chat_history" not in session_data[session_id]:
            session_data[session_id]['chat_history'] = []
        session_data[session_id]['chat_history'].append({"role": "user", "content": prompt})
        session_data[session_id]['chat_history'].append({"role": "assistant", "content": content})

        return jsonify({"response": content, "chat_history": session_data[session_id]['chat_history']}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    try:
        session_id = request.args.get('session_id')
        chat_history = session_data[session_id].get('chat_history', [])
        return jsonify({"chat_history": chat_history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5002, debug=True)
