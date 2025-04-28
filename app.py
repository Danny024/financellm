import streamlit as st
import os
import json
import requests
import boto3
import time
from pathlib import Path
from modules.data_processor import process_json_to_markdown, load_markdown_document, get_markdown_splits
from modules.rag_pipeline import setup_vector_store, create_rag_chain
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd

# Environment setup
st.set_page_config(page_title="FinQA Chatbot", layout="wide")
DATA_FOLDER = "data"
VECTOR_DB_FOLDER = "vector_db"
S3_BUCKET = "conv-fin-qa-dataset"  # Replace with your S3 bucket name
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)
OLLAMA_BASE_URL = "http://localhost:11434"

def check_ollama():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json()
        required_models = ["deepseek-r1:1.5b", "nomic-embed-text:latest"]
        available_models = [model["name"] for model in models.get("models", [])]
        return all(model in available_models for model in required_models)
    except requests.RequestException:
        return False

def check_openai_api():
    return os.getenv("OPENAI_API_KEY") is not None

def extract_tables_from_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        tables = []
        for entry in json_data[:3]:  # Limit to first 3 entries
            table = entry.get("table", [])
            if table:
                tables.append({"id": entry.get("id", "Unknown"), "table": table})
        return tables
    except Exception:
        return []

def render_table(table_data):
    if not table_data:
        return ""
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    return df.to_html(index=False, border=1, classes="table-auto w-full border-collapse border border-gray-300")

def estimate_processing_time(file_size_bytes):
    file_size_mb = file_size_bytes / (1024 * 1024)
    total_time = file_size_mb  # 1 second per MB
    stage_time = total_time / 4  # 4 stages
    return total_time, stage_time

def s3_upload_vector_store(s3_client, local_path, s3_key):
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            s3_file = f"{s3_key}/{relative_path}"
            s3_client.upload_file(local_file, S3_BUCKET, s3_file)

def s3_download_vector_store(s3_client, s3_key, local_path):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_key):
        for obj in page.get('Contents', []):
            s3_file = obj['Key']
            local_file = os.path.join(local_path, os.path.relpath(s3_file, s3_key))
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            s3_client.download_file(S3_BUCKET, s3_file, local_file)

def create_or_load_vector_store(filename, chunks, embeddings, s3_client):
    s3_key = f"{filename}.faiss"
    local_path = os.path.join(VECTOR_DB_FOLDER, filename)
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=f"{s3_key}/index.faiss")
        s3_download_vector_store(s3_client, s3_key, local_path)
        vector_store = FAISS.load_local(local_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    except s3_client.exceptions.ClientError:
        vector_store = setup_vector_store(chunks, base_url=OLLAMA_BASE_URL)
        os.makedirs(local_path, exist_ok=True)
        vector_store.save_local(local_path)
        s3_upload_vector_store(s3_client, local_path, s3_key)
    return vector_store

# Streamlit UI
st.title("Your Financial Analyst Chatbot")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'answer' not in st.session_state:
    st.session_state.answer = ""
if 'retrieved_docs' not in st.session_state:
    st.session_state.retrieved_docs = []

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# Sidebar: LLM and API status
st.sidebar.subheader("LLM Status")
if check_ollama():
    st.sidebar.success("Ollama server running with required models.")
else:
    st.sidebar.warning("Ollama server not running or missing required models.")
if check_openai_api():
    st.sidebar.success("OpenAI API key detected for gpt-4o-mini.")
else:
    st.sidebar.warning("OpenAI API key not set. Set OPENAI_API_KEY in .env.")

# Vector DB selection or JSON upload
vector_db_options = []
try:
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Delimiter='/')
    for prefix in response.get('CommonPrefixes', []):
        vector_db_options.append(prefix['Prefix'].rstrip('/').split('/')[-1])
except s3_client.exceptions.NoSuchBucket:
    st.sidebar.error("S3 bucket 'conv-fin-qa-dataset' not found. Please create it.")
vector_db_options.append("Upload New JSON")
selected_vector_db = st.selectbox("Select Vector DB or Upload New JSON", vector_db_options, index=len(vector_db_options)-1)

# Display tables for selected vector DB
if selected_vector_db != "Upload New JSON":
    json_path = os.path.join(DATA_FOLDER, f"{selected_vector_db}.json")
    if os.path.exists(json_path):
        tables = extract_tables_from_json(json_path)
        st.sidebar.subheader("Tables from JSON")
        for table_data in tables:
            st.sidebar.markdown(f"**Table from Entry: {table_data['id']}**")
            st.sidebar.markdown(render_table(table_data['table']), unsafe_allow_html=True)
    else:
        st.sidebar.warning("JSON file not found for the selected vector DB.")

# JSON upload and processing
if selected_vector_db == "Upload New JSON":
    st.subheader("Upload train.json")
    uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])
    if uploaded_file:
        st.sidebar.subheader("Uploaded File")
        st.sidebar.write(uploaded_file.name)
        file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        tables = extract_tables_from_json(file_path)
        st.sidebar.subheader("Tables from JSON")
        for table_data in tables:
            st.sidebar.markdown(f"**Table from Entry: {table_data['id']}**")
            st.sidebar.markdown(render_table(table_data['table']), unsafe_allow_html=True)
        if st.button("Process JSON"):
            with st.spinner("Processing document..."):
                progress_bar = st.progress(0)
                file_size_bytes = os.path.getsize(file_path)
                total_time, stage_time = estimate_processing_time(file_size_bytes)
                st.sidebar.text(f"Estimated total processing time: {total_time:.1f} seconds")
                stages = ['Uploading file', 'Converting to Markdown', 'Splitting into chunks', 'Indexing in vector store']
                for i, stage in enumerate(stages):
                    start_time = time.time()
                    progress_bar.progress((i + 1) * 25)
                    st.sidebar.text(f"Processing: {stage}")
                    st.sidebar.text(f"Estimated time remaining: {(total_time - (i + 1) * stage_time):.1f} seconds")
                    try:
                        if stage == 'Uploading file':
                            pass
                        elif stage == 'Converting to Markdown':
                            markdown_output = process_json_to_markdown(file_path, os.path.join(DATA_FOLDER, 'train_output.md'))
                            markdown_content = load_markdown_document(markdown_output)
                            if not markdown_content:
                                st.error("Failed to process Markdown content.")
                                break
                        elif stage == 'Splitting into chunks':
                            chunks = get_markdown_splits(markdown_content)
                        elif stage == 'Indexing in vector store':
                            embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url=OLLAMA_BASE_URL)
                            st.session_state.vector_store = create_or_load_vector_store(uploaded_file.name.split(".")[0], chunks, embeddings, s3_client)
                            st.session_state.retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 1})
                            st.sidebar.text(f"File processed successfully, {len(chunks)} chunks indexed")
                            st.success("JSON processed and stored in the vector database.")
                    except Exception as e:
                        st.error(f"Error processing file at stage '{stage}': {str(e)}")
                        break
                    elapsed = time.time() - start_time
                    if elapsed < stage_time:
                        time.sleep(stage_time - elapsed)
                progress_bar.empty()
elif selected_vector_db != "Upload New JSON":
    try:
        embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url=OLLAMA_BASE_URL)
        st.session_state.vector_store = create_or_load_vector_store(selected_vector_db, [], embeddings, s3_client)
        st.session_state.retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 1})
        st.sidebar.text(f"Loaded vector DB: {selected_vector_db}")
    except Exception as e:
        st.sidebar.warning(f"Failed to load vector DB '{selected_vector_db}': {str(e)}")

# Chatbot section
st.subheader("Chat with FinQA")
if st.session_state.vector_store is None:
    st.warning("Please upload and process train.json or select a vector DB to start chatting.")
else:
    llm_option = st.selectbox("Select LLM:", ["Ollama (deepseek-r1:1.5b)", "OpenAI (gpt-4o-mini)"], index=0 if check_ollama() else 1 if check_openai_api() else 0)
    k = st.selectbox("Number of Retrieved Documents (k):", [1, 2, 3, 5, 10], index=0)
    question = st.text_input("Enter your question:", placeholder="e.g., What was the percentage change in the net cash from operating activities from 2008 to 2009?")
    if st.button("Ask"):
        if question:
            with st.spinner("Retrieving documents and generating answer..."):
                st.session_state.retriever.search_kwargs['k'] = k
                docs = st.session_state.retriever.invoke(question)
                st.session_state.retrieved_docs = [
                    {'id': doc.metadata.get('Header 1', 'Unknown').split(' (')[0], 'content': doc.page_content[:500] + '...'}
                    for doc in docs
                ]
                rag_chain = create_rag_chain(st.session_state.retriever, base_url=OLLAMA_BASE_URL, use_openai=llm_option.startswith("OpenAI"))
                response = ""
                for chunk in rag_chain.stream(question):
                    response += chunk
                actual_answer = ""
                if "<ANSWER>" in response and "</ANSWER>" in response:
                    start = response.index("<ANSWER>") + len("<ANSWER>")
                    end = response.index("</ANSWER>")
                    actual_answer = response[start:end].strip().replace("- ", "").strip()
                actual_answer = actual_answer or response
                st.session_state.answer = actual_answer

    if st.session_state.retrieved_docs:
        st.subheader("Retrieved Documents")
        for index, doc in enumerate(st.session_state.retrieved_docs):
            st.markdown(f"**Document {index + 1} (ID: {doc['id']})**")
            st.text_area("", doc['content'], height=100, disabled=True, key=f"doc_{index}")

    if st.session_state.answer:
        st.subheader("Answer")
        st.text_area("", st.session_state.answer, height=150, disabled=True, key="answer_output")