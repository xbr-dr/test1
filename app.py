import streamlit as st
import pandas as pd
import pypdf
from io import BytesIO
import faiss
from sentence_transformers import SentenceTransformer
import folium
from streamlit_folium import folium_static
from langdetect import detect
import numpy as np
import os
import tempfile
from groq import Groq
import json

# Initialize models
@st.cache_resource
def load_models():
    english_model = SentenceTransformer('all-mpnet-base-v2')
    multilingual_model = SentenceTransformer('LaBSE')
    return english_model, multilingual_model

# Initialize app state
def init_state():
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = []
    if 'locations' not in st.session_state:
        st.session_state.locations = []
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

init_state()

# File processing functions
def extract_text_from_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    return " ".join([page.extract_text() for page in pdf_reader.pages])

def process_text(text):
    return [s.strip() for s in text.split('.') if s.strip()]

def process_file(file, file_type):
    if file_type == 'pdf':
        text = extract_text_from_pdf(file)
        return process_text(text), None
    elif file_type in ['csv', 'xlsx', 'xls']:
        df = pd.read_csv(file) if file_type == 'csv' else pd.read_excel(file)
        if all(col in df.columns for col in ['name', 'latitude', 'longitude', 'description']):
            return None, df.to_dict('records')
        return process_text(df.to_string()), None
    return None, None

def update_system(new_texts, new_locations):
    if new_texts:
        st.session_state.knowledge_base.extend(new_texts)
        embeddings = english_model.encode(new_texts)
        if st.session_state.faiss_index is None:
            st.session_state.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        st.session_state.faiss_index.add(embeddings)
    if new_locations:
        st.session_state.locations.extend(new_locations)

# Initialize models
english_model, multilingual_model = load_models()

# Admin functions
def admin_page():
    st.title("Admin Portal")
    uploaded_files = st.file_uploader("Upload Files", 
                                   type=['pdf', 'csv', 'xlsx', 'xls'],
                                   accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            file_type = file.name.split('.')[-1].lower()
            with st.spinner(f"Processing {file.name}"):
                texts, locations = process_file(file, file_type)
                update_system(texts, locations)
                st.success(f"Processed {file.name}")

    st.header("Current System Status")
    st.write(f"Knowledge Base: {len(st.session_state.knowledge_base)} entries")
    st.write(f"Locations: {len(st.session_state.locations)} entries")
    
    if st.button("Reset System"):
        st.session_state.clear()
        init_state()
        st.success("System reset complete")

# User functions
def get_location(query):
    query = query.lower()
    return next((loc for loc in st.session_state.locations 
                if loc['name'].lower() in query), None)

def get_response(query):
    if not st.session_state.knowledge_base:
        return "System is still learning. Please check back later."
    
    try:
        lang = detect(query)
        model = multilingual_model if lang != 'en' else english_model
        query_embed = model.encode([query])
        _, indices = st.session_state.faiss_index.search(query_embed, 3)
        context = "\n".join(st.session_state.knowledge_base[i] 
                          for i in indices[0] if i < len(st.session_state.knowledge_base))
        
        if client:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "system", "content": "You are a helpful campus assistant."},
                         {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}]
            )
            return response.choices[0].message.content
        return context
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def user_page():
    st.title("Campus Assistant")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask about campus"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        location = get_location(prompt)
        if location:
            response = f"**{location['name']}**\n{location['description']}"
            m = folium.Map(location=[location['latitude'], location['longitude']], zoom_start=16)
            folium.Marker([location['latitude'], location['longitude']], 
                         popup=location['name']).add_to(m)
            with st.chat_message("assistant"):
                st.write(response)
                folium_static(m)
        else:
            with st.spinner("Thinking..."):
                response = get_response(prompt)
            with st.chat_message("assistant"):
                st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["User", "Admin"])
    admin_page() if page == "Admin" else user_page()

if __name__ == "__main__":
    try:
        client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))
    except:
        client = None
    main()
