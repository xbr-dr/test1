import streamlit as st
import pandas as pd
import PyPDF2
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

# Initialize Groq client (you'll need to set up your API key)
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    client = None

# Initialize models
@st.cache_resource
def load_models():
    english_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    multilingual_model = SentenceTransformer('sentence-transformers/LaBSE')
    return english_model, multilingual_model

english_model, multilingual_model = load_models()

# Initialize FAISS index
def init_faiss_index(dimension):
    return faiss.IndexFlatL2(dimension)

# Knowledge base storage
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []
if 'locations' not in st.session_state:
    st.session_state.locations = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = english_model

# File processing functions
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_text_to_sentences(text):
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 0]
    return sentences

def process_file(file, file_type):
    if file_type == 'pdf':
        text = extract_text_from_pdf(file)
        sentences = process_text_to_sentences(text)
        return sentences, None
    elif file_type in ['csv', 'xlsx', 'xls']:
        if file_type == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Check for location data
        location_cols = ['name', 'latitude', 'longitude', 'description']
        if all(col in df.columns for col in location_cols):
            locations = df[location_cols].to_dict('records')
            return None, locations
        else:
            # Treat as regular data
            text = df.to_string()
            sentences = process_text_to_sentences(text)
            return sentences, None
    return None, None

def update_knowledge_base(new_sentences):
    st.session_state.knowledge_base.extend(new_sentences)
    
    # Update FAISS index
    if new_sentences:
        embeddings = st.session_state.current_model.encode(new_sentences)
        if st.session_state.faiss_index is None:
            st.session_state.faiss_index = init_faiss_index(embeddings.shape[1])
        st.session_state.faiss_index.add(embeddings)

def update_locations(new_locations):
    st.session_state.locations.extend(new_locations)

# RAG functions
def retrieve_relevant_info(query, k=3):
    if not st.session_state.knowledge_base or st.session_state.faiss_index is None:
        return []
    
    # Detect language and use appropriate model
    try:
        lang = detect(query)
        model = multilingual_model if lang != 'en' else english_model
    except:
        model = english_model
    
    query_embedding = model.encode([query])
    distances, indices = st.session_state.faiss_index.search(query_embedding, k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx >= 0 and idx < len(st.session_state.knowledge_base):
            results.append({
                'text': st.session_state.knowledge_base[idx],
                'score': float(dist)
            })
    
    return results

def generate_response(query, context):
    if not client:
        return "Error: Groq API not configured. Please set up your API key."
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful campus information assistant. Use the provided context to answer questions accurately."
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def check_for_location(query):
    for loc in st.session_state.locations:
        if loc['name'].lower() in query.lower():
            return loc
    return None

# Admin Page
def admin_page():
    st.title("Admin Portal - Campus Information System")
    
    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=['pdf', 'csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split('.')[-1].lower()
            st.write(f"Processing {uploaded_file.name}...")
            
            with st.spinner(f"Processing {uploaded_file.name}"):
                sentences, locations = process_file(uploaded_file, file_type)
                
                if sentences:
                    update_knowledge_base(sentences)
                    st.success(f"Added {len(sentences)} sentences from {uploaded_file.name}")
                
                if locations:
                    update_locations(locations)
                    st.success(f"Added {len(locations)} locations from {uploaded_file.name}")
    
    st.header("Current Knowledge Base")
    st.write(f"Total sentences: {len(st.session_state.knowledge_base)}")
    st.write(f"Total locations: {len(st.session_state.locations)}")
    
    if st.button("Clear Knowledge Base"):
        st.session_state.knowledge_base = []
        st.session_state.locations = []
        st.session_state.faiss_index = None
        st.success("Knowledge base cleared")

# User Page
def user_page():
    st.title("Campus Information Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask about campus information or locations"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if this is a location query
        location = check_for_location(prompt)
        
        if location:
            # Display location information
            response = f"**{location['name']}**\n\n{location['description']}"
            
            # Create map
            m = folium.Map(
                location=[location['latitude'], location['longitude']],
                zoom_start=17
            )
            folium.Marker(
                [location['latitude'], location['longitude']],
                popup=location['name'],
                tooltip=location['name']
            ).add_to(m)
            
            with st.chat_message("assistant"):
                st.markdown(response)
                folium_static(m)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
        else:
            # Process as informational query
            with st.spinner("Thinking..."):
                relevant_info = retrieve_relevant_info(prompt)
                context = "\n".join([info['text'] for info in relevant_info])
                response = generate_response(prompt, context)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })

# Main App
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["User", "Admin"])
    
    if page == "Admin":
        admin_page()
    else:
        user_page()

if __name__ == "__main__":
    main()
