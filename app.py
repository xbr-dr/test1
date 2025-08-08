import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional

# Page Config
st.set_page_config(
    page_title="Campus Assistant",
    page_icon="🏫",
    layout="wide"
)

# Initialize Session State
def init_session_state():
    session_defaults = {
        'knowledge_base': [],
        'location_data': [],
        'chat_history': [],
        'search_model': None,
        'authenticated': False,
        'user_type': None,
        'faiss_index': None
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Authentication
def login_section():
    """Login form for admin/user"""
    if not st.session_state.authenticated:
        with st.sidebar:
            with st.form("login_form"):
                st.subheader("Login")
                user_type = st.radio("Login as:", ["User", "Admin"])
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login"):
                    if user_type == "Admin" and password == st.secrets.get("ADMIN_PASSWORD", "admin123"):
                        st.session_state.authenticated = True
                        st.session_state.user_type = "admin"
                        st.rerun()
                    elif user_type == "User":
                        st.session_state.authenticated = True
                        st.session_state.user_type = "user"
                        st.rerun()
                    else:
                        st.error("Incorrect password for admin")

# File Processing
def extract_pdf_text(file):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])
    except:
        return ""

def process_text(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if 20 <= len(s.strip()) <= 500]

def extract_locations(df):
    locations = []
    if not df.empty:
        cols = {col.lower(): col for col in df.columns}
        if 'name' in cols and 'latitude' in cols and 'longitude' in cols:
            for _, row in df.iterrows():
                try:
                    locations.append({
                        'name': str(row[cols['name']]),
                        'latitude': float(row[cols['latitude']]),
                        'longitude': float(row[cols['longitude']]),
                        'description': str(row[cols.get('description', '')])
                    })
                except:
                    continue
    return locations

# Search Functions
def create_search_index(texts, model):
    try:
        import faiss
        embeddings = model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index
    except:
        return None

def search_knowledge(query, model):
    if not st.session_state.knowledge_base:
        return []
    
    if model and st.session_state.faiss_index:
        try:
            query_embedding = model.encode([query])
            _, indices = st.session_state.faiss_index.search(query_embedding, 3)
            return [st.session_state.knowledge_base[i] for i in indices[0] if i < len(st.session_state.knowledge_base)]
        except:
            pass
    
    # Fallback to basic search
    query_words = set(query.lower().split())
    results = []
    for text in st.session_state.knowledge_base:
        if any(word in text.lower() for word in query_words):
            results.append(text)
            if len(results) >= 3:
                break
    return results

# Response Generation
def generate_response(query, context):
    try:
        from groq import Groq
        client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))
        
        prompt = f"""You are a campus information assistant. Answer ONLY using the provided context.
If you don't know, say "I don't have that information."

Context: {context}

Question: {query}

Answer:"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
    except:
        return context[0] if context else "I don't have that information."

# Admin Page
def admin_page():
    st.title("🔧 Admin Portal")
    
    # File Upload Section
    with st.expander("📤 Upload Data", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload campus files (PDF, CSV, Excel)",
            type=['pdf', 'csv', 'xlsx', 'xls'],
            accept_multiple_files=True
        )
        
        if st.button("Process Files") and uploaded_files:
            with st.spinner("Processing files..."):
                new_texts = []
                new_locations = []
                
                for file in uploaded_files:
                    if file.type == "application/pdf":
                        text = extract_pdf_text(file)
                        new_texts.extend(process_text(text))
                    else:
                        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                        new_texts.extend(process_text(df.to_string()))
                        new_locations.extend(extract_locations(df))
                
                st.session_state.knowledge_base.extend(new_texts)
                st.session_state.location_data.extend(new_locations)
                
                # Update search index
                if new_texts and st.session_state.search_model:
                    st.session_state.faiss_index = create_search_index(
                        st.session_state.knowledge_base, 
                        st.session_state.search_model
                    )
                
                st.success(f"Added {len(new_texts)} text chunks and {len(new_locations)} locations")
    
    # Data Management
    with st.expander("🗃️ Data Management"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Text Chunks", len(st.session_state.knowledge_base))
        with col2:
            st.metric("Locations", len(st.session_state.location_data))
        
        if st.button("Clear All Data", type="primary"):
            st.session_state.knowledge_base = []
            st.session_state.location_data = []
            st.session_state.chat_history = []
            st.session_state.faiss_index = None
            st.success("All data cleared!")
    
    # System Status
    with st.expander("⚙️ System Status"):
        try:
            import PyPDF2
            st.success("✅ PDF Processing: Available")
        except:
            st.warning("⚠️ PDF Processing: Not Available")
        
        try:
            from sentence_transformers import SentenceTransformer
            st.success("✅ AI Models: Available")
        except:
            st.warning("⚠️ AI Models: Not Available")
        
        st.info(f"Knowledge Base: {len(st.session_state.knowledge_base)} entries")
        st.info(f"Locations: {len(st.session_state.location_data)} entries")

# User Page
def user_page():
    st.title("💬 Campus Assistant")
    
    # Load model if not loaded
    if not st.session_state.search_model:
        try:
            from sentence_transformers import SentenceTransformer
            st.session_state.search_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            st.session_state.search_model = None
    
    # Chat Interface
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("location"):
                show_location(msg["location"])
    
    if prompt := st.chat_input("Ask about campus..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Check for location
        location = next(
            (loc for loc in st.session_state.location_data 
             if loc['name'].lower() in prompt.lower()), 
            None
        )
        
        # Get response
        context = search_knowledge(prompt, st.session_state.search_model)
        response = generate_response(prompt, context)
        
        # Add assistant response
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response,
            "location": location
        })
        
        st.rerun()

def show_location(location):
    try:
        import folium
        from streamlit_folium import st_folium
        
        m = folium.Map(
            location=[location['latitude'], location['longitude']],
            zoom_start=17,
            tiles="cartodbpositron"
        )
        folium.Marker(
            [location['latitude'], location['longitude']],
            popup=location['name'],
            tooltip=location['name']
        ).add_to(m)
        
        st_folium(m, width=700, height=300)
    except:
        st.write(f"📍 {location['name']}")
        st.write(f"Latitude: {location['latitude']}, Longitude: {location['longitude']}")

# Main App
def main():
    login_section()
    
    if st.session_state.authenticated:
        # Sidebar
        st.sidebar.title("🏫 Navigation")
        
        # Credits
        st.sidebar.markdown("---")
        st.sidebar.caption("Campus Assistant v1.0")
        
        # Page Selection
        if st.session_state.user_type == "admin":
            admin_page()
        else:
            user_page()
    else:
        st.info("Please login to access the campus assistant")

if __name__ == "__main__":
    main()
