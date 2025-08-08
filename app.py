import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from typing import List, Dict, Optional

# Set page config
st.set_page_config(
    page_title="Campus Assistant",
    page_icon="🏫",
    layout="wide"
)

# Initialize session state
def init_session_state():
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = []
    if 'location_data' not in st.session_state:
        st.session_state.location_data = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'search_index' not in st.session_state:
        st.session_state.search_index = None
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

init_session_state()

# Load dependencies with error handling
@st.cache_resource
def load_dependencies():
    """Load optional dependencies with graceful fallback"""
    deps = {
        'pdf': False,
        'transformers': False,
        'faiss': False,
        'folium': False,
        'langdetect': False,
        'groq': False
    }
    
    try:
        import PyPDF2
        deps['pdf'] = True
    except ImportError:
        pass
    
    try:
        from sentence_transformers import SentenceTransformer
        deps['transformers'] = True
    except ImportError:
        pass
    
    try:
        import faiss
        deps['faiss'] = True
    except ImportError:
        pass
    
    try:
        import folium
        from streamlit_folium import st_folium
        deps['folium'] = True
    except ImportError:
        pass
    
    try:
        from langdetect import detect
        deps['langdetect'] = True
    except ImportError:
        pass
    
    try:
        from groq import Groq
        deps['groq'] = True
    except ImportError:
        pass
    
    return deps

@st.cache_resource
def load_ai_model():
    """Load AI model if available"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception:
        return None

def get_api_key():
    """Get API key from secrets only"""
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return None

def extract_pdf_text(pdf_file):
    """Extract text from PDF with fallback"""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def process_text(text: str) -> List[str]:
    """Split text into chunks"""
    if not text:
        return []
    
    # Improved sentence splitting
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    
    # Clean and filter
    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if 20 <= len(sentence) <= 500:
            chunks.append(sentence)
    
    return chunks

def extract_locations(df: pd.DataFrame) -> List[Dict]:
    """Improved location extraction from DataFrame"""
    locations = []
    
    if df.empty:
        return locations
    
    # Normalize column names
    col_map = {col.lower().strip(): col for col in df.columns}
    col_map.update({col.replace('_', ' ').lower(): col for col in df.columns})
    
    # Find relevant columns
    name_col = next((col_map[key] for key in ['name', 'location', 'place', 'building'] 
                   if key in col_map), None)
    lat_col = next((col_map[key] for key in ['latitude', 'lat'] 
                  if key in col_map), None)
    lon_col = next((col_map[key] for key in ['longitude', 'lon', 'long'] 
                  if key in col_map), None)
    desc_col = next((col_map[key] for key in ['description', 'desc', 'info'] 
                   if key in col_map), None)
    
    # Extract if we have required columns
    if name_col and lat_col and lon_col:
        for _, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    location = {
                        'name': str(row[name_col]).strip(),
                        'latitude': lat,
                        'longitude': lon,
                        'description': str(row[desc_col]).strip() if desc_col else ""
                    }
                    locations.append(location)
            except (ValueError, TypeError):
                continue
    
    return locations

def basic_search(query: str, texts: List[str], k: int = 3) -> List[str]:
    """Enhanced basic keyword search"""
    if not query or not texts:
        return []
    
    query_words = set(re.findall(r'\w+', query.lower()))
    
    # Score texts based on keyword matches
    scored_texts = []
    for text in texts:
        text_words = set(re.findall(r'\w+', text.lower()))
        common_words = query_words & text_words
        score = len(common_words)
        if score > 0:
            scored_texts.append((score, text))
    
    # Sort by score and return top k
    scored_texts.sort(reverse=True, key=lambda x: x[0])
    return [text for _, text in scored_texts[:k]]

def ai_search(query: str, texts: List[str], model, k: int = 3) -> List[str]:
    """Improved AI-powered search"""
    if not model or not texts:
        return basic_search(query, texts, k)
    
    try:
        import faiss
        
        # Create embeddings
        embeddings = model.encode(texts)
        query_embedding = model.encode([query])
        
        # Create index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        
        # Normalize
        faiss.normalize_L2(embeddings)
        faiss.normalize_L2(query_embedding)
        
        # Add and search
        index.add(embeddings.astype('float32'))
        _, indices = index.search(query_embedding.astype('float32'), k)
        
        return [texts[i] for i in indices[0] if i < len(texts)]
    except:
        return basic_search(query, texts, k)

def find_location(query: str, locations: List[Dict]) -> Optional[Dict]:
    """Improved location finding with fuzzy matching"""
    query_lower = query.lower()
    
    for location in locations:
        loc_name = location['name'].lower()
        if loc_name in query_lower or query_lower in loc_name:
            return location
    
    return None

def generate_response(query: str, context: List[str], api_key: str) -> str:
    """Generate precise AI response with anti-hallucination measures"""
    if not api_key:
        if context:
            return f"🔍 Here's what I found:\n\n{context[0]}"
        return "ℹ️ Please configure your Groq API key in .streamlit/secrets.toml to enable AI responses."
    
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        context_text = "\n".join([f"- {text}" for text in context[:3]]) if context else "No specific context available."
        
        prompt = f"""You are a precise campus information assistant. Only answer using the provided context. 
If the answer isn't in the context, say "I don't have information about that."

Context:
{context_text}

Question: {query}

Answer concisely and accurately:"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            max_tokens=300,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        # Ensure the model didn't hallucinate
        if not context and ("don't know" not in answer.lower() and "don't have" not in answer.lower()):
            return "I don't have information about that in our records."
        return answer
    except Exception:
        if context:
            return f"🔍 Here's what I found:\n\n{context[0]}"
        return "I couldn't find information about that in our records."

def show_map(location: Dict):
    """Improved map display"""
    try:
        import folium
        from streamlit_folium import st_folium
        
        with st.expander(f"📍 {location['name']}", expanded=True):
            st.write(location['description'])
            
            m = folium.Map(
                location=[location['latitude'], location['longitude']], 
                zoom_start=16,
                tiles='cartodbpositron'
            )
            
            folium.Marker(
                [location['latitude'], location['longitude']],
                popup=f"<b>{location['name']}</b><br>{location['description']}",
                tooltip=location['name'],
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            st_folium(m, height=300, width=700)
    except Exception:
        st.write(f"📍 **{location['name']}**")
        st.write(f"Coordinates: {location['latitude']}, {location['longitude']}")
        if location['description']:
            st.write(f"Description: {location['description']}")

def login_section():
    """Login/logout button"""
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.session_state.authenticated:
            if st.button("🚪 Logout"):
                st.session_state.authenticated = False
                st.rerun()
        else:
            if st.button("🔑 Login"):
                st.session_state.authenticated = True
                st.rerun()

def admin_page():
    """Admin interface with improved UI"""
    if not st.session_state.authenticated:
        st.warning("Please login to access admin panel")
        return
    
    st.title("🔧 Admin Panel")
    
    # System status
    deps = load_dependencies()
    with st.expander("System Status", expanded=True):
        cols = st.columns(4)
        with cols[0]:
            st.metric("PDF Support", "✅" if deps['pdf'] else "❌")
        with cols[1]:
            st.metric("AI Search", "✅" if deps['transformers'] and deps['faiss'] else "❌")
        with cols[2]:
            st.metric("Maps", "✅" if deps['folium'] else "❌")
        with cols[3]:
            st.metric("AI Chat", "✅" if deps['groq'] and get_api_key() else "❌")
    
    # File uploader
    with st.expander("Upload Files", expanded=True):
        files = st.file_uploader(
            "Upload campus files (PDF, CSV, Excel)",
            type=['pdf', 'csv', 'xlsx', 'xls'],
            accept_multiple_files=True
        )
        
        if files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_texts = []
            all_locations = []
            
            for i, file in enumerate(files):
                status_text.write(f"Processing {file.name}...")
                
                try:
                    if file.type == "application/pdf":
                        text = extract_pdf_text(file)
                        if text and not text.startswith("Error"):
                            chunks = process_text(text)
                            all_texts.extend(chunks)
                    
                    elif file.name.endswith(('.csv', '.xlsx', '.xls')):
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:
                            df = pd.read_excel(file)
                        
                        locations = extract_locations(df)
                        all_locations.extend(locations)
                        
                        text = df.to_string()
                        chunks = process_text(text)
                        all_texts.extend(chunks)
                
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(files))
            
            # Update session state
            if all_texts:
                st.session_state.knowledge_base.extend(all_texts)
            if all_locations:
                st.session_state.location_data.extend(all_locations)
            
            status_text.success(f"Processed {len(files)} files. Added {len(all_texts)} text chunks and {len(all_locations)} locations.")
    
    # Data management
    with st.expander("Data Management", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Text Chunks", len(st.session_state.knowledge_base))
            if st.button("View Sample Texts"):
                if st.session_state.knowledge_base:
                    st.write(st.session_state.knowledge_base[:3])
                else:
                    st.warning("No text data available")
        
        with col2:
            st.metric("Locations", len(st.session_state.location_data))
            if st.button("View Locations"):
                if st.session_state.location_data:
                    st.dataframe(pd.DataFrame(st.session_state.location_data))
                else:
                    st.warning("No location data available")
        
        if st.button("❌ Clear All Data", type="primary"):
            st.session_state.knowledge_base = []
            st.session_state.location_data = []
            st.session_state.chat_history = []
            st.success("All data cleared!")

def user_page():
    """Improved user chat interface"""
    st.title("💬 Campus Assistant")
    
    # Load model if available
    if not st.session_state.models_loaded:
        model = load_ai_model()
        st.session_state.search_model = model
        st.session_state.models_loaded = True
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for i, (user_msg, bot_msg, location) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(user_msg)
            
            with st.chat_message("assistant"):
                st.markdown(bot_msg)
                if location:
                    show_map(location)
    
    # Chat input
    with st.container():
        if prompt := st.chat_input("Ask about the campus..."):
            # Add user message to history
            st.session_state.chat_history.append((prompt, None, None))
            
            # Search knowledge base
            if st.session_state.knowledge_base:
                if hasattr(st.session_state, 'search_model') and st.session_state.search_model:
                    context = ai_search(prompt, st.session_state.knowledge_base, st.session_state.search_model)
                else:
                    context = basic_search(prompt, st.session_state.knowledge_base)
            else:
                context = []
            
            # Find location
            location = find_location(prompt, st.session_state.location_data)
            
            # Generate response
            api_key = get_api_key()
            response = generate_response(prompt, context, api_key)
            
            # Update the last message with the response
            if st.session_state.chat_history:
                st.session_state.chat_history[-1] = (prompt, response, location)
            
            st.rerun()

def credits():
    """Display credits in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.caption("""
    **Campus Assistant**  
    Developed with ❤️  
    v1.0 · © 2024
    """)

def main():
    """Main app with improved layout"""
    # Login section
    login_section()
    
    # Sidebar
    st.sidebar.title("🏫 Navigation")
    page = st.sidebar.radio("Go to", ["User Chat", "Admin Panel"], label_visibility="collapsed")
    
    # Credits
    credits()
    
    # Page routing
    if page == "Admin Panel":
        admin_page()
    else:
        user_page()

if __name__ == "__main__":
    main()
