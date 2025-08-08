import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import hashlib

# Set page config
st.set_page_config(
    page_title="Campus Assistant AI",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful UI
def load_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        display: flex;
        align-items: flex-start;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
    }
    
    .chat-input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        border-radius: 15px 15px 0 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 500;
        transition: all 0.3s ease;
        margin: 0.25rem;
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .location-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    .map-button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 500;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
    }
    
    .map-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
        text-decoration: none;
        color: white;
    }
    
    .search-stats {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 1rem;
        color: #666;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #2d5a27;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .error-message {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state with enhanced structure
def init_session_state():
    defaults = {
        'knowledge_base': [],
        'location_data': [],
        'chat_history': [],
        'models_loaded': False,
        'search_index': None,
        'search_stats': {'total_queries': 0, 'successful_matches': 0},
        'user_preferences': {'response_style': 'friendly', 'show_sources': False},
        'analytics': {'popular_queries': {}, 'location_requests': {}},
        'data_sources': {},
        'processed_files': [],
        'last_activity': datetime.now()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Load dependencies with enhanced error handling
@st.cache_resource(show_spinner=False)
def load_dependencies():
    """Load optional dependencies with enhanced reporting"""
    deps = {
        'pdf': False, 'transformers': False, 'faiss': False, 
        'folium': False, 'langdetect': False, 'groq': False,
        'details': {}
    }
    
    # Test each dependency
    try:
        import PyPDF2
        deps['pdf'] = True
        deps['details']['pdf'] = "✓ PDF processing available"
    except ImportError:
        deps['details']['pdf'] = "⚠ Install PyPDF2 for PDF support"
    
    try:
        from sentence_transformers import SentenceTransformer
        deps['transformers'] = True
        deps['details']['transformers'] = "✓ AI-powered search available"
    except ImportError:
        deps['details']['transformers'] = "⚠ Install sentence-transformers for AI search"
    
    try:
        import faiss
        deps['faiss'] = True
        deps['details']['faiss'] = "✓ Vector search available"
    except ImportError:
        deps['details']['faiss'] = "⚠ Install faiss-cpu for advanced search"
    
    try:
        import folium
        from streamlit_folium import st_folium
        deps['folium'] = True
        deps['details']['folium'] = "✓ Interactive maps available"
    except ImportError:
        deps['details']['folium'] = "⚠ Install folium and streamlit-folium for maps"
    
    try:
        from groq import Groq
        deps['groq'] = True
        deps['details']['groq'] = "✓ AI chat available"
    except ImportError:
        deps['details']['groq'] = "⚠ Install groq for AI responses"
    
    try:
        from langdetect import detect
        deps['langdetect'] = True
    except ImportError:
        pass
    
    return deps

@st.cache_resource(show_spinner=False)
def load_ai_model():
    """Load AI model with progress indication"""
    try:
        from sentence_transformers import SentenceTransformer
        with st.spinner("🧠 Loading AI model..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        return None

def get_api_key():
    """Enhanced API key retrieval"""
    try:
        # Try secrets first
        if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
        
        # Try environment
        import os
        return os.getenv("GROQ_API_KEY")
    except:
        return None

def enhanced_pdf_extraction(pdf_file):
    """Enhanced PDF text extraction with better formatting"""
    deps = load_dependencies()
    if not deps['pdf']:
        return "PDF processing not available. Please install PyPDF2."
    
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                # Clean up text formatting
                page_text = re.sub(r'\n+', '\n', page_text)
                page_text = re.sub(r' +', ' ', page_text)
                text_parts.append(f"Page {page_num + 1}:\n{page_text}")
        
        full_text = '\n\n'.join(text_parts)
        return full_text.strip()
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def intelligent_text_processing(text: str, source_name: str = "") -> List[Dict]:
    """Enhanced text processing with metadata"""
    if not text or len(text.strip()) < 10:
        return []
    
    # Smart sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would make chunk too long, save current chunk
        if len(current_chunk) + len(sentence) > 400 and current_chunk:
            if len(current_chunk) >= 50:  # Only save if meaningful length
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': source_name,
                    'length': len(current_chunk),
                    'timestamp': datetime.now(),
                    'chunk_id': hashlib.md5(current_chunk.encode()).hexdigest()[:8]
                })
            current_chunk = sentence
        else:
            current_chunk = f"{current_chunk} {sentence}".strip()
    
    # Add final chunk
    if current_chunk and len(current_chunk) >= 50:
        chunks.append({
            'text': current_chunk.strip(),
            'source': source_name,
            'length': len(current_chunk),
            'timestamp': datetime.now(),
            'chunk_id': hashlib.md5(current_chunk.encode()).hexdigest()[:8]
        })
    
    return chunks

def smart_location_extraction(df: pd.DataFrame, filename: str = "") -> List[Dict]:
    """Enhanced location extraction with fuzzy matching"""
    locations = []
    
    if df.empty:
        return locations
    
    # Normalize column names
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.lower().str.strip()
    
    # Enhanced column mapping
    column_mappings = {
        'name': ['name', 'location', 'place', 'building', 'facility', 'venue', 'site', 'area'],
        'latitude': ['latitude', 'lat', 'y', 'y_coord', 'lat_coord'],
        'longitude': ['longitude', 'lng', 'lon', 'x', 'x_coord', 'lng_coord', 'long'],
        'description': ['description', 'desc', 'info', 'details', 'about', 'summary'],
        'category': ['category', 'type', 'kind', 'class', 'group'],
        'address': ['address', 'location_address', 'full_address', 'addr'],
        'phone': ['phone', 'contact', 'telephone', 'tel', 'phone_number'],
        'hours': ['hours', 'timing', 'schedule', 'open_hours', 'operating_hours']
    }
    
    # Find matching columns
    matched_cols = {}
    for target, variants in column_mappings.items():
        for variant in variants:
            if variant in df_clean.columns:
                matched_cols[target] = variant
                break
    
    # Must have name and coordinates
    if 'name' not in matched_cols or 'latitude' not in matched_cols or 'longitude' not in matched_cols:
        return locations
    
    # Extract locations with enhanced data
    for idx, row in df_clean.iterrows():
        try:
            lat = float(row[matched_cols['latitude']])
            lon = float(row[matched_cols['longitude']])
            
            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
                
            name = str(row[matched_cols['name']]).strip()
            if not name or name.lower() in ['nan', 'none', '']:
                continue
            
            location = {
                'name': name,
                'latitude': lat,
                'longitude': lon,
                'source_file': filename,
                'id': f"{filename}_{idx}",
                'search_terms': name.lower().split()
            }
            
            # Add optional fields
            for field in ['description', 'category', 'address', 'phone', 'hours']:
                if field in matched_cols:
                    value = str(row[matched_cols[field]]).strip()
                    if value and value.lower() not in ['nan', 'none', '']:
                        location[field] = value
                        if field == 'description':
                            location['search_terms'].extend(value.lower().split())
            
            locations.append(location)
            
        except (ValueError, TypeError, KeyError) as e:
            continue
    
    return locations

def fuzzy_location_search(query: str, locations: List[Dict], threshold: float = 0.4) -> List[Dict]:
    """Find locations using fuzzy string matching"""
    if not query or not locations:
        return []
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    scored_locations = []
    
    for location in locations:
        max_score = 0
        
        # Direct name matching
        name_score = SequenceMatcher(None, query_lower, location['name'].lower()).ratio()
        max_score = max(max_score, name_score)
        
        # Word-based matching
        location_words = set(location['search_terms'])
        word_overlap = len(query_words.intersection(location_words)) / max(len(query_words), 1)
        max_score = max(max_score, word_overlap)
        
        # Partial matching
        for word in query_words:
            for loc_word in location_words:
                if len(word) > 3 and len(loc_word) > 3:
                    partial_score = SequenceMatcher(None, word, loc_word).ratio()
                    if partial_score > 0.8:
                        max_score = max(max_score, partial_score * 0.8)
        
        if max_score >= threshold:
            location_copy = location.copy()
            location_copy['match_score'] = max_score
            scored_locations.append(location_copy)
    
    # Sort by score and return top matches
    scored_locations.sort(key=lambda x: x['match_score'], reverse=True)
    return scored_locations[:3]

def enhanced_search(query: str, knowledge_base: List[Dict], model=None, k: int = 5) -> List[Dict]:
    """Enhanced search with multiple strategies"""
    if not query or not knowledge_base:
        return []
    
    deps = load_dependencies()
    
    # Try AI search first
    if deps['transformers'] and deps['faiss'] and model:
        try:
            return ai_semantic_search(query, knowledge_base, model, k)
        except:
            pass
    
    # Fallback to enhanced keyword search
    return enhanced_keyword_search(query, knowledge_base, k)

def ai_semantic_search(query: str, knowledge_base: List[Dict], model, k: int = 5) -> List[Dict]:
    """AI-powered semantic search with metadata"""
    try:
        import faiss
        
        texts = [chunk['text'] for chunk in knowledge_base]
        
        # Generate embeddings
        text_embeddings = model.encode(texts)
        query_embedding = model.encode([query])
        
        # Create and populate index
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        faiss.normalize_L2(text_embeddings)
        faiss.normalize_L2(query_embedding)
        index.add(text_embeddings.astype('float32'))
        
        # Search
        scores, indices = index.search(query_embedding.astype('float32'), min(k, len(texts)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(knowledge_base) and score > 0.3:  # Threshold for relevance
                result = knowledge_base[idx].copy()
                result['relevance_score'] = float(score)
                result['search_method'] = 'semantic'
                results.append(result)
        
        return results
    except Exception as e:
        return enhanced_keyword_search(query, knowledge_base, k)

def enhanced_keyword_search(query: str, knowledge_base: List[Dict], k: int = 5) -> List[Dict]:
    """Enhanced keyword search with scoring"""
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    
    scored_results = []
    
    for chunk in knowledge_base:
        text_lower = chunk['text'].lower()
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Calculate various scores
        exact_matches = sum(1 for word in query_words if word in text_lower)
        word_overlap = len(query_words.intersection(text_words))
        
        # Phrase matching bonus
        phrase_bonus = 0
        if len(query_words) > 1 and query.lower() in text_lower:
            phrase_bonus = 0.5
        
        # Calculate final score
        base_score = (exact_matches + word_overlap) / max(len(query_words), 1)
        final_score = base_score + phrase_bonus
        
        if final_score > 0:
            result = chunk.copy()
            result['relevance_score'] = final_score
            result['search_method'] = 'keyword'
            scored_results.append(result)
    
    # Sort by relevance and return top k
    scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return scored_results[:k]

def generate_intelligent_response(query: str, context: List[Dict], locations: List[Dict], api_key: str) -> str:
    """Generate contextual, natural responses"""
    deps = load_dependencies()
    
    # Prepare context information
    context_texts = [item['text'] for item in context[:3]] if context else []
    location_info = []
    
    if locations:
        for loc in locations[:2]:  # Limit to 2 locations
            info = f"Location: {loc['name']}"
            if 'description' in loc:
                info += f" - {loc['description']}"
            location_info.append(info)
    
    # Fallback response if no API
    if not deps['groq'] or not api_key:
        if context_texts or location_info:
            response_parts = []
            if location_info:
                response_parts.extend(location_info)
            if context_texts:
                response_parts.append(f"Based on the available information: {context_texts[0][:150]}...")
            return " ".join(response_parts)
        return "I'd be happy to help you with campus information. Please make sure you've uploaded relevant documents and configured the AI settings."
    
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Build comprehensive context
        full_context = []
        if location_info:
            full_context.extend(location_info)
        if context_texts:
            full_context.extend(context_texts)
        
        context_text = "\n".join(full_context) if full_context else "General campus knowledge"
        
        # Enhanced system prompt
        system_prompt = """You are a helpful and knowledgeable campus assistant AI. Your role is to provide accurate, friendly, and natural responses about campus information.

Guidelines:
- Be conversational and helpful, like talking to a friend
- Provide specific, actionable information when possible
- Never mention "based on the data" or "according to the information provided"
- Respond naturally as if you inherently know this information
- Be concise but comprehensive
- If asked about locations, provide helpful details
- If you don't have specific information, admit it politely and suggest alternatives
- Use a warm, approachable tone
- Focus on being genuinely helpful to students, faculty, and visitors"""

        user_prompt = f"""Context information:
{context_text}

User question: {query}

Provide a helpful, natural response that directly answers their question. Be friendly and conversational."""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # Enhanced fallback
        if location_info or context_texts:
            response = "Here's what I found: "
            if location_info:
                response += f"{location_info[0]}. "
            if context_texts:
                response += f"{context_texts[0][:100]}..."
            return response
        return "I'm here to help with campus information. Could you try rephrasing your question or check that the relevant data has been uploaded?"

def create_beautiful_map(location: Dict):
    """Create an enhanced map with navigation"""
    deps = load_dependencies()
    
    if not deps['folium']:
        # Fallback location display
        st.markdown(f"""
        <div class="location-card">
            <h3>📍 {location['name']}</h3>
            <p><strong>Coordinates:</strong> {location['latitude']:.6f}, {location['longitude']:.6f}</p>
            {f"<p><strong>Description:</strong> {location.get('description', 'No description available')}</p>" if location.get('description') else ""}
            {f"<p><strong>Category:</strong> {location['category']}</p>" if location.get('category') else ""}
            {f"<p><strong>Address:</strong> {location['address']}</p>" if location.get('address') else ""}
        </div>
        """, unsafe_allow_html=True)
        return
    
    try:
        import folium
        from streamlit_folium import st_folium
        
        # Create map centered on location
        m = folium.Map(
            location=[location['latitude'], location['longitude']], 
            zoom_start=17,
            tiles='OpenStreetMap'
        )
        
        # Add marker with just the name
        folium.Marker(
            [location['latitude'], location['longitude']],
            popup=location['name'],
            tooltip=location['name'],
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
        # Display map
        map_data = st_folium(m, height=400, width=700)
        
        # Navigation button
        google_maps_url = f"https://www.google.com/maps/dir/?api=1&destination={location['latitude']},{location['longitude']}"
        
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <a href="{google_maps_url}" target="_blank" class="map-button">
                🧭 Navigate to {location['name']}
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional location info
        if any(key in location for key in ['description', 'address', 'phone', 'hours']):
            with st.expander("📋 More Details"):
                if location.get('description'):
                    st.write(f"**Description:** {location['description']}")
                if location.get('address'):
                    st.write(f"**Address:** {location['address']}")
                if location.get('phone'):
                    st.write(f"**Phone:** {location['phone']}")
                if location.get('hours'):
                    st.write(f"**Hours:** {location['hours']}")
        
    except Exception as e:
        st.error(f"Map error: {str(e)}")

def update_analytics(query: str, found_locations: bool = False):
    """Update usage analytics"""
    # Update query analytics
    query_lower = query.lower()
    if query_lower in st.session_state.analytics['popular_queries']:
        st.session_state.analytics['popular_queries'][query_lower] += 1
    else:
        st.session_state.analytics['popular_queries'][query_lower] = 1
    
    # Update search stats
    st.session_state.search_stats['total_queries'] += 1
    if found_locations:
        st.session_state.search_stats['successful_matches'] += 1
    
    # Keep only top 50 queries to prevent memory issues
    if len(st.session_state.analytics['popular_queries']) > 50:
        sorted_queries = sorted(
            st.session_state.analytics['popular_queries'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]
        st.session_state.analytics['popular_queries'] = dict(sorted_queries)

def admin_dashboard():
    """Enhanced admin interface"""
    st.title("🔧 Admin Dashboard")
    
    # File upload section
    st.subheader("📁 File Management")
    
    with st.container():
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        files = st.file_uploader(
            "Upload Campus Data Files",
            type=['pdf', 'csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload PDFs for text content or CSV/Excel files for location data"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if files:
        process_uploaded_files(files)
    
    # Statistics dashboard
    st.subheader("📊 Analytics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(st.session_state.knowledge_base)}</h3>
            <p>Text Chunks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(st.session_state.location_data)}</h3>
            <p>Locations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{st.session_state.search_stats['total_queries']}</h3>
            <p>Total Queries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        success_rate = (st.session_state.search_stats['successful_matches'] / 
                       max(st.session_state.search_stats['total_queries'], 1)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>{success_rate:.1f}%</h3>
            <p>Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data management
    if st.session_state.knowledge_base or st.session_state.location_data:
        st.subheader("🗂️ Data Management")
        
        # Show processed files
        if st.session_state.processed_files:
            st.write("**Processed Files:**")
            for file_info in st.session_state.processed_files:
                st.write(f"- {file_info['name']} ({file_info['type']}) - {file_info['timestamp']}")
        
        # Location data viewer
        if st.session_state.location_data:
            st.write("**Location Data:**")
            location_df = pd.DataFrame(st.session_state.location_data)
            st.dataframe(location_df, use_container_width=True)
        
        # Popular queries
        if st.session_state.analytics['popular_queries']:
            st.subheader("🔍 Popular Queries")
            popular_queries = sorted(
                st.session_state.analytics['popular_queries'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for query, count in popular_queries:
                st.write(f"- {query} ({count} times)")
        
        # Export functionality
        st.subheader("📤 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Chat History", type="secondary"):
                export_chat_history()
        
        with col2:
            if st.button("Export Analytics", type="secondary"):
                export_analytics()
        
        # Clear data options
        st.subheader("🗑️ Data Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
        
        with col2:
            if st.button("Clear Knowledge Base", type="secondary"):
                st.session_state.knowledge_base = []
                st.success("Knowledge base cleared!")
                st.rerun()
        
        with col3:
            if st.button("Clear All Data", type="primary"):
                clear_all_data()
                st.rerun()

def process_uploaded_files(files):
    """Process uploaded files with enhanced feedback"""
    progress_bar = st.progress(0)
    status_container = st.container()
    
    all_chunks = []
    all_locations = []
    
    for i, file in enumerate(files):
        with status_container:
            st.write(f"📄 Processing {file.name}...")
        
        try:
            if file.type == "application/pdf":
                text = enhanced_pdf_extraction(file)
                if text and not text.startswith("Error"):
                    chunks = intelligent_text_processing(text, file.name)
                    all_chunks.extend(chunks)
                    
                    # Record file processing
                    st.session_state.processed_files.append({
                        'name': file.name,
                        'type': 'PDF',
                        'chunks': len(chunks),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    
                    st.success(f"✅ {file.name}: {len(chunks)} text chunks extracted")
                else:
                    st.error(f"❌ Failed to process {file.name}")
            
            elif file.name.lower().endswith(('.csv', '.xlsx', '.xls')):
                # Read file
                if file.name.lower().endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                
                # Extract locations
                locations = smart_location_extraction(df, file.name)
                all_locations.extend(locations)
                
                # Also process as text for general queries
                if not df.empty:
                    text_content = df.to_string()
                    chunks = intelligent_text_processing(text_content, file.name)
                    all_chunks.extend(chunks)
                
                # Record file processing
                st.session_state.processed_files.append({
                    'name': file.name,
                    'type': 'Data',
                    'locations': len(locations),
                    'chunks': len(chunks) if 'chunks' in locals() else 0,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                
                st.success(f"✅ {file.name}: {len(locations)} locations, {len(chunks) if 'chunks' in locals() else 0} text chunks")
                
                # Show preview
                if not df.empty:
                    with st.expander(f"Preview: {file.name}"):
                        st.dataframe(df.head(), use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(files))
    
    # Update session state
    if all_chunks:
        st.session_state.knowledge_base.extend(all_chunks)
    if all_locations:
        st.session_state.location_data.extend(all_locations)
    
    if all_chunks or all_locations:
        st.markdown(f"""
        <div class="success-message">
            🎉 Processing Complete! Added {len(all_chunks)} text chunks and {len(all_locations)} locations to the knowledge base.
        </div>
        """, unsafe_allow_html=True)

def export_chat_history():
    """Export chat history as JSON"""
    if st.session_state.chat_history:
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'total_conversations': len(st.session_state.chat_history),
            'conversations': []
        }
        
        for user_msg, bot_msg, locations in st.session_state.chat_history:
            conversation = {
                'user_message': user_msg,
                'assistant_response': bot_msg,
                'locations_found': len(locations) if locations else 0
            }
            export_data['conversations'].append(conversation)
        
        # Create download
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="Download Chat History",
            data=json_str,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

def export_analytics():
    """Export analytics data"""
    analytics_data = {
        'timestamp': datetime.now().isoformat(),
        'search_stats': st.session_state.search_stats,
        'popular_queries': st.session_state.analytics['popular_queries'],
        'total_knowledge_chunks': len(st.session_state.knowledge_base),
        'total_locations': len(st.session_state.location_data)
    }
    
    json_str = json.dumps(analytics_data, indent=2)
    st.download_button(
        label="Download Analytics",
        data=json_str,
        file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )

def clear_all_data():
    """Clear all application data"""
    st.session_state.knowledge_base = []
    st.session_state.location_data = []
    st.session_state.chat_history = []
    st.session_state.analytics = {'popular_queries': {}, 'location_requests': {}}
    st.session_state.search_stats = {'total_queries': 0, 'successful_matches': 0}
    st.session_state.processed_files = []
    st.success("All data cleared successfully!")

def user_chat_interface():
    """Enhanced user chat interface"""
    st.title("💬 Campus Assistant AI")
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Initializing AI systems..."):
            model = load_ai_model()
            st.session_state.search_model = model
            st.session_state.models_loaded = True
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="success-message">
                👋 Hello! I'm your Campus Assistant AI. I can help you with:
                <ul>
                    <li>🏢 Finding locations and buildings</li>
                    <li>📚 Campus information and services</li>
                    <li>🗺️ Navigation and directions</li>
                    <li>❓ General campus questions</li>
                </ul>
                Just ask me anything about the campus!
            </div>
            """, unsafe_allow_html=True)
        
        # Display conversation history
        for user_msg, bot_msg, locations in st.session_state.chat_history:
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <div>
                    <strong>You:</strong><br>
                    {user_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant message
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div>
                    <strong>Campus Assistant:</strong><br>
                    {bot_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show locations if found
            if locations:
                st.write("📍 **Locations Found:**")
                for location in locations:
                    create_beautiful_map(location)
    
    # Chat input
    st.markdown("---")
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask me anything about the campus...",
            placeholder="e.g., Where is the library? How do I get to the cafeteria?",
            key="user_chat_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("Send 🚀", type="primary")
        with col2:
            if st.form_submit_button("Clear Chat 🗑️", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
    
    if submitted and user_input.strip():
        process_user_query(user_input.strip())

def process_user_query(query: str):
    """Process user query with enhanced intelligence"""
    with st.spinner("🤔 Thinking..."):
        # Search knowledge base
        context = []
        if st.session_state.knowledge_base:
            context = enhanced_search(
                query, 
                st.session_state.knowledge_base, 
                st.session_state.search_model if hasattr(st.session_state, 'search_model') else None
            )
        
        # Find locations with fuzzy matching
        locations = fuzzy_location_search(query, st.session_state.location_data)
        
        # Generate response
        api_key = get_api_key()
        response = generate_intelligent_response(query, context, locations, api_key)
        
        # Update analytics
        update_analytics(query, bool(locations))
        
        # Add to chat history
        st.session_state.chat_history.append((query, response, locations))
        
        # Show search statistics
        if context or locations:
            relevance_info = []
            if context:
                relevance_info.append(f"{len(context)} relevant text sources")
            if locations:
                relevance_info.append(f"{len(locations)} location matches")
            
            st.markdown(f"""
            <div class="search-stats">
                📊 Found: {' and '.join(relevance_info)}
            </div>
            """, unsafe_allow_html=True)
        
        st.rerun()

def system_status_sidebar():
    """Enhanced system status in sidebar"""
    st.sidebar.title("🏫 Campus Assistant AI")
    
    # System status
    st.sidebar.subheader("🔧 System Status")
    deps = load_dependencies()
    
    status_items = [
        ("PDF Processing", deps['pdf'], "Process PDF documents"),
        ("AI Search", deps['transformers'] and deps['faiss'], "Semantic search capabilities"),
        ("Interactive Maps", deps['folium'], "Location mapping"),
        ("AI Chat", deps['groq'], "Natural language responses")
    ]
    
    for name, status, description in status_items:
        if status:
            st.sidebar.markdown(f"""
            <div class="status-card">
                ✅ <strong>{name}</strong><br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"""
            <div class="status-card" style="border-left-color: #ff9800;">
                ⚠️ <strong>{name}</strong><br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # API status
    st.sidebar.subheader("🔑 API Configuration")
    api_key = get_api_key()
    if api_key:
        st.sidebar.success("✅ AI Chat Enabled")
    else:
        st.sidebar.warning("⚠️ AI Chat Limited")
        st.sidebar.info("Add GROQ_API_KEY to secrets.toml for full AI capabilities")
    
    # Quick stats
    if st.session_state.knowledge_base or st.session_state.location_data:
        st.sidebar.subheader("📊 Quick Stats")
        st.sidebar.metric("Knowledge Base", f"{len(st.session_state.knowledge_base)} chunks")
        st.sidebar.metric("Locations", f"{len(st.session_state.location_data)} places")
        st.sidebar.metric("Queries Today", st.session_state.search_stats['total_queries'])
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")

def main():
    """Main application with enhanced UI"""
    load_css()
    init_session_state()
    
    # Sidebar
    system_status_sidebar()
    
    # Page navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Chat Interface", "⚙️ Admin Dashboard"],
        key="page_navigation"
    )
    
    # User preferences
    st.sidebar.subheader("⚙️ Preferences")
    st.session_state.user_preferences['response_style'] = st.sidebar.selectbox(
        "Response Style",
        ["friendly", "formal", "detailed"],
        index=0
    )
    
    st.session_state.user_preferences['show_sources'] = st.sidebar.checkbox(
        "Show Source Information",
        value=False
    )
    
    # Main content
    if "Admin Dashboard" in page:
        admin_dashboard()
    else:
        # Check if data is available
        if not st.session_state.knowledge_base and not st.session_state.location_data:
            st.warning("⚠️ No data loaded yet. Please go to the Admin Dashboard to upload campus data files.")
            if st.button("Go to Admin Dashboard", type="primary"):
                st.session_state.page_navigation = "⚙️ Admin Dashboard"
                st.rerun()
        
        user_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        🏫 Campus Assistant AI - Powered by Advanced NLP and Machine Learning
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
