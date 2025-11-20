"""Streamlit UI application for the AI Assignment."""
import os
import time
from typing import Dict

import streamlit as st

from agent import AIAgent
from weather_service import WeatherService
from rag_service import RAGService
from vector_store import VectorStore
from pdf_processor import PDFProcessor
from evaluator import ResponseEvaluator
from config import PDF_UPLOAD_DIR


# Page configuration
st.set_page_config(
    page_title="AI Assignment - LangGraph Agent",
    page_icon="🤖",
    layout="wide",
    menu_items={"About": "LangGraph x LangChain assistant for weather + PDF QA"}
)

# Light styling helpers
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "theme_toggle" in st.session_state:
    st.session_state.dark_mode = st.session_state.theme_toggle


def _get_theme_palette(is_dark: bool) -> Dict[str, str]:
    """Return design tokens for current theme."""
    if is_dark:
        return {
            "page_bg": "#030712",
            "gradient_a": "#020617",
            "gradient_b": "#0f172a",
            "text": "#e2e8f0",
            "muted": "rgba(226, 232, 240, 0.72)",
            "panel_bg": "rgba(15, 23, 42, 0.82)",
            "panel_border": "rgba(148, 163, 184, 0.35)",
            "sidebar_bg": "rgba(2, 6, 23, 0.92)",
            "input_bg": "rgba(15, 23, 42, 0.95)",
            "accent": "#38bdf8",
            "accent_soft": "rgba(56, 189, 248, 0.18)",
            "accent_alt": "#8b5cf6",
            "shadow": "0 25px 65px rgba(2, 6, 23, 0.65)",
        }
    return {
        "page_bg": "#f8fafc",
        "gradient_a": "#f1f5f9",
        "gradient_b": "#e2e8f0",
        "text": "#0f172a",
        "muted": "rgba(15, 23, 42, 0.65)",
        "panel_bg": "#ffffff",
        "panel_border": "rgba(15, 23, 42, 0.08)",
        "sidebar_bg": "#f1f5f9",
        "input_bg": "#ffffff",
        "accent": "#0f62fe",
        "accent_soft": "rgba(15, 98, 254, 0.15)",
        "accent_alt": "#9333ea",
        "shadow": "0 20px 55px rgba(15, 23, 42, 0.12)",
    }


def _apply_theme_styles():
    palette = _get_theme_palette(st.session_state.dark_mode)
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');

        :root {{
            --page-bg: {palette["page_bg"]};
            --text-color: {palette["text"]};
            --muted-color: {palette["muted"]};
            --panel-bg: {palette["panel_bg"]};
            --panel-border: {palette["panel_border"]};
            --sidebar-bg: {palette["sidebar_bg"]};
            --input-bg: {palette["input_bg"]};
            --accent: {palette["accent"]};
            --accent-soft: {palette["accent_soft"]};
            --accent-alt: {palette["accent_alt"]};
            --card-shadow: {palette["shadow"]};
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Space Grotesk', sans-serif;
            background: var(--page-bg);
            color: var(--text-color);
        }}

        [data-testid="stAppViewContainer"] > .main {{
            padding-left: 2.5rem;
            padding-right: 2.5rem;
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            background:
                radial-gradient(circle at 12% 18%, rgba(56,189,248,0.12), transparent 38%),
                radial-gradient(circle at 88% 8%, rgba(139,92,246,0.15), transparent 32%),
                linear-gradient(135deg, {palette["gradient_a"]} 0%, {palette["gradient_b"]} 60%, {palette["page_bg"]} 100%);
        }}

        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg);
            border-right: 1px solid var(--panel-border);
        }}

        [data-testid="stAppViewContainer"] .block-container {{
            padding-top: 0;
            max-width: 1300px;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            box-shadow: var(--card-shadow);
            border-radius: 24px;
            padding: 1.25rem 1.5rem;
        }}

        .metric-card {{
            border-radius: 18px;
            padding: 1.25rem;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            box-shadow: var(--card-shadow);
            min-height: 150px;
        }}

        .top-nav-left {{
            display: flex;
            gap: 0.85rem;
            align-items: center;
        }}

        .logo-chip {{
            width: 54px;
            height: 54px;
            border-radius: 16px;
            background: var(--accent-soft);
            display: grid;
            place-items: center;
            font-size: 1.5rem;
            color: var(--accent);
        }}

        .eyebrow {{
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.2em;
            color: var(--muted-color);
            margin-bottom: 0.1rem;
        }}

        .nav-pills {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            justify-content: center;
        }}

        .nav-pill {{
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            border: 1px solid var(--panel-border);
            color: var(--muted-color);
            font-size: 0.82rem;
        }}

        .nav-actions {{
            display: flex;
            gap: 0.5rem;
            justify-content: flex-end;
            align-items: center;
        }}

        .nav-icon {{
            width: 46px;
            height: 46px;
            border-radius: 50%;
            border: 1px solid var(--panel-border);
            background: rgba(148, 163, 184, 0.08);
            color: var(--text-color);
            font-size: 1.2rem;
            display: grid;
            place-items: center;
        }}

        .route-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            border: 1px solid rgba(226, 232, 240, 0.4);
            background: rgba(148, 163, 184, 0.15);
            letter-spacing: 0.05em;
        }}

        .route-pill.weather {{
            background: rgba(56, 189, 248, 0.15);
            border-color: rgba(56, 189, 248, 0.5);
        }}

        .route-pill.rag {{
            background: rgba(168, 85, 247, 0.15);
            border-color: rgba(168, 85, 247, 0.5);
        }}

        [data-testid="stFileUploaderDropzone"] {{
            border: 1px dashed var(--panel-border);
            background: var(--input-bg);
            border-radius: 18px;
            padding: 1.2rem;
            transition: border 0.2s ease, box-shadow 0.2s ease, transform 0.2s;
        }}

        [data-testid="stFileUploaderDropzone"]:hover,
        [data-testid="stFileUploaderDropzone"]:focus-within {{
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-soft);
            transform: translateY(-2px);
        }}

        .upload-hint {{
            display: flex;
            gap: 0.9rem;
            align-items: center;
            margin-bottom: 0.6rem;
        }}

        .upload-icon {{
            width: 48px;
            height: 48px;
            border-radius: 15px;
            background: var(--accent-soft);
            display: grid;
            place-items: center;
            font-size: 1.35rem;
            color: var(--accent);
        }}

        form[data-testid="stForm"] textarea {{
            background: var(--input-bg);
            border-radius: 20px;
            border: 1px solid var(--panel-border);
            box-shadow: var(--card-shadow);
            color: var(--text-color);
            font-size: 1rem;
            padding: 0.75rem;
            min-height: 100px;
        }}

        form[data-testid="stForm"] textarea:focus {{
            border-color: var(--accent);
            outline: none;
        }}

        form[data-testid="stForm"] button[type="submit"] {{
            height: 100%;
            border-radius: 18px;
            border: none;
            background: linear-gradient(135deg, var(--accent), var(--accent-alt));
            color: white;
            font-weight: 600;
            box-shadow: 0 18px 40px rgba(3, 7, 18, 0.55);
        }}

        form[data-testid="stForm"] button[type="submit"]:hover {{
            filter: brightness(1.05);
        }}

        div[data-testid^="stChatMessage"] {{
            background: transparent !important;
        }}

        @keyframes fadeInUp {{
            from {{ transform: translateY(12px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}

        div[data-testid^="stChatMessage"] > div:nth-child(2) {{
            border-radius: 22px;
            padding: 0.95rem 1.2rem;
            border: 1px solid var(--panel-border);
            background: var(--panel-bg);
            box-shadow: var(--card-shadow);
            margin-bottom: 0.4rem;
            animation: fadeInUp 0.35s ease;
        }}

        div[data-testid="stChatMessageUser"] > div:nth-child(2),
        div[data-testid="stChatMessage-user"] > div:nth-child(2) {{
            margin-left: auto;
            border: none;
            color: #041019;
            background: linear-gradient(135deg, #38bdf8, #6366f1);
            box-shadow: 0 25px 50px rgba(3, 7, 18, 0.65);
        }}

        div[data-testid="stChatMessageUser"] > div:nth-child(2) .stMarkdown p,
        div[data-testid="stChatMessage-user"] > div:nth-child(2) .stMarkdown p {{
            color: #041019 !important;
        }}

        div[data-testid="stChatMessageAssistant"] > div:nth-child(2),
        div[data-testid="stChatMessage-assistant"] > div:nth-child(2) {{
            border-left: 4px solid var(--accent);
        }}

        .history-list {{
            list-style: none;
            padding-left: 0;
            margin: 0;
        }}

        .history-list li {{
            padding: 0.5rem 0;
            border-bottom: 1px dashed var(--panel-border);
            color: var(--muted-color);
            font-size: 0.9rem;
        }}

        .sidebar-nav {{
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }}

        .sidebar-item {{
            padding: 0.6rem 0.75rem;
            border-radius: 12px;
            border: 1px solid transparent;
            color: var(--muted-color);
        }}

        .sidebar-item.active {{
            border-color: var(--panel-border);
            background: var(--panel-bg);
            color: var(--text-color);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 16px;
            padding: 0.6rem 1.2rem;
            background: rgba(148, 163, 184, 0.08);
        }}

        .stButton > button {{
            border-radius: 18px;
            border: 1px solid var(--panel-border);
            background: rgba(148, 163, 184, 0.08);
            color: var(--text-color);
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }}

        .stButton > button:hover {{
            border-color: var(--accent);
            color: var(--accent);
            transform: translateY(-1px);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_theme_styles()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "route_stats" not in st.session_state:
    st.session_state.route_stats = {"weather": 0, "rag": 0}
if "last_route" not in st.session_state:
    st.session_state.last_route = "—"


def initialize_services():
    """Initialize all required services."""
    try:
        # Initialize services
        weather_service = WeatherService()
        vector_store = VectorStore()
        rag_service = RAGService(vector_store)
        agent = AIAgent(weather_service, rag_service)
        
        st.session_state.agent = agent
        st.session_state.vector_store_initialized = True
        return True
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        return False


def process_pdf_upload(uploaded_file):
    """Process uploaded PDF file and add to vector store."""
    try:
        # Save uploaded file
        os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(PDF_UPLOAD_DIR, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process PDF
        processor = PDFProcessor()
        chunks = processor.process_pdf(file_path)
        
        # Add to vector store
        vector_store = VectorStore()
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        vector_store.add_documents(texts, metadatas)
        
        st.session_state.documents.append(
            {
                "name": uploaded_file.name,
                "chunks": len(chunks),
            }
        )
        return True, f"Successfully processed {len(chunks)} chunks from {uploaded_file.name}"
    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"


# Main UI
st.title("🤖 AI Agentic Copilot - Weather + PDF RAG")
st.caption("Agentic workflow that blends real-time weather with Retrieval-Augmented Generation over your PDFs.")

# Status glance
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1])
with col1:
    st.markdown(
        f"""
        <div class='metric-card'>
            <h5>Knowledge Base</h5>
            <h2>{len(st.session_state.documents) or '0'}</h2>
            <p>PDFs processed</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class='metric-card'>
            <h5>Weather Answers</h5>
            <h2>{st.session_state.route_stats['weather']}</h2>
            <p>Calls to OpenWeather</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class='metric-card'>
            <h5>RAG Answers</h5>
            <h2>{st.session_state.route_stats['rag']}</h2>
            <p>PDF-grounded replies</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
        <div class='metric-card'>
            <h5>Last Route</h5>
            <h2>{st.session_state.last_route.upper()}</h2>
            <p>Decision node output</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 PDF Document Upload")
    st.markdown("Upload a PDF document to enable RAG-based question answering.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                success, message = process_pdf_upload(uploaded_file)
                if success:
                    st.success(message)
                    st.session_state.vector_store_initialized = True
                else:
                    st.error(message)
    
    st.markdown("---")
    st.markdown("### Configuration")
    st.info("""
    Make sure you have set up:
    - OpenAI API Key
    - OpenWeatherMap API Key
    - LangSmith API Key (optional)
    - Qdrant running on localhost:6333
    """)

# Initialize services
if st.session_state.agent is None:
    with st.spinner("Initializing services..."):
        if not initialize_services():
            st.error("Failed to initialize services. Please check your configuration.")
            st.stop()

# Main workspace tabs
chat_tab, kb_tab = st.tabs(["💬 Chat", "📚 Knowledge Base"])

with chat_tab:
    st.subheader("Ask about weather or your documents")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "route" in message:
                st.markdown(
                    f"<span class='route-pill {message['route']}'>Route · {message['route'].upper()}</span>",
                    unsafe_allow_html=True,
                )

    # Chat input
    if prompt := st.chat_input("Ask a question (e.g., 'What's the weather in Pune?' or 'Summarize the assignment')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent.process_query(prompt)
                    st.session_state.last_route = result["route"]
                    if result["route"] in st.session_state.route_stats:
                        st.session_state.route_stats[result["route"]] += 1
                    
                    # Display response
                    st.markdown(result["response"])
                    st.markdown(
                        f"<span class='route-pill {result['route']}'>Route · {result['route'].upper()}</span>",
                        unsafe_allow_html=True,
                    )
                    
                    # Evaluate with LangSmith (if configured)
                    try:
                        evaluator = ResponseEvaluator()
                        evaluator.evaluate_response(
                            query=prompt,
                            response=result["response"],
                            route=result["route"]
                        )
                    except Exception:
                        pass
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "route": result["route"]
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

with kb_tab:
    st.subheader("Knowledge base monitor")
    if st.session_state.documents:
        st.success("These PDFs are indexed and ready for RAG:")
        kb_data = [
            {"Document": doc["name"], "Chunks": doc["chunks"]}
            for doc in st.session_state.documents
        ]
        st.table(kb_data)
    else:
        st.info("No PDFs have been ingested yet. Upload one from the sidebar to enable RAG.")
    
    st.markdown("#### Suggested prompts")
    st.markdown(
        """
        - *“Summarize the assignment in bullet points.”*  
        - *“What are the deliverables mentioned in the document?”*  
        - *“What is the evaluation criteria?”*
        """
    )

# Footer
st.markdown("---")
st.markdown("""
### Features
- **LangGraph Agent**: Intelligent routing between weather API and RAG
- **Weather API**: Real-time weather data from OpenWeatherMap
- **RAG System**: Question answering from PDF documents using embeddings and vector search
- **LangSmith Integration**: Response evaluation and logging
- **Streamlit UI**: Interactive chat interface
""")

