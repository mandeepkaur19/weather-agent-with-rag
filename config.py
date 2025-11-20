"""Configuration settings for the AI Assignment application."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# LangSmith Configuration
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ai-assignment")
LANGSMITH_TRACING_V2 = "true"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = "pdf_documents"

# OpenAI Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

# Application Configuration
PDF_UPLOAD_DIR = "uploads"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

