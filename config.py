"""
Configuration file for RAG Pipeline Comparison
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Centralized configuration management"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # Vector Database Settings
    CHROMA_PERSIST_DIR = "./chroma_db"
    
    # Embedding Model
    EMBEDDING_MODEL = "models/text-embedding-004"
    
    # LLM Settings
    LLM_MODEL = "gemini-1.5-flash"
    LLM_TEMPERATURE = 0
    
    # Retrieval Settings
    RETRIEVAL_K = 4  # Number of documents to retrieve
    
    # Data Ingestion Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Data file
    DATA_FILE = "creditcard.csv"
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in .env file or environment."
            )
        return True
