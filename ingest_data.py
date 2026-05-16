"""
Data Ingestion Script for Vector Database
Loads CSV data, chunks it, and creates embeddings in ChromaDB
"""
import os
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from config import Config


def build_vector_db(data_file=None):
    """
    Build vector database from CSV file
    
    Args:
        data_file: Path to CSV file (uses Config.DATA_FILE if not provided)
    """
    try:
        # Validate configuration
        Config.validate()
        os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY
        
        # Use provided file or default from config
        file_path = data_file or Config.DATA_FILE
        
        # Check if file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        print(f"📁 Loading data from {file_path}...")
        loader = CSVLoader(file_path)
        docs = loader.load()
        print(f"✓ Loaded {len(docs)} documents")

        print("✂️  Chunking data...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)
        print(f"✓ Created {len(splits)} chunks")

        print("🔨 Building Vector DB with Gemini embeddings...")
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL),
            persist_directory=Config.CHROMA_PERSIST_DIR
        )
        print(f"✓ Database built successfully at {Config.CHROMA_PERSIST_DIR}")
        print(f"✓ Total documents in DB: {vectorstore._collection.count()}")
        
        return vectorstore
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your CSV file exists in the current directory")
        raise
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error during database build: {e}")
        raise


def verify_database():
    """Verify that the vector database exists and is accessible"""
    try:
        os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY
        vectorstore = Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR,
            embedding_function=GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL)
        )
        count = vectorstore._collection.count()
        print(f"✓ Database verified: {count} documents found")
        return True
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("RAG Pipeline - Data Ingestion")
    print("=" * 50)
    
    # Check if database already exists
    if Path(Config.CHROMA_PERSIST_DIR).exists():
        print(f"⚠️  Database already exists at {Config.CHROMA_PERSIST_DIR}")
        response = input("Do you want to rebuild it? (yes/no): ").lower()
        if response != 'yes':
            print("Skipping build. Verifying existing database...")
            verify_database()
            exit(0)
    
    # Build the database
    build_vector_db()
    
    print("\n✅ Ingestion complete! You can now run the pipelines.")
