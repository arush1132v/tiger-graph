#!/usr/bin/env python3
"""
Quick Setup Verification Script
Run this to check if everything is configured correctly
"""
import sys
from pathlib import Path


def check_files():
    """Check if all required files exist"""
    print("📁 Checking files...")
    
    required_files = [
        'config.py',
        'ingest_data.py',
        'pipeline_1_llm.py',
        'pipeline_2_rag.py',
        'pipeline_3_graphrag.py',
        'dashboard.py',
        'requirements.txt',
        '.env'
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        if '.env' in missing:
            print("  💡 Copy .env.example to .env and add your API key")
        return False
    
    print("  ✓ All files present\n")
    return True


def check_config():
    """Check if configuration is valid"""
    print("⚙️  Checking configuration...")
    
    try:
        from config import Config
        
        # Check API key
        if not Config.GOOGLE_API_KEY or Config.GOOGLE_API_KEY == "":
            print("  ✗ GOOGLE_API_KEY not set")
            print("  💡 Add your API key to .env file")
            return False
        
        if Config.GOOGLE_API_KEY == "a" or Config.GOOGLE_API_KEY == "your_google_api_key_here":
            print("  ✗ GOOGLE_API_KEY is placeholder")
            print("  💡 Replace with actual API key in .env file")
            return False
        
        print(f"  ✓ GOOGLE_API_KEY configured")
        print(f"  ✓ Vector DB path: {Config.CHROMA_PERSIST_DIR}")
        print(f"  ✓ LLM Model: {Config.LLM_MODEL}")
        print(f"  ✓ Embedding Model: {Config.EMBEDDING_MODEL}\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration error: {e}\n")
        return False


def check_dependencies():
    """Check if all dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        ('streamlit', 'streamlit'),
        ('langchain', 'langchain'),
        ('langchain_google_genai', 'langchain-google-genai'),
        ('langchain_chroma', 'langchain-chroma'),
        ('chromadb', 'chromadb'),
        ('google.generativeai', 'google-generativeai'),
    ]
    
    missing = []
    for module, package in required_packages:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (not installed)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("  💡 Run: pip install -r requirements.txt")
        return False
    
    print("  ✓ All dependencies installed\n")
    return True


def check_database():
    """Check if vector database exists"""
    print("💾 Checking vector database...")
    
    from config import Config
    db_path = Path(Config.CHROMA_PERSIST_DIR)
    
    if not db_path.exists():
        print(f"  ✗ Vector database not found at {Config.CHROMA_PERSIST_DIR}")
        print("  💡 Run: python ingest_data.py")
        return False
    
    print(f"  ✓ Vector database exists at {Config.CHROMA_PERSIST_DIR}\n")
    return True


def check_data_file():
    """Check if data file exists"""
    print("📊 Checking data file...")
    
    from config import Config
    data_file = Path(Config.DATA_FILE)
    
    if not data_file.exists():
        print(f"  ✗ Data file not found: {Config.DATA_FILE}")
        print("  💡 Place your creditcard.csv in the project directory")
        return False
    
    print(f"  ✓ Data file exists: {Config.DATA_FILE}\n")
    return True


def main():
    """Run all checks"""
    print("=" * 60)
    print("RAG Pipeline Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Files", check_files),
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
        ("Data File", check_data_file),
        ("Vector Database", check_database),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error during {name} check: {e}\n")
            results[name] = False
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    print()
    
    if all_passed:
        print("🎉 All checks passed! You're ready to go!")
        print()
        print("Next steps:")
        print("  1. Test pipelines: python pipeline_1_llm.py")
        print("  2. Run dashboard: streamlit run dashboard.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("Quick fixes:")
        print("  • Missing .env? Copy .env.example to .env and add your API key")
        print("  • Missing packages? Run: pip install -r requirements.txt")
        print("  • Missing database? Run: python ingest_data.py")
    
    print()
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
