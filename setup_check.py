"""Setup verification script to check if all dependencies and configurations are correct."""
import sys
import os

def check_imports():
    """Check if all required packages can be imported."""
    print("Checking imports...")
    required_packages = [
        "langchain",
        "langchain_openai",
        "langchain_community",
        "langgraph",
        "langsmith",
        "qdrant_client",
        "streamlit",
        "pypdf",
        "dotenv",
        "requests",
        "pytest",
        "openai"
    ]
    
    failed = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            failed.append(package)
    
    return len(failed) == 0

def check_env_file():
    """Check if .env file exists and has required keys."""
    print("\nChecking environment configuration...")
    if not os.path.exists(".env"):
        print("  ✗ .env file not found")
        print("  → Create a .env file with required API keys")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_keys = [
        "OPENAI_API_KEY",
        "OPENWEATHERMAP_API_KEY"
    ]
    
    optional_keys = [
        "LANGSMITH_API_KEY",
        "QDRANT_HOST",
        "QDRANT_PORT"
    ]
    
    missing = []
    for key in required_keys:
        if not os.getenv(key):
            missing.append(key)
            print(f"  ✗ {key} - NOT SET")
        else:
            print(f"  ✓ {key}")
    
    for key in optional_keys:
        if os.getenv(key):
            print(f"  ✓ {key}")
        else:
            print(f"  ⚠ {key} - Optional (not set)")
    
    return len(missing) == 0

def check_qdrant():
    """Check if Qdrant is accessible."""
    print("\nChecking Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        from config import QDRANT_HOST, QDRANT_PORT
        
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections()
        print(f"  ✓ Qdrant connected at {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"  → Found {len(collections.collections)} collection(s)")
        return True
    except Exception as e:
        print(f"  ✗ Qdrant connection failed: {str(e)}")
        print("  → Make sure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)")
        return False

def main():
    """Run all checks."""
    print("=" * 50)
    print("AI Assignment Setup Verification")
    print("=" * 50)
    
    all_ok = True
    
    # Check imports
    if not check_imports():
        all_ok = False
        print("\n→ Install missing packages: pip install -r requirements.txt")
    
    # Check environment
    if not check_env_file():
        all_ok = False
    
    # Check Qdrant
    qdrant_ok = check_qdrant()
    if not qdrant_ok:
        all_ok = False
    
    print("\n" + "=" * 50)
    if all_ok:
        print("✓ All checks passed! You're ready to run the application.")
        print("\nTo start the Streamlit app:")
        print("  streamlit run app.py")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
    print("=" * 50)

if __name__ == "__main__":
    main()

