
import os
import sys

# Ensure backend imports work
sys.path.append(os.getcwd())

from backend import process_documents, analyze_process
# Check if ollama is reachable? We'll see in the try block.

FILE_PATH = os.path.abspath("mock_processo.txt")
DB_TYPE = "processos"

def test_flow():
    print(f"--- 1. Ingesting {FILE_PATH} ---")
    try:
        # Ingest
        process_documents([FILE_PATH], db_type=DB_TYPE)
        print("✅ Ingestion successful.")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return

    print(f"\n--- 2. Running Analysis (Assistente Jurídico) ---")
    try:
        # Tenta usar Llama3 (Ollama local). Se falhar, avisa.
        # Usuario não deu API Key, então assume local.
        print("Attempting to connect to Ollama (mistral-nemo or llama3)...")
        
        # Test basic connection first? nah, let analyze_process handle it.
        # We prefer mistral-nemo as per recent request, but default might be llama3 in code if not passed.
        # backend.py signature: analyze_process(filename, db_type, api_key=None, model_name="llama3")
        
        result = analyze_process("mock_processo.txt", db_type=DB_TYPE, model_name="mistral-nemo")
        
        print("\n--- 3. RESULTADO DA ANÁLISE ---")
        print(result)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    test_flow()
