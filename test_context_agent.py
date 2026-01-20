
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from v2_engine.agents.context_agent import run_context_agent

def test_extraction():
    # Load mock process
    with open("mock_processo.txt", "r") as f:
        text = f.read()
    
    # Get Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment.")
        return

    print("🚀 Initializing Context Agent Test...")
    print(f"📄 Mock Process Length: {len(text)}")
    
    try:
        result = run_context_agent(text, api_key)
        print("\n✅ Result:")
        print(result)
        
        # Validation
        if "fatos_principais" in result and len(result["fatos_principais"]) > 10:
             print("SUCCESS: Facts extracted.")
        else:
             print("FAILURE: Facts missing or empty.")
             
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extraction()
