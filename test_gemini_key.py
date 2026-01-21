from langchain_google_genai import ChatGoogleGenerativeAI
import sys

def test_key(api_key):
    print(f"🔑 Testando API Key: {api_key}")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key)
        response = llm.invoke("Say 'Hello' if this works.")
        print("✅ SUCESSO! A chave retornou:")
        print(response.content)
    except Exception as e:
        print("\n❌ ERRO NA CHAVE:")
        print(str(e))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_key(sys.argv[1])
    else:
        print("Uso: python3 test_gemini_key.py <SUA_API_KEY>")
