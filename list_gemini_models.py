import google.generativeai as genai
import sys
import os

def list_models(api_key):
    print(f"🔑 Listando Modelos para API Key...")
    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        found = False
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                found = True
        
        if not found:
            print("⚠️ Nenhum modelo com suporte a 'generateContent' encontrado.")
            
    except Exception as e:
        print("\n❌ ERRO AO LISTAR:")
        print(str(e))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        list_models(sys.argv[1])
    else:
        print("Uso: python3 list_gemini_models.py <SUA_API_KEY>")
