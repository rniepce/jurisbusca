"""
Quick test: validates Azure OpenAI GPT-5.2-chat and GPT-4.1-mini connections.
Usage: python3 test_azure_gpt.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

def test_model(deployment, use_temp=False):
    key = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")

    if not key or not endpoint:
        print("❌ Variáveis AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT não configuradas no .env")
        return False

    print(f"🔗 Endpoint: {endpoint}")
    print(f"📦 Deployment: {deployment}")
    print(f"🔑 API Key: {key[:8]}...{key[-4:]}")
    print()

    try:
        kwargs = dict(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=key,
            api_version="2024-12-01-preview",
        )
        if use_temp:
            kwargs["temperature"] = 0.1

        llm = AzureChatOpenAI(**kwargs)

        print("📤 Enviando prompt de teste...")
        response = llm.invoke([HumanMessage(content="Diga apenas: 'Conexão OK!'")])

        print(f"✅ Resposta: {response.content}")
        print()
        return True

    except Exception as e:
        print(f"❌ Erro: {e}")
        print()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTE 1: GPT-5.2-chat (sem temperature)")
    print("=" * 50)
    ok1 = test_model("gpt-5.2-chat", use_temp=False)

    print("=" * 50)
    print("TESTE 2: GPT-4.1-mini (com temperature)")
    print("=" * 50)
    ok2 = test_model("gpt-5.4-mini", use_temp=True)

    print("=" * 50)
    if ok1 and ok2:
        print("🎉 Todos os modelos validados com sucesso!")
    elif ok1:
        print("⚠️ GPT-5.2 OK, mas GPT-4.1-mini falhou.")
    elif ok2:
        print("⚠️ GPT-4.1-mini OK, mas GPT-5.2 falhou.")
    else:
        print("❌ Nenhum modelo conectou.")
