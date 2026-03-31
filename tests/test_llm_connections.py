"""
Integration tests for LLM provider connections.
Skipped automatically when API keys are not configured.

Run with: pytest tests/test_llm_connections.py -v
"""
import pytest


@pytest.mark.integration
class TestGemini:
    def test_gemini_responds(self, google_api_key):
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=google_api_key,
        )
        response = llm.invoke("Say 'Hello' if this works.")
        assert response.content


@pytest.mark.integration
class TestClaude:
    def test_claude_responds(self, anthropic_api_key):
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model="claude-4-6-sonnet-20260220",
            api_key=anthropic_api_key,
            max_tokens=10,
        )
        response = llm.invoke("Say the word 'Banana'.")
        assert response.content


@pytest.mark.integration
class TestAzureOpenAI:
    def test_gpt_responds(self, azure_openai_keys):
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            azure_deployment="gpt-5.2-chat",
            azure_endpoint=azure_openai_keys["endpoint"],
            api_key=azure_openai_keys["key"],
            api_version="2024-12-01-preview",
        )
        response = llm.invoke("Diga apenas: 'Conexão OK!'")
        assert response.content
