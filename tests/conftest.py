import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def google_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set")
    return key


@pytest.fixture
def anthropic_api_key():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture
def azure_openai_keys():
    key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not key or not endpoint:
        pytest.skip("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")
    return {"key": key, "endpoint": endpoint}
