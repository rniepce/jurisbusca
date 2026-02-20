import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

key = os.getenv("ANTHROPIC_API_KEY")
if not key:
    print("NO_KEY: ANTHROPIC_API_KEY is not set.")
    exit(1)

try:
    llm = ChatAnthropic(model="claude-4-6-sonnet-20260220", api_key=key, max_tokens=10)
    res = llm.invoke("Say the word 'Banana'.")
    print(f"SUCCESS: {res.content}")
except Exception as e:
    print(f"ERROR: {str(e)}")
