# test_llm.py
from src.agent.config import llm
from langchain_core.messages import HumanMessage

def test_connection():
    try:
        print("Connecting to local Ollama Qwen model...")
        response = llm.invoke([HumanMessage(content="Hello! If you can read this, your connection is successful.")])
        print("\nSuccess! LLM Response:")
        print(response.content)
    except Exception as e:
        print(f"\nConnection Failed: {e}")

if __name__ == "__main__":
    test_connection()
