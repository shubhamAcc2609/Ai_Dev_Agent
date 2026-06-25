import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()



def initialise_llm() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


llm = initialise_llm()