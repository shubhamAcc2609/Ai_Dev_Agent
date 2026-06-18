import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

# Load variables from .env
load_dotenv()

def initialise_llm():
    """Initializes and returns the Mistral LLM."""
    if not os.getenv("MISTRAL_API_KEY"):
        raise EnvironmentError(
            "CRITICAL: MISTRAL_API_KEY not found in .env file."
        )
        
    return ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.0
    )

# Usage
llm = initialise_llm()