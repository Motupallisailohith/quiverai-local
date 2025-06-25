# model.py

import os
from dotenv import load_dotenv

from config import Config, ModelProvider
from langchain.llms import Ollama
# from langchain_groq import Groq  # if you need GROQ support

# Load .env (so OLLAMA_HOST and OLLAMA_CONTEXT_LENGTH are seen)
load_dotenv()

def get_llm():
    """
    Factory to return a LangChain LLM instance according to Config.MODEL.
    Ollama now picks up host & context length from env vars:
      - OLLAMA_HOST      (e.g. http://127.0.0.1:11435)
      - OLLAMA_CONTEXT_LENGTH (e.g. 4096)
    """
    provider   = Config.MODEL.provider
    model_name = Config.MODEL.name
    temp       = Config.MODEL.temperature

    if provider == ModelProvider.OLLAMA:
        # Ollama will read OLLAMA_HOST and OLLAMA_CONTEXT_LENGTH from os.environ
        return Ollama(
            model=model_name,
            temperature=temp,
        )

    # elif provider == ModelProvider.GROQ:
    #     return Groq(model=model_name, temperature=temp)

    else:
        raise ValueError(f"Unsupported model provider: {provider}")
