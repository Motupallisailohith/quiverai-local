# model.py

import os
from dotenv import load_dotenv

from config import Config, ModelProvider

# Ollama integration now lives in langchain_community
from langchain_community.llms.ollama import Ollama 
# from langchain_groq import Groq  # if/when you want to enable GROQ

# Load OLLAMA_HOST, OLLAMA_CONTEXT_LENGTH, etc.
load_dotenv()

def get_llm():
    """
    Factory to return a LangChain LLM instance according to Config.MODEL.
    Ollama parameters:
      - model:        your Ollama model name (e.g. "qwen3:4b")
      - temperature:  Config.MODEL.temperature
      - base_url:     the Ollama server endpoint (http://host:port) - CORRECTED
      - num_ctx:      OLLAMA_CONTEXT_LENGTH (optional) - CORRECTED PARAMETER NAME
    """
    provider = Config.MODEL.provider
    model_name = Config.MODEL.name
    temperature = Config.MODEL.temperature

    if provider == ModelProvider.OLLAMA:
        ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11435")
        ctx_len = os.getenv("OLLAMA_CONTEXT_LENGTH")

        ollama_params = {
            "model": model_name,
            "temperature": temperature,
            "base_url": ollama_url, # Changed 'url' to 'base_url'
        }
        if ctx_len:
            # Pass context length as 'num_ctx' during initialization
            ollama_params["num_ctx"] = int(ctx_len)

        llm = Ollama(**ollama_params) # Unpack parameters for initialization
        return llm

    # elif provider == ModelProvider.GROQ:
    #     return Groq(model=model_name, temperature=temperature)

    else:
        raise ValueError(f"Unsupported model provider: {provider}")