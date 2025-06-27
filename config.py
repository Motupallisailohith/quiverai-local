# config.py

from dataclasses import dataclass
from enum import Enum
import os

class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    GROQ   = "groq"

@dataclass(frozen=True)
class ModelConfig:
    name: str
    temperature: float
    provider: ModelProvider

# ————— Model presets —————
QWEN_4B = ModelConfig(
    name="qwen3:4b",
    temperature=0.0,
    provider=ModelProvider.OLLAMA,
)
LLAMA_4 = ModelConfig(
    name="meta-llama/Llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    provider=ModelProvider.GROQ,
)

class Config:
    # — LLM Settings —
    MODEL: ModelConfig         = QWEN_4B
    CONTEXT_WINDOW: int        = 8192

    # — Document Ingestion & Embedding —
    DOCS_PATH: str             = os.getenv("DOCS_PATH", "docs/")
    INDEX_PATH: str            = os.getenv("INDEX_PATH", "index/")
    CHUNK_SIZE: int            = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int         = int(os.getenv("CHUNK_OVERLAP", 100))
    EMBED_MODEL: str           = os.getenv("CHROMA_EMBED_MODEL", "all-MiniLM-L6-v2")

    # — Retrieval & Cache —
    TOP_K: int                 = int(os.getenv("TOP_K", 5))
    CACHE_SIZE: int            = int(os.getenv("CACHE_SIZE", 1024))
    CACHE_BACKEND: str         = os.getenv("CACHE_BACKEND", "lru")  # or "redis"/"diskcache"

    # — UI & Server —
    STREAMLIT_HOST: str        = os.getenv("STREAMLIT_HOST", "0.0.0.0")
    STREAMLIT_PORT: int        = int(os.getenv("STREAMLIT_PORT", 8501))
    ALLOWED_FILE_TYPES: list    = ["pdf", "txt", "md"]

    # — Misc / Dev —
    SEED: int                  = 42
    LOG_LEVEL: str             = os.getenv("LOG_LEVEL", "INFO")