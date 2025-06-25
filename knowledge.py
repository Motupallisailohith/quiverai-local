# knowledge.py

import os
from dataclasses import dataclass
from enum import Enum
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

from config import Config

# Ensure the index folder exists
os.makedirs(Config.INDEX_PATH, exist_ok=True)

# — Module-level Chroma client & collection —  
CHROMA_CLIENT = chromadb.Client()
CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(
    name="quiverai",
    # Use pure-Python default embedder
    embedding_function=embedding_functions.DefaultEmbeddingFunction(),  # type: ignore[arg-type]
)


class KnowledgeType(str, Enum):
    DOCUMENT = "document"
    URL      = "url"


@dataclass
class KnowledgeSource:
    id: str
    name: str
    type: KnowledgeType
    content: str


def load_from_file(path: str) -> KnowledgeSource:
    """
    Load a local file (pdf, txt, md) by path into a KnowledgeSource.
    """
    ext = os.path.splitext(path)[1].lower()
    source_id = f"file://{os.path.basename(path)}"

    if ext in (".txt", ".md"):
        loader = TextLoader(path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    docs = loader.load()
    content = "\n\n".join(d.page_content for d in docs)

    return KnowledgeSource(
        id=source_id,
        name=os.path.basename(path),
        type=KnowledgeType.DOCUMENT,
        content=content,
    )


def load_from_url(url: str) -> KnowledgeSource:
    """
    Load a web page by URL into a KnowledgeSource.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    content = "\n\n".join(d.page_content for d in docs)

    return KnowledgeSource(
        id=f"url://{hash(url)}",
        name=url,
        type=KnowledgeType.URL,
        content=content,
    )


def _get_collection():
    """
    Return the shared Chroma client and collection.
    """
    return CHROMA_CLIENT, CHROMA_COLLECTION


def build_vector_store(sources: List[KnowledgeSource]) -> None:
    """
    Wipe & rebuild the entire index from a list of sources.
    """
    client, coll = _get_collection()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )

    ids, docs, metas = [], [], []
    for src in sources:
        chunks = splitter.split_text(src.content)
        for i, chunk in enumerate(chunks):
            ids.append(f"{src.id}::chunk{i}")
            docs.append(chunk)
            metas.append({"source": src.name, "type": src.type.value})

    coll.reset()  # type: ignore[attr-defined]
    coll.add(ids=ids, documents=docs, metadatas=metas)  # type: ignore[arg-type]
    


def initialize_knowledge() -> None:
    """
    Scan DOCS_PATH and fully rebuild the index from every PDF/TXT/MD file.
    """
    sources: List[KnowledgeSource] = []
    for root, _, files in os.walk(Config.DOCS_PATH):
        for fname in files:
            if fname.lower().endswith((".pdf", ".txt", ".md")):
                path = os.path.join(root, fname)
                sources.append(load_from_file(path))
    build_vector_store(sources)


def add_source_to_index(source: KnowledgeSource) -> None:
    """
    Incrementally add one new source into the existing index.
    """
    client, coll = _get_collection()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(source.content)
    ids    = [f"{source.id}::chunk{i}" for i in range(len(chunks))]
    metas  = [{"source": source.name, "type": source.type.value} for _ in chunks]

    coll.add(ids=ids, documents=chunks, metadatas=metas)  # type: ignore[arg-type]
    


def delete_source_from_index(source_id: str) -> None:
    """
    Remove all chunks in the index whose IDs begin with source_id.
    """
    client, coll = _get_collection()
    existing = coll.get().get("ids", []) or []
    to_delete = [cid for cid in existing if cid.startswith(source_id)]
    if to_delete:
        coll.delete(ids=to_delete)
        