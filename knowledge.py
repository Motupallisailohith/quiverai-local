# knowledge.py

import os
from dataclasses import dataclass
from enum import Enum
from typing import List
import faiss
from config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
# — Set up embeddings & storage paths —
EMBEDDER = SentenceTransformerEmbeddings(model_name=Config.EMBED_MODEL)
os.makedirs(Config.INDEX_PATH, exist_ok=True)

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
    return KnowledgeSource(id=source_id, name=os.path.basename(path),
                            type=KnowledgeType.DOCUMENT, content=content)

def load_from_url(url: str) -> KnowledgeSource:
    loader = WebBaseLoader(url)
    docs = loader.load()
    content = "\n\n".join(d.page_content for d in docs)
    return KnowledgeSource(id=f"url://{hash(url)}", name=url,
                            type=KnowledgeType.URL, content=content)

def build_vector_store(sources: List[KnowledgeSource]) -> None:
    """
    Wipe & rebuild the entire FAISS index from a list of sources.
    If there are no sources yet, do nothing (avoids empty-FAISS error).
    """
    # flatten documents & metadata
    texts, metas, ids = [], [], []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )

    for src in sources:
        for idx, chunk in enumerate(splitter.split_text(src.content)):
            texts.append(chunk)
            metas.append({"source": src.name, "type": src.type.value})
            ids.append(f"{src.id}::chunk{idx}")

    if not texts:
        # nothing to index
        return

    # build (or overwrite) the FAISS index on disk
    faiss_index_path = os.path.join(Config.INDEX_PATH, "faiss_index.faiss")
    store = FAISS.from_texts(
        texts,
        EMBEDDER,
        metadatas=metas,
        ids=ids,

    )
    store.save_local(folder_path=Config.INDEX_PATH)

def initialize_knowledge() -> None:
    """
    Scan DOCS_PATH and fully rebuild the index from every PDF/TXT/MD file.
    """
    sources: List[KnowledgeSource] = []
    for root, _, files in os.walk(Config.DOCS_PATH):
        for fname in files:
            if fname.lower().endswith((".pdf", ".txt", ".md")):
                sources.append(load_from_file(os.path.join(root, fname)))

    build_vector_store(sources)
def init_faiss() -> FAISS:
    """
    Load or initialize FAISS index:
     1) If index/index.faiss exists, load it
     2) Else run initialize_knowledge(), then try loading again
     3) If still no index (no docs at all), create an *empty* FAISS index
    """
    os.makedirs(Config.INDEX_PATH, exist_ok=True)
    index_path = os.path.join(Config.INDEX_PATH, "index.faiss")

    # 1) If already built: load it
    if os.path.exists(index_path):
        return FAISS.load_local(
            folder_path=Config.INDEX_PATH,
            embeddings=EMBEDDER,
            allow_dangerous_deserialization=True,
        )

    # 2) Rebuild from any existing docs
    initialize_knowledge()

    # 3) Try loading again
    if os.path.exists(index_path):
        return FAISS.load_local(
            folder_path=Config.INDEX_PATH,
            embeddings=EMBEDDER,
            allow_dangerous_deserialization=True,
        )

    # 4) No docs → return an empty FAISS store
    #    We need the embedding dim
    dummy = EMBEDDER.embed_query(" ")  
    dim = len(dummy)
    index = faiss.IndexFlatL2(dim)
    return FAISS(
        embedding_function=EMBEDDER,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

def add_source_to_index(source: KnowledgeSource) -> None:
    """
    Incrementally add one new source into the existing FAISS index.
    If the index isn’t there yet, delegates to build_vector_store.
    """
    # Load or create the on-disk store
   

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    texts = splitter.split_text(source.content)
    ids   = [f"{source.id}::chunk{i}" for i in range(len(texts))]
    metas = [{"source": source.name, "type": source.type.value} for _ in texts]
    index_file = os.path.join(Config.INDEX_PATH, "faiss_index.index")
    if os.path.exists(index_file):
        store = FAISS.load_local(
            folder_path=Config.INDEX_PATH,
            embeddings=EMBEDDER,
        )
    else:
        store = None
    if store is None:
        # no existing index; just rebuild from scratch
        build_vector_store([source])
    else:
        store.add_texts(texts, metadatas=metas, ids=ids)
        store.save_local(folder_path=Config.INDEX_PATH)

def delete_source_from_index(source_id: str) -> None:
    """
    Remove all chunks in the FAISS index whose IDs begin with source_id.
    (requires full rebuild, since FAISS doesn’t support per-vector deletion)
    """
    # Rebuild from all remaining sources
    # You’ll need to track which sources remain in-memory or on-disk;
    # for simplicity here we just re-index everything from the docs/ folder.
    initialize_knowledge()
