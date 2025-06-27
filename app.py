import os
import streamlit as st  # type: ignore
from dotenv import load_dotenv

from config import Config
from knowledge import (
    initialize_knowledge,
    load_from_file,
    load_from_url,
    KnowledgeSource,
    add_source_to_index,
    delete_source_from_index,
    init_faiss,             # ← import our FAISS helper
)
from chatbot import ask, ChunkType
from model import get_llm

# — Ensure we have a FAISS index on startup —
# (builds from DOCS_PATH if empty, or loads existing)
faiss_store = init_faiss()

# — If the Chroma/FAISS index dir was empty, do a full rebuild —
if not os.listdir(Config.INDEX_PATH):
    initialize_knowledge()

# — Load environment variables —
load_dotenv()

# — Streamlit page config —
st.set_page_config(
    page_title="QuiverAI Local Chat",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📚 QuiverAI")
st.subheader("Your private, local RAG-powered assistant")

# — Initialize session state —
if "history" not in st.session_state:
    st.session_state.history = []
if "sources" not in st.session_state:
    st.session_state.sources = {}
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0

# Ensure docs & index folders exist
os.makedirs(Config.DOCS_PATH, exist_ok=True)
os.makedirs(Config.INDEX_PATH, exist_ok=True)

# — Sidebar: Knowledge Vault management —
with st.sidebar:
    st.header("📂 Knowledge Vault")

    if st.button("🔄 Re-index all documents"):
        initialize_knowledge()
        st.rerun()

    st.divider()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=Config.ALLOWED_FILE_TYPES,
        key=f"file_uploader_{st.session_state.upload_counter}",
    )
    if uploaded_file:
        save_path = os.path.join(Config.DOCS_PATH, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved `{uploaded_file.name}`")

        # Load & register the source
        source = load_from_file(save_path)
        st.session_state.sources[source.id] = source

        # Incremental index via helper (auto-creates collections)
        add_source_to_index(source)
        # Also update FAISS
        chunks = source.content.split("\n\n")
        ids    = [f"{source.id}::chunk{i}" for i in range(len(chunks))]
        metas  = [{"source": source.name, "type": source.type.value} for _ in chunks]
        faiss_store.add_texts(
    texts=chunks,
    metadatas=metas,
    ids=ids,
)
        faiss_store.save_local(folder_path=Config.INDEX_PATH)

        st.info("Indexed new document.")
        st.session_state.upload_counter += 1
        st.success("Indexed new document.")  # type: ignore[attr-defined]

    st.divider()

    # URL uploader
    url_input = st.text_input("Or paste a Web Page URL", placeholder="https://example.com")
    if st.button("Add Web Page"):
        if url_input.startswith(("http://", "https://")):
            source = load_from_url(url_input)
            st.session_state.sources[source.id] = source

            add_source_to_index(source)
            chunks = source.content.split("\n\n")
            ids    = [f"{source.id}::chunk{i}" for i in range(len(chunks))]
            metas  = [{"source": source.name, "type": source.type.value} for _ in chunks]
            faiss_store.add_texts(
    texts=chunks,
    metadatas=metas,
    ids=ids,
)
            
            faiss_store.save_local(folder_path=Config.INDEX_PATH)

            st.success(f"Indexed URL `{url_input}`")
            st.success("Indexed new document.")  # type: ignore[attr-defined]
        else:
            st.error("Please enter a valid URL (including http:// or https://)")

    st.divider()
    st.subheader("Loaded Sources")

    # Dynamic source list with delete buttons
    for source_id in list(st.session_state.sources.keys()):
        source = st.session_state.sources[source_id]
        display_name = source.name
        if len(display_name) > 35:
            display_name = display_name[:30] + "..."
        col1, col2 = st.columns([0.8, 0.15])
        col1.text(display_name)
        if col2.button("🗑️", key=f"delete_{source_id}"):
            delete_source_from_index(source_id)
            # remove from state
            del st.session_state.sources[source_id]
            # immediately rerun so we don't keep iterating
            st.rerun()


# — Cache the LLM client for performance —
@st.cache_resource
def load_llm():
    return get_llm()

llm = load_llm()

# — Chat interface —
query = st.chat_input("Ask your documents…")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    assistant_msg = st.chat_message("assistant")
    placeholder = assistant_msg.empty()
    buffer = ""

    for chunk in ask(query, st.session_state.history, st.session_state.sources, llm):
        if chunk.type == ChunkType.START_THINK:
            buffer += "🤔 "
        elif chunk.type == ChunkType.END_THINK:
            buffer += "\n"
        else:
            buffer += chunk.content
        placeholder.markdown(buffer)

    st.session_state.history.append({"role": "assistant", "content": buffer})

# — Show conversation history —
if st.session_state.history:
    st.divider()
    st.subheader("Conversation History")
    for msg in st.session_state.history:
        role = "You" if msg["role"] == "user" else "QuiverAI"
        st.write(f"**{role}:** {msg['content']}")