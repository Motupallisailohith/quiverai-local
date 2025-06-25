# chatbot.py

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Iterator, List, Dict,Any

import os

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import Config
from knowledge import CHROMA_COLLECTION,KnowledgeSource  
from model import get_llm

# — Think markers & chunk types —

THINK_START = "<think>"
THINK_END   = "</think>"

class ChunkType(str, Enum):
    CONTENT     = "content"
    START_THINK = "start_think"
    THINKING    = "thinking"
    END_THINK   = "end_think"

@dataclass
class Chunk:
    type: ChunkType
    content: str

# — Prompt templates —

SYSTEM_PROMPT = """
You're QuiverAI, an assistant answering questions about the user's documents.
Use only the provided excerpts. If you don't know the answer, say so and ask for clarification.
""".strip()

PROMPT = """
Here's the information you have about the files:

<context>
{context}
</context>

Please respond to the query below:

<question>
{question}
</question>

Answer:
""".strip()

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", PROMPT),
])

# — Cache-Augmented Retrieval & Generation ——

@lru_cache(maxsize=Config.CACHE_SIZE)
def _cached_retrieve(question: str) -> List[str]:
    """
    Layer 1 of CAG: retrieve top-K chunks from the shared collection.
    """
    # Ensure index folder exists (Chroma was initialized on import)
    os.makedirs(Config.INDEX_PATH, exist_ok=True)

    # Query the single shared collection
    res = CHROMA_COLLECTION.query(
        query_texts=[question],
        n_results=Config.TOP_K
    )

    # Safely extract and format
    docs      = (res.get("documents")  or [[]])[0] or []
    metadatas = (res.get("metadatas") or [[]])[0] or []

    return [
        f"[{metadatas[i].get('source','unknown')}] {docs[i]}"
        for i in range(len(docs))
    ]

@lru_cache(maxsize=Config.CACHE_SIZE)
def _cached_answer(question: str, context: str) -> str:
    """
    Layer 2 of CAG: cache the final full answer keyed on (question, context).
    """
    llm = get_llm()
    prompt_val = PROMPT_TEMPLATE.format_prompt(
        context=context,
        question=question,
        chat_history=[]
    )
    prompt_str = prompt_val.to_string()
    resp = llm.generate([prompt_val.to_string()])
    return resp.generations[0][0].text

# — Chat history helpers —

def _create_chat_history(history: List[Dict]) -> List[BaseMessage]:
    msgs: List[BaseMessage] = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(m["content"]))
        else:
            msgs.append(AIMessage(m["content"]))
    return msgs

def _create_prompt(
    query: str,
    history: List[Dict[str, Any]],                # <-- also history could be Any
    sources: Dict[str, KnowledgeSource]           # <-- use KnowledgeSource here
):
    context = "\n\n".join(_cached_retrieve(query))
    chat_hist = _create_chat_history(history)
    return PROMPT_TEMPLATE.format_prompt(
        context=context,
        question=query,
        chat_history=chat_hist
    )

# — Streaming ask() ——

def ask(
    query: str,
    history: List[Dict],                # <-- use Any, not any
    sources: Dict[str, KnowledgeSource],          # <-- correct type
    llm: BaseChatModel | None = None
) -> Iterator[Chunk]:
    # 1) Final-answer cache
    context = "\n\n".join(_cached_retrieve(query))
    full_ans = _cached_answer(query, context)
    if full_ans:
        yield Chunk(type=ChunkType.CONTENT, content=full_ans)
        return

    # 2) Stream from LLM
    in_think = False
    client = llm or get_llm()
    prompt_val = _create_prompt(query, history, sources)
    messages = prompt_val.to_messages()
    prompt_str = prompt_val.to_string()
    for token in client.stream([prompt_str]):
        text = str(token.content or "")
        if text == THINK_START:
            in_think = True
            yield Chunk(type=ChunkType.START_THINK, content="")
            continue
        if text == THINK_END:
            in_think = False
            yield Chunk(type=ChunkType.END_THINK, content="")
            continue

        ctype = ChunkType.THINKING if in_think else ChunkType.CONTENT
        yield Chunk(type=ctype, content=text)

    # 3) Cache complete answer post-stream
    _cached_answer(query, context)
