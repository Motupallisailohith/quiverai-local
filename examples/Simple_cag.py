# examples/Simple_cag.py

from knowledge import initialize_knowledge
from model import get_llm
from chatbot import ask, ChunkType

def main():
    # 1) (Re)build the index
    print("🔍 Indexing documents…")
    initialize_knowledge()

    # 2) Get the LLM
    llm = get_llm()

    # 3) Sample question
    question = "What is the main topic of the documents?"

    # 4) Run through CAG
    print(f"\n❓ Question: {question}\n")
    history = []
    sources = {}  # not used by this example; retrieval pulls from Chroma directly

    buffer = ""
    for chunk in ask(question, history, sources, llm):
        if chunk.type == ChunkType.START_THINK:
            print("[Thinking…]")
        elif chunk.type == ChunkType.END_THINK:
            print("\n[End Thinking]\n")
        else:
            print(chunk.content, end="", flush=True)
            buffer += chunk.content

    print(f"\n\n🏁 Final Answer:\n{buffer}\n")

if __name__ == "__main__":
    main()
