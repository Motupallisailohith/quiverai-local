


# QuiverAI: Local RAG + Agent Platform with Ollama and Qwen3

QuiverAI is an **end-to-end retrieval-augmented generation (RAG)** and document QA chatbot that combines local document indexing (FAISS or Chroma) with an Ollama-hosted Qwen3 language model. The system is designed for privacy, speed, and easy extensibility — no calls to OpenAI or other cloud models required.

---

## ✨ Key Features

* **Local LLM**: runs *Qwen3:4b* on your machine using Ollama
* **FAISS / Chroma**: supports similarity search over ingested documents
* **Streamlit UI**: fast document upload + chat experience
* **Cache Layer**: persistent cache of previous queries and answers
* **Extensible**: designed to plug in more chains, agents, or tools

---

## 🚀 Quickstart (Local)

### 1️⃣ Prerequisites

* Python **3.10+**
* [Ollama](https://ollama.com) installed (`ollama serve` must work)
* Docker (optional, for a fully containerized experience)
* Sufficient GPU VRAM (Qwen3:4b runs best on a 6–8 GB+ card)

---

### 2️⃣ Install Ollama + Qwen3

```bash
ollama pull qwen3:4b
ollama serve --port 11434
```

* Validate:

  ```bash
  curl http://localhost:11434
  ```

  should return something like:

  ```json
  {"models":["qwen3:4b"]}
  ```

---

### 3️⃣ Project Setup

```bash
git clone https://github.com/YOUR_GITHUB/quiverai-local.git
cd quiverai-local
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4️⃣ Folder Layout

```
quiverai-local/
│
├── app.py                # Streamlit chat UI
├── ingest.py             # Ingest and index docs
├── llm.py                # Ollama LangChain wrapper
├── chatbot.py            # Conversational RAG logic
├── knowledge.py          # FAISS + document loaders
├── requirements.txt
├── Dockerfile
├── entrypoint.sh
│
├── docs/                 # place .pdf/.txt/.md here
└── index/                # generated FAISS/Chroma indexes
```

---

### 5️⃣ Ingest Documents

1. Place documents in `docs/`
2. Run:

   ```bash
   python ingest.py
   ```

   This splits, embeds, and indexes them into `index/`.

---

### 6️⃣ Launch the UI

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

* Visit [http://localhost:8501](http://localhost:8501)
* Upload a PDF or paste a website link
* Start chatting!

---

## 🐳 Docker Deployment

QuiverAI supports Docker deployment to Hugging Face Spaces or your own VPS. A **two-stage Dockerfile** is recommended:

1. **Stage 1** pulls the Ollama model in advance
2. **Stage 2** runs your Streamlit UI with Ollama side-by-side

**entrypoint.sh** manages ordering (Ollama first, Streamlit second) for consistent start-up.

✅ Full instructions are inside `Docker Deployment` doc or the `Dockerfile` itself.

---

## 🛠 System Design Highlights

* LangChain orchestration with `ChatOllama`
* Ollama as a local inference backend
* Streamlit for front-end
* FAISS for fast semantic search
* Caching layer for consistent answers
* System designed with separation of concerns: ingestion, indexing, retrieval, generation

---

## 💡 How Recruiters See It

If you describe this in a resume or interview, you could say:

* **“Designed a local RAG platform combining Ollama and Qwen3 with FAISS-based retrieval, Streamlit-based chat UI, and modular caching for near real-time document QA.”**

---

## 🧩 Next Steps

✅ Add multiple LLM selection (mix of local + cloud)
✅ Hybrid RAG with keyword + embedding retrieval
✅ Agent workflows (e.g., ReAct, Toolformer)
✅ User auth + secure document vault
✅ Better error monitoring

---

## 🤝 Contributing

Pull requests welcome! If you want to extend to ReAct agents, plugins, or new retrieval strategies, open an issue or PR.

---

## 📄 License

MIT License
(c) 2025 \[YourName]

---

If you want, I can **tailor** this even more (for Hugging Face Spaces, for GCP, etc.). Just say **“Refine for \[target]”** and I’ll do it! 🚀
