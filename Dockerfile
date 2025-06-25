# ── Dockerfile ──

# 1) Use an official Python runtime
FROM python:3.10-slim

# 2) Set a working directory
WORKDIR /app

# 3) Copy & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy your application code
COPY . .

# 5) Default Ollama API URL (override via ENV on your host)
ENV OLLAMA_API_URL="http://ollama-api.onrender.com"

# 6) Expose Streamlit’s port
EXPOSE 8501

# 7) Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
