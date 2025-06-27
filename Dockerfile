###############################
# Stage 1: Pre-pull Ollama model
###############################
FROM ollama/ollama:latest AS ollama_stage

# Expose Ollama’s API port inside this build container
ENV OLLAMA_HOST="0.0.0.0:11434"

# Start Ollama, wait for it to come up, pull Qwen3:4b, then shut it down
RUN ollama serve --host 0.0.0.0 --port 11434 & \
    pid=$! && \
    for i in $(seq 1 30); do \
      curl -s http://127.0.0.1:11434/api/tags >/dev/null && break || sleep 1; \
    done && \
    ollama pull qwen3:4b && \
    kill $pid

###############################
# Stage 2: Build your Streamlit app
###############################
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy Ollama binary + pre-pulled model store from stage 1
COPY --from=ollama_stage /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama_stage /root/.ollama /root/.ollama

# Make sure your code picks up Ollama on localhost
ENV OLLAMA_HOST="http://127.0.0.1:11434"

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire app
COPY . .

# Expose Ollama + Streamlit ports
EXPOSE 11434
EXPOSE 8501

# On Spaces, $PORT will be set—bind Streamlit there
CMD bash -lc "\
    ollama serve --host 0.0.0.0 --port 11434 & \
    streamlit run app.py \
      --server.address=0.0.0.0 \
      --server.port=\${PORT:-8501} \
      --server.headless=true \
"
