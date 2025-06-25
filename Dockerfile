# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy & install your Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire application
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Point Streamlit at 0.0.0.0 so it listens on all interfaces
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.enableCORS=false"]
