# syntax=docker/dockerfile:1
FROM python:3.11-slim

# set working directory
WORKDIR /app

# install system deps, then python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# now copy the rest of your source
COPY . .

# expose & run
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
