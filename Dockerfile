FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
# and skip reinstalling when only app code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + trained model
COPY app.py model.pkl ./

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]