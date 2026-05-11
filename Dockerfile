FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code, catalog, and frontend
COPY main.py agent.py catalog.py ./
COPY shl_product_catalog.json ./
COPY static/ ./static/

# Port (Render sets $PORT dynamically)
EXPOSE 8000

# Run — use shell form so $PORT is resolved at runtime
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
