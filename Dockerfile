FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, catalog, and frontend
COPY main.py agent.py catalog.py prebuild_index.py ./
COPY shl_product_catalog.json ./
COPY static/ ./static/

# Pre-build embeddings at build time (avoids slow first-request on Render)
# This also downloads the sentence-transformers model into the image
RUN python prebuild_index.py

# Port (Render sets $PORT dynamically)
EXPOSE 8000

# Run — use shell form so $PORT is resolved at runtime
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
