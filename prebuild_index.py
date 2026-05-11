"""
prebuild_index.py — Pre-compute FAISS index + embeddings at Docker build time.
This avoids loading the 90MB sentence-transformers model at runtime,
cutting RAM usage from ~600MB to ~200MB.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CATALOG_PATH = Path(__file__).parent / "shl_product_catalog.json"
EMBEDDINGS_PATH = Path(__file__).parent / "embeddings.npy"

def main():
    print("Loading catalog...")
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"), strict=False)
    assessments = [r for r in raw if r.get("status") == "ok"]
    print(f"Found {len(assessments)} assessments.")

    # Build embedding text for each assessment
    texts = []
    for r in assessments:
        name = r.get("name", "")
        desc = r.get("description", "")
        keys = ", ".join(r.get("keys", []))
        levels = ", ".join(r.get("job_levels", []))
        duration = r.get("duration", "")
        parts = [name, desc, f"Categories: {keys}", f"Job levels: {levels}"]
        if duration:
            parts.append(f"Duration: {duration}")
        texts.append(" | ".join(p for p in parts if p))

    print("Encoding with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print(f"Saving embeddings ({embeddings.shape}) to {EMBEDDINGS_PATH}")
    np.save(str(EMBEDDINGS_PATH), embeddings.astype(np.float32))
    print("Done! Embeddings saved.")

if __name__ == "__main__":
    main()
