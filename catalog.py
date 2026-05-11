"""
catalog.py — Loads the SHL product catalog and builds a FAISS vector index
for semantic retrieval of assessments.
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog path resolution
# ---------------------------------------------------------------------------
_CATALOG_FILE = os.environ.get(
    "SHL_CATALOG_PATH",
    str(Path(__file__).parent / "shl_product_catalog.json"),
)

# ---------------------------------------------------------------------------
# Test-type abbreviation map (derived from sample conversations)
# ---------------------------------------------------------------------------
_KEY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Competencies": "C",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


def _abbreviate_keys(keys: list[str]) -> str:
    """Return a compact test-type code string like 'K' or 'P,C'."""
    codes = []
    for k in keys:
        code = _KEY_TO_CODE.get(k)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"


# ---------------------------------------------------------------------------
# Assessment data class
# ---------------------------------------------------------------------------
class Assessment:
    """Plain object holding one catalog row."""

    __slots__ = (
        "entity_id", "name", "link", "description", "keys", "keys_raw",
        "test_type_code", "job_levels", "languages", "duration",
        "remote", "adaptive",
    )

    def __init__(self, raw: dict):
        self.entity_id: str = raw.get("entity_id", "")
        self.name: str = raw.get("name", "")
        self.link: str = raw.get("link", "")
        self.description: str = raw.get("description", "")
        self.keys: list[str] = raw.get("keys", [])
        self.keys_raw: str = ", ".join(self.keys)
        self.test_type_code: str = _abbreviate_keys(self.keys)
        self.job_levels: list[str] = raw.get("job_levels", [])
        self.languages: list[str] = raw.get("languages", [])
        self.duration: str = raw.get("duration", "")
        self.remote: str = raw.get("remote", "")
        self.adaptive: str = raw.get("adaptive", "")

    # Text blob used for building the embedding
    def embedding_text(self) -> str:
        parts = [
            self.name,
            self.description,
            f"Categories: {self.keys_raw}",
            f"Job levels: {', '.join(self.job_levels)}",
            f"Duration: {self.duration}" if self.duration else "",
        ]
        return " | ".join(p for p in parts if p)

    # Compact string sent to the LLM as tool-call output
    def to_context_str(self) -> str:
        langs_str = ", ".join(self.languages[:3])
        if len(self.languages) > 3:
            langs_str += f" (+{len(self.languages) - 3} more)"
        return (
            f"Name: {self.name}\n"
            f"URL: {self.link}\n"
            f"Test Type: {self.test_type_code}\n"
            f"Keys: {self.keys_raw}\n"
            f"Description: {self.description}\n"
            f"Duration: {self.duration or '—'}\n"
            f"Job Levels: {', '.join(self.job_levels)}\n"
            f"Languages: {langs_str or '—'}\n"
            f"Remote: {self.remote} | Adaptive: {self.adaptive}"
        )

    def to_recommendation_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.link,
            "test_type": self.test_type_code,
        }


# ---------------------------------------------------------------------------
# Catalog singleton
# ---------------------------------------------------------------------------
class SHLCatalog:
    """Loads the catalog once and provides semantic search."""

    def __init__(self, catalog_path: str = _CATALOG_FILE):
        logger.info("Loading SHL catalog from %s …", catalog_path)
        raw_text = Path(catalog_path).read_text(encoding="utf-8")
        raw_data: list[dict] = json.loads(raw_text, strict=False)

        # Keep only items with status == "ok"
        self.assessments: list[Assessment] = [
            Assessment(r) for r in raw_data if r.get("status") == "ok"
        ]
        logger.info("Loaded %d assessments.", len(self.assessments))

        # Build name→assessment lookup for exact-match queries
        self._by_name: dict[str, Assessment] = {}
        for a in self.assessments:
            self._by_name[a.name.lower()] = a

        # Embedding model + FAISS index (built lazily on first search)
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._embeddings: Optional[np.ndarray] = None

    # ----- lazy init -----
    def _ensure_index(self):
        if self._index is not None:
            return

        # Try to load pre-built embeddings first (fast, low RAM)
        prebuilt_path = Path(__file__).parent / "embeddings.npy"
        if prebuilt_path.exists():
            logger.info("Loading pre-built embeddings from %s …", prebuilt_path)
            self._embeddings = np.load(str(prebuilt_path)).astype(np.float32)
            dim = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(self._embeddings)
            logger.info("FAISS index loaded — %d vectors, dim=%d.", len(self.assessments), dim)
        else:
            # Fallback: compute embeddings (requires sentence-transformers)
            logger.info("Building FAISS index (first call) …")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [a.embedding_text() for a in self.assessments]
            self._embeddings = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False,
            )
            dim = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(self._embeddings.astype(np.float32))
            logger.info("FAISS index built — %d vectors, dim=%d.", len(texts), dim)

    def _ensure_model(self):
        """Load the sentence-transformers model for query encoding."""
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    # ----- public API -----
    def search(self, query: str, top_k: int = 15) -> list[Assessment]:
        """Return the top_k most relevant assessments for a query string."""
        self._ensure_index()
        self._ensure_model()
        q_emb = self._model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self._index.search(q_emb, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.assessments):
                results.append(self.assessments[idx])
        return results

    def search_multi(self, queries: list[str], top_k: int = 15) -> list[Assessment]:
        """Run multiple queries and merge/deduplicate the results (union, ranked by best score)."""
        self._ensure_index()
        self._ensure_model()
        seen_ids: set[str] = set()
        merged: list[tuple[float, Assessment]] = []
        for query in queries:
            q_emb = self._model.encode([query], normalize_embeddings=True).astype(np.float32)
            scores, indices = self._index.search(q_emb, top_k)
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.assessments):
                    a = self.assessments[idx]
                    if a.entity_id not in seen_ids:
                        seen_ids.add(a.entity_id)
                        merged.append((float(score), a))
        merged.sort(key=lambda t: t[0], reverse=True)
        return [a for _, a in merged[:top_k]]

    def lookup_by_name(self, name: str) -> Optional[Assessment]:
        """Exact (case-insensitive) name lookup."""
        return self._by_name.get(name.strip().lower())

    def get_all_names(self) -> list[str]:
        """Return every assessment name for grounding checks."""
        return [a.name for a in self.assessments]


# Module-level singleton — imported by agent.py
catalog = SHLCatalog()

