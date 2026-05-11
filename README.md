# SHL Assessment Recommender — Conversational Agent

A production-ready, stateless conversational agent that recommends SHL Individual Test Solutions based on a hiring manager's requirements. Built as a FastAPI service with a real-time chat UI.

> **Live Demo:** Run the server and open [http://localhost:8000](http://localhost:8000)

---

## Features

- **Conversational AI** — Multi-turn dialogue with clarification, refinement, and comparison support.
- **Catalog-Grounded** — Every recommendation comes from the 377-item SHL product catalog via semantic search (zero hallucination).
- **Tool-Calling Agent** — The LLM autonomously decides when to search the catalog using native function calling.
- **Beautiful Chat UI** — Glassmorphic dark-mode web interface served at `/`.
- **Strict Schema** — Every response follows the required `{reply, recommendations, end_of_conversation}` JSON contract.
- **Resilient** — Automatic retry with exponential backoff and multi-model fallback chain.

---

## Architecture

```
User ──▶ FastAPI (/chat) ──▶ Agent (Groq LLM) ──▶ search_catalog tool
                                     │                     │
                                     │              FAISS + MiniLM
                                     │                     │
                                     ◀──── catalog results ◀──
                                     │
                              JSON response
                                     │
User ◀── {reply, recommendations} ◀──┘
```

| Component | Technology |
|---|---|
| **API Framework** | FastAPI + Uvicorn |
| **LLM Backend** | Groq API (LLaMA 3.3 70B) with OpenAI-compatible SDK |
| **Retrieval** | FAISS (cosine similarity) + `all-MiniLM-L6-v2` sentence embeddings |
| **Frontend** | Single-page HTML/JS chat interface |
| **Containerization** | Docker |

### Design Decisions

- **RAG with Agentic Tool Use** — Instead of stuffing the entire catalog into the prompt, the LLM decides *what* and *when* to retrieve via a `search_catalog` function call. This keeps prompts focused and eliminates hallucination.
- **Stateless** — Every `/chat` request carries the full conversation history. The server stores nothing between calls.
- **Groq + LLaMA 3.3** — Sub-second inference latency with native tool-calling support, comfortably meeting the 30-second evaluator timeout.
- **Post-Processing Validation** — Every recommendation is verified against the catalog before being returned. Hallucinated names/URLs are silently dropped.

---

## Project Structure

```
SHL/
├── main.py                    # FastAPI app — endpoints, models, static mount
├── agent.py                   # Core agent — LLM orchestration, tool calling, retries
├── catalog.py                 # Catalog loader, FAISS indexing, semantic search
├── shl_product_catalog.json   # 377-item SHL product catalog
├── static/
│   └── index.html             # Chat UI frontend
├── test_agent.py              # Smoke tests (schema, behavioral, conversation replay)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container build
└── README.md                  # This file
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key (optional — a default key is embedded)

```bash
# Linux / macOS
export GROQ_API_KEY="gsk_your_key_here"

# Windows PowerShell
$env:GROQ_API_KEY = "gsk_your_key_here"
```

### 3. Run the server

```bash
python main.py
```

### 4. Open the chat UI

Navigate to **[http://localhost:8000](http://localhost:8000)** in your browser.

### 5. Or use the API directly

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need a Java assessment for a mid-level developer"}]}'
```

---

## API Reference

### `GET /health`

Readiness probe.

**Response:** `{"status": "ok"}`

### `POST /chat`

Stateless chat endpoint.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for senior leadership selection"},
    {"role": "assistant", "content": "...previous reply..."},
    {"role": "user", "content": "Focus on personality and competency tests"}
  ]
}
```

**Response Body:**
```json
{
  "reply": "Here are my recommendations for senior leadership...",
  "recommendations": [
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

### Test Type Codes

| Code | Category |
|------|----------|
| K | Knowledge & Skills |
| P | Personality & Behavior |
| A | Ability & Aptitude |
| C | Competencies |
| B | Biodata & Situational Judgment |
| S | Simulations |
| D | Development & 360 |
| E | Assessment Exercises |

---

## Running Tests

With the server running in one terminal:

```bash
python test_agent.py
```

This runs:
- Health check
- Vague query → agent asks clarifying questions
- Specific query → agent returns recommendations
- Refinement → agent updates shortlist
- Off-topic → agent refuses politely
- C1 conversation trace replay

---

## Docker

```bash
docker build -t shl-agent .
docker run -p 8000:8000 -e GROQ_API_KEY="gsk_your_key" shl-agent
```

---

## Approach & Methodology

### Retrieval Setup

The catalog JSON (377 assessments) is loaded at startup. Each assessment is embedded using `all-MiniLM-L6-v2` (384-dim) with a composite text: `name | description | categories | job levels | duration`. Embeddings are indexed in FAISS `IndexFlatIP` (inner-product on L2-normalized vectors = cosine similarity).

At query time, the LLM formulates a search query and the `search_catalog` tool returns the top-10 results with metadata. Results are returned in a compact format to stay within context limits.

### Prompt Design

The system prompt enforces four key behaviors:

1. **Clarify before recommending** — For vague inputs, the agent asks 1–2 targeted questions.
2. **Ground every recommendation** — Names and URLs must come from tool results, never from prior knowledge.
3. **Support refinement** — Updates the shortlist without restarting when the user refines.
4. **Stay in scope** — Off-topic questions and prompt injection are refused.

### Evaluation

Development used the public conversation traces (C1–C10), checking:

- **Schema compliance** — Required fields present; URLs start with `https://www.shl.com/`.
- **Behavioral probes** — Vague queries produce no recommendations; off-topic is refused; refinements update the shortlist.
- **Recall** — Final shortlists compared against expected assessment sets.

### What Didn't Work

- Embedding the entire catalog into the system prompt caused context overflow and confused the model. Switching to tool-based retrieval with FAISS fixed both issues.
- Native tool calling on some providers (NVIDIA NIM) returned empty responses. Groq's OpenAI-compatible API with LLaMA 3.3 resolved this.

### AI Tools Used

Groq API (LLaMA 3.3 70B) for the agent LLM; sentence-transformers for embedding; development assisted by an AI coding agent for scaffolding and testing.

---

## License

Built for the SHL AI Intern Assessment.
