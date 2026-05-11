"""
main.py — FastAPI application for the SHL Assessment Recommender Agent.

Endpoints
---------
GET  /health  → {"status": "ok"}
POST /chat    → Agent reply with recommendations
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from agent import run_agent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL Individual Test Solutions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mount Frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    """Redirect root to the chat UI."""
    return RedirectResponse(url="/static/index.html")



# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message text")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ..., description="Full conversation history, oldest first."
    )


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Readiness probe."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Stateless chat endpoint. Accepts full conversation history and returns
    the agent's next reply plus any recommendations.
    """
    # Convert Pydantic models to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if not messages:
        raise HTTPException(status_code=400, detail="messages list must not be empty")

    try:
        result = run_agent(messages)
    except Exception:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail="Internal agent error")

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(**r) for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
