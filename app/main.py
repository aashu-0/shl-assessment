"""FastAPI application exposing the recommender.

Endpoints:
* ``GET /health`` — readiness probe, returns ``{"status": "ok"}``.
* ``POST /chat``  — stateless: takes the full conversation history, returns the
  next agent reply plus (when committed) a shortlist of recommendations.

The service holds no per-conversation state; everything needed comes in the
request body, exactly as the assignment requires.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .agent import get_agent
from .catalog import get_catalog
from .schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-recommender")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the catalog and warm the retriever (embeddings, if enabled) so the
    # first real /chat is fast. Cold-start hosting is allowed up to 2 min.
    catalog = get_catalog()
    logger.info("Loaded %d assessments", len(catalog))
    get_agent()  # builds retriever + warms embeddings
    logger.info("Agent ready")
    yield


app = FastAPI(
    title="Conversational SHL Assessment Recommender",
    version=__version__,
    description="Turns a vague hiring intent into a grounded SHL assessment shortlist.",
    lifespan=lifespan,
)

# Permissive CORS: the evaluator and any demo UI call this from elsewhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/")
def root() -> dict:
    return {
        "service": "Conversational SHL Assessment Recommender",
        "version": __version__,
        "endpoints": {"health": "GET /health", "chat": "POST /chat"},
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    agent = get_agent()
    return agent.respond(request.messages)
