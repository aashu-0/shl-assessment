"""Pydantic request/response models for the ``/chat`` endpoint.

The response schema is dictated by the assignment and is *non-negotiable* — the
automated evaluator rejects any deviation. Keep this module faithful to the
spec: ``reply`` (str), ``recommendations`` (0..10 items, each name/url/test_type)
and ``end_of_conversation`` (bool).
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """A single turn in the stateless conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str

    @field_validator("content")
    @classmethod
    def _content_is_str(cls, v: str) -> str:
        # Guard against null content sneaking through as the string "None".
        return v if isinstance(v, str) else str(v or "")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str = ""


class ChatResponse(BaseModel):
    reply: str
    # 0 items while clarifying/refusing; 1..10 once a shortlist is committed.
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
