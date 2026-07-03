"""End-to-end HTTP tests via FastAPI's TestClient (no LLM key required).

With no GEMINI_API_KEY set the agent uses its deterministic fallback, which is
enough to assert the endpoints and response schema behave correctly.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_schema_shape():
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "I need an assessment"}]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"reply", "recommendations", "end_of_conversation"}
    assert isinstance(body["reply"], str) and body["reply"]
    assert isinstance(body["recommendations"], list)
    assert isinstance(body["end_of_conversation"], bool)


def test_vague_first_turn_does_not_recommend():
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "I need an assessment"}]})
    body = r.json()
    # Behavior probe: no shortlist on a vague turn-1 query.
    assert body["recommendations"] == []
    assert body["end_of_conversation"] is False


def test_recommendations_have_valid_urls_and_cap():
    msgs = [
        {"role": "user", "content": "Hiring a Java developer, mid level, works with SQL and Spring"},
    ]
    r = client.post("/chat", json={"messages": msgs})
    body = r.json()
    assert len(body["recommendations"]) <= 10
    for rec in body["recommendations"]:
        assert set(rec) >= {"name", "url", "test_type"}
        assert rec["url"].startswith("http")


def test_empty_messages_is_handled():
    r = client.post("/chat", json={"messages": []})
    assert r.status_code == 200
    assert isinstance(r.json()["reply"], str)
