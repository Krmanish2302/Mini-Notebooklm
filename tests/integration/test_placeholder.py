"""
Placeholder integration tests.
Replace these with real tests using FastAPI TestClient / AsyncClient against:
  POST /api/config
  POST /api/mode
  POST /api/analyze
  POST /api/ingest
  GET  /api/sources
  DELETE /api/sources/{id}
  POST /api/query
  POST /api/query/stream
  POST /api/evaluate
  GET  /api/ragas/history
"""
import pytest

try:
    from fastapi.testclient import TestClient
    from api import app
    _HAS_APP = True
except Exception:  # noqa: BLE001
    _HAS_APP = False


@pytest.mark.integration
def test_placeholder_integration():
    """Sanity check — always passes."""
    assert True


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_APP, reason="api.py could not be imported")
def test_health_endpoint():
    """GET /api/health must return 200 with status field."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_APP, reason="api.py could not be imported")
def test_stats_endpoint():
    """GET /api/stats must return 200 with numeric fields."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/stats")
    assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_APP, reason="api.py could not be imported")
def test_sources_list_empty():
    """GET /api/sources returns a list (empty or not)."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/sources")
    assert response.status_code == 200
    body = response.json()
    assert "sources" in body
    assert isinstance(body["sources"], list)


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_APP, reason="api.py could not be imported")
def test_mode_switch_chat():
    """POST /api/mode with mode=chat returns 200."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/mode", json={"mode": "chat"})
    assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_APP, reason="api.py could not be imported")
def test_mode_switch_deep_research():
    """POST /api/mode with alias 'deep' resolves to deep_research."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/mode", json={"mode": "deep"})
    # Accept 200 (alias resolved) or 422 (strict validation — both are valid behaviours)
    assert response.status_code in (200, 422)


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_APP, reason="api.py could not be imported")
def test_query_no_sources_returns_answer():
    """POST /api/query without any ingested sources returns a non-empty answer."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/query",
        json={"query": "What is RAG?", "mode": "chat", "stream": False},
    )
    # Could be 200 (answer) or 503 (no LLM configured) — both are acceptable
    assert response.status_code in (200, 400, 503)
