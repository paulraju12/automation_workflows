"""
Tests for the FastAPI application layer (app.py).

Uses httpx.AsyncClient against the ASGI app with mocked infrastructure
(Redis, PostgreSQL, LangGraph) so tests run without containers.
"""

import sys
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.conftest import SAMPLE_WORKFLOW

# ---------------------------------------------------------------------------
# We must patch heavy dependencies *before* importing app.py, because app.py
# performs module-level initialization (redis client, graph import, template
# loading, etc.).
# ---------------------------------------------------------------------------

# Patch the workflow template file load
_TEMPLATE_JSON = {
    "name": "WorkflowStructure",
    "description": "Template for generated workflows",
    "version": "1.0",
    "structure": {},
    "data": {},
}


def _mock_open_template(*args, **kwargs):
    """Return a mock file-like with the template JSON."""
    from io import StringIO
    return StringIO(json.dumps(_TEMPLATE_JSON))


@pytest.fixture(autouse=True)
def _patch_module_level():
    """
    Patch module-level side effects in app.py so the test module can import
    without needing live Redis, Postgres, Pinecone, or a real template file.
    """
    with (
        patch("builtins.open", side_effect=_mock_open_template),
        patch("redis.asyncio.Redis") as _mock_redis_cls,
    ):
        _mock_redis_cls.return_value = AsyncMock()
        yield


# Now safe to import
from httpx import ASGITransport, AsyncClient


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(mock_redis, mock_db_pool):
    """
    Provide an httpx AsyncClient wired to the FastAPI app, with Redis and
    Postgres fully mocked.
    """
    # Import app *inside* the fixture so patches above take effect
    import importlib
    import app as app_module
    importlib.reload(app_module)  # pick up the patched builtins.open

    the_app = app_module.app

    # Replace the global singletons
    app_module.redis_client = mock_redis
    app_module.db_pool = mock_db_pool

    transport = ASGITransport(app=the_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _graph_result(prompt: str = "hello") -> dict:
    """Simulate what ``graph.ainvoke`` returns."""
    return {
        "prompt": prompt,
        "history": [],
        "workflow": SAMPLE_WORKFLOW,
        "intent": "new_workflow",
        "response": "Let's build it! I've created a workflow based on your request.",
        "awaiting_input": True,
        "next_question": "Anything else to add?",
        "error": {},
    }


# ── Endpoint tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestWorkflowEndpoint:
    """Tests for POST /api/v1/workflow."""

    async def test_successful_workflow_creation(self, client, mock_redis, mock_db_pool):
        """A valid prompt returns 200 with a workflow and session_id."""
        with patch("app.graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=_graph_result("create a GitHub workflow"))

            resp = await client.post(
                "/api/v1/workflow",
                json={"prompt": "create a GitHub workflow"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"]
        assert body["conversation"]
        assert body["workflow"] is not None
        assert body["workflow"]["structure"]
        assert body["next_question"]

    async def test_returns_session_id_when_not_provided(self, client, mock_redis, mock_db_pool):
        """When no session_id is sent, the API generates one (UUID4)."""
        with patch("app.graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=_graph_result())

            resp = await client.post(
                "/api/v1/workflow",
                json={"prompt": "hello"},
            )

        body = resp.json()
        assert len(body["session_id"]) == 36  # UUID4 format

    async def test_uses_provided_session_id(self, client, mock_redis, mock_db_pool):
        """When session_id is sent, the API echoes it back."""
        with patch("app.graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=_graph_result())

            resp = await client.post(
                "/api/v1/workflow",
                json={"prompt": "hello", "session_id": "my-session-123"},
            )

        assert resp.json()["session_id"] == "my-session-123"

    async def test_cache_hit_returns_cached_response(self, client, mock_redis, mock_db_pool):
        """If Redis has a cached response for the prompt hash, it is returned directly."""
        cached_payload = {
            "conversation": "cached answer",
            "session_id": "sess-1",
            "workflow": None,
            "next_question": "cached question?",
            "interaction_id": 42,
        }
        mock_redis.get.return_value = json.dumps(cached_payload).encode()

        resp = await client.post(
            "/api/v1/workflow",
            json={"prompt": "cached prompt", "session_id": "sess-1"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["conversation"] == "cached answer"
        assert body["interaction_id"] == 42

    async def test_cache_miss_calls_graph(self, client, mock_redis, mock_db_pool):
        """On a cache miss, the LangGraph state machine is invoked."""
        mock_redis.get.return_value = None  # cache miss

        with patch("app.graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=_graph_result("new prompt"))

            resp = await client.post(
                "/api/v1/workflow",
                json={"prompt": "new prompt"},
            )

        assert resp.status_code == 200
        mock_graph.ainvoke.assert_awaited_once()

    async def test_missing_prompt_returns_422(self, client):
        """A request without a 'prompt' field should fail validation."""
        resp = await client.post("/api/v1/workflow", json={})
        assert resp.status_code == 422

    async def test_stores_interaction_in_db(self, client, mock_redis, mock_db_pool):
        """After processing, the interaction is persisted to PostgreSQL."""
        with patch("app.graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=_graph_result())

            await client.post(
                "/api/v1/workflow",
                json={"prompt": "store me"},
            )

        # The pool.acquire() context manager's connection should have fetchval called
        conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        conn.fetchval.assert_awaited()

    async def test_stores_state_in_redis(self, client, mock_redis, mock_db_pool):
        """After processing, the updated state is cached in Redis."""
        with patch("app.graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=_graph_result())

            await client.post(
                "/api/v1/workflow",
                json={"prompt": "cache my state"},
            )

        # setex should be called at least twice: once for state, once for response cache
        assert mock_redis.setex.await_count >= 2


@pytest.mark.asyncio
class TestCachingBehavior:
    """Focused tests on the Redis caching strategy."""

    async def test_cache_key_includes_session_and_prompt_hash(self, client, mock_redis, mock_db_pool):
        """
        The cache key is ``cache:{session_id}:{sha256(prompt)}``, ensuring
        different prompts within the same session don't collide.
        """
        import hashlib
        prompt = "unique prompt text"
        session_id = "sess-42"
        expected_hash = hashlib.sha256(prompt.encode()).hexdigest()
        expected_key = f"cache:{session_id}:{expected_hash}"

        with patch("app.graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=_graph_result(prompt))

            await client.post(
                "/api/v1/workflow",
                json={"prompt": prompt, "session_id": session_id},
            )

        # Check that redis.get was called with the expected cache key
        calls = [str(c) for c in mock_redis.get.call_args_list]
        assert any(expected_key in c for c in calls)

    async def test_cache_ttl_is_24_hours(self, client, mock_redis, mock_db_pool):
        """Both state and response caches use 86400s (24h) TTL."""
        with patch("app.graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=_graph_result())

            await client.post(
                "/api/v1/workflow",
                json={"prompt": "ttl check"},
            )

        for call in mock_redis.setex.call_args_list:
            args = call[0]
            # Second positional arg is the TTL
            assert args[1] == 86400
