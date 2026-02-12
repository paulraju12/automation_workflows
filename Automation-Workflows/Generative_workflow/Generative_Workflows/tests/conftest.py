"""
Shared pytest fixtures for the workflow automation test suite.

Provides mock services, sample states, and reusable test infrastructure
so that individual test modules stay focused on behavior, not setup.
"""

import sys
import os
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so bare imports work (agents.*, etc.)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Sample data ──────────────────────────────────────────────────────────────

SAMPLE_WORKFLOW: Dict[str, Any] = {
    "structure": [
        {
            "id": "node-1",
            "name": "github-issue-created",
            "type": "normal",
            "content": {},
            "position": {"x": 58, "y": 261},
        },
        {
            "id": "node-2",
            "name": "jira-create-ticket",
            "type": "normal",
            "content": {},
            "position": {"x": 158, "y": 261},
        },
    ],
    "data": [
        {
            "id": "node-1",
            "name": "github-issue-created",
            "type": "SCM_ACTION",
            "version": "1.0",
            "properties": {"action": "issue_created"},
            "metadata": {
                "title": "GitHub Issue Created",
                "connector": {"name": "GitHub"},
            },
            "scm_id": "adf1f67b-e369-4701-af47-d9733ef27326",
        },
        {
            "id": "node-2",
            "name": "jira-create-ticket",
            "type": "EXTERNAL_SOURCE",
            "version": "1.0",
            "properties": {"action": "create_ticket"},
            "metadata": {
                "title": "Jira Create Ticket",
                "connector": {"name": "Jira"},
            },
            "ticketing_id": "ticket-jira-placeholder",
        },
    ],
}

EMPTY_WORKFLOW: Dict[str, Any] = {"structure": [], "data": []}

SAMPLE_PINECONE_RESULTS: List[Dict[str, Any]] = [
    {
        "metadata": {
            "name": "GitHub Enterprise",
            "type": "scm",
            "id": "adf1f67b-e369-4701-af47-d9733ef27326",
        }
    },
    {
        "metadata": {
            "name": "Bitbucket",
            "type": "scm",
            "id": "bcd2a78c-f480-5812-bg58-e0844fg38437",
        }
    },
]


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_workflow() -> Dict[str, Any]:
    """Return a realistic workflow dict with structure and data."""
    return json.loads(json.dumps(SAMPLE_WORKFLOW))  # deep-copy via JSON


@pytest.fixture
def empty_workflow() -> Dict[str, Any]:
    """Return a workflow dict with empty structure/data lists."""
    return json.loads(json.dumps(EMPTY_WORKFLOW))


@pytest.fixture
def base_agent_state(empty_workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal AgentState suitable for invoking WorkflowGraph nodes.
    Fields mirror the ``AgentState`` TypedDict in workflow_graph.py.
    """
    return {
        "prompt": "",
        "history": [],
        "workflow": empty_workflow,
        "intent": None,
        "response": "",
        "awaiting_input": False,
        "next_question": "",
        "error": {},
    }


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """
    A pre-configured mock for ``LLMService`` that returns deterministic
    responses depending on the ``structured`` flag.
    """
    service = MagicMock()

    def _invoke_side_effect(template: str, structured: bool = False, **kwargs: Any) -> str:
        if structured:
            return json.dumps(SAMPLE_WORKFLOW)
        # For classification prompts, default to "new_workflow"
        return "new_workflow"

    service.invoke.side_effect = _invoke_side_effect
    return service


@pytest.fixture
def mock_history_service() -> MagicMock:
    """A no-op HistoryService mock."""
    service = MagicMock()
    service.load.return_value = []
    service.save.return_value = None
    return service


@pytest.fixture
def mock_pinecone_results() -> List[Dict[str, Any]]:
    """Return sample Pinecone query results."""
    return json.loads(json.dumps(SAMPLE_PINECONE_RESULTS))


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Async mock for redis.asyncio.Redis."""
    r = AsyncMock()
    r.get.return_value = None  # cache miss by default
    r.setex.return_value = True
    r.close.return_value = None
    return r


@pytest.fixture
def mock_db_pool() -> AsyncMock:
    """Async mock for asyncpg connection pool."""
    pool = AsyncMock()
    conn = AsyncMock()
    conn.fetch.return_value = []
    conn.fetchrow.return_value = None
    conn.fetchval.return_value = 1  # interaction_id

    # Make the pool's acquire() work as an async context manager
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = cm

    return pool
