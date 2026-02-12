"""
Tests for service-layer classes:
  - LLMService           (services/llm_service.py)
  - HistoryService        (services/history_service.py)
  - ConnectorRegistry     (connectors/registry.py)
  - WorkflowEngine        (engine/workflow_engine.py)

All external calls (OpenAI, filesystem, connectors) are mocked.
"""

import sys
import os
import json
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.conftest import SAMPLE_WORKFLOW


# ═══════════════════════════════════════════════════════════════════════════
# LLMService
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMService:
    """Tests for the LLM invocation wrapper."""

    @pytest.fixture(autouse=True)
    def _patch_openai(self):
        """Patch ChatOpenAI so no real API calls are made."""
        with patch("services.llm_service.ChatOpenAI") as mock_cls:
            self.mock_llm_instance = MagicMock()
            mock_cls.return_value = self.mock_llm_instance
            yield

    def _make_service(self):
        from services.llm_service import LLMService
        return LLMService()

    # ── Plain-text invocation ────────────────────────────────────────────

    def test_invoke_returns_plain_text(self):
        """Non-structured invoke returns the LLM content as-is."""
        self.mock_llm_instance.invoke.return_value = MagicMock(content="  new_workflow  ")
        svc = self._make_service()

        result = svc.invoke("Classify: {prompt}", structured=False, prompt="test")

        assert result == "new_workflow"

    def test_invoke_strips_whitespace(self):
        """Leading/trailing whitespace is stripped from plain text."""
        self.mock_llm_instance.invoke.return_value = MagicMock(content="\n  hello \n")
        svc = self._make_service()

        result = svc.invoke("{prompt}", structured=False, prompt="x")
        assert result == "hello"

    # ── Structured (JSON) invocation ─────────────────────────────────────

    def test_invoke_structured_returns_json(self):
        """Structured invoke parses and re-serializes JSON."""
        payload = json.dumps(SAMPLE_WORKFLOW)
        self.mock_llm_instance.invoke.return_value = MagicMock(content=payload)
        svc = self._make_service()

        result = svc.invoke("{prompt}", structured=True, prompt="x")
        parsed = json.loads(result)

        assert "structure" in parsed
        assert "data" in parsed

    def test_invoke_structured_strips_markdown_fences(self):
        """If the LLM wraps JSON in ```json ... ```, the wrapper is removed."""
        payload = f"```json\n{json.dumps(SAMPLE_WORKFLOW)}\n```"
        self.mock_llm_instance.invoke.return_value = MagicMock(content=payload)
        svc = self._make_service()

        result = svc.invoke("{prompt}", structured=True, prompt="x")
        parsed = json.loads(result)

        assert parsed["structure"]

    def test_invoke_structured_raises_on_missing_keys(self):
        """If 'structure' or 'data' is missing, a ValueError propagates."""
        self.mock_llm_instance.invoke.return_value = MagicMock(
            content=json.dumps({"structure": [], "data": None})
        )
        svc = self._make_service()

        with pytest.raises(Exception):
            svc.invoke("{prompt}", structured=True, prompt="x")

    # ── Retry logic ──────────────────────────────────────────────────────

    def test_retries_on_failure(self):
        """Invoke retries up to max_retries and succeeds on later attempt."""
        self.mock_llm_instance.invoke.side_effect = [
            Exception("transient"),
            MagicMock(content="recovered"),
        ]
        svc = self._make_service()
        svc.max_retries = 3

        result = svc.invoke("{prompt}", structured=False, prompt="x")
        assert result == "recovered"

    def test_raises_after_exhausting_retries(self):
        """If all retries fail, the final exception propagates."""
        self.mock_llm_instance.invoke.side_effect = Exception("permanent")
        svc = self._make_service()
        svc.max_retries = 2

        with pytest.raises(Exception, match="LLM error"):
            svc.invoke("{prompt}", structured=False, prompt="x")


# ═══════════════════════════════════════════════════════════════════════════
# HistoryService
# ═══════════════════════════════════════════════════════════════════════════

class TestHistoryService:
    """Tests for file-based conversation history."""

    def _make_service(self, filepath: str):
        from services.history_service import HistoryService
        return HistoryService(filename=filepath)

    def test_load_returns_empty_when_file_missing(self, tmp_path):
        """If the history file does not exist, load returns []."""
        svc = self._make_service(str(tmp_path / "does_not_exist.json"))
        assert svc.load("user1") == []

    def test_save_and_load_roundtrip(self, tmp_path):
        """Data saved for a user can be loaded back."""
        filepath = str(tmp_path / "history.json")
        svc = self._make_service(filepath)

        history = [("hello", "Hi! How can I help?"), ("create workflow", "Here's your workflow")]
        svc.save("user1", history)
        loaded = svc.load("user1")

        assert loaded == history

    def test_load_returns_empty_for_unknown_user(self, tmp_path):
        """A known file but unknown user_id returns []."""
        filepath = str(tmp_path / "history.json")
        svc = self._make_service(filepath)
        svc.save("user1", [("a", "b")])

        assert svc.load("user_unknown") == []

    def test_load_handles_corrupt_json(self, tmp_path):
        """If the JSON file is corrupt, load returns [] gracefully."""
        filepath = str(tmp_path / "history.json")
        with open(filepath, "w") as f:
            f.write("{{{not valid json")

        svc = self._make_service(filepath)
        assert svc.load("user1") == []

    def test_load_handles_empty_file(self, tmp_path):
        """An empty file returns []."""
        filepath = str(tmp_path / "history.json")
        with open(filepath, "w") as f:
            f.write("")

        svc = self._make_service(filepath)
        assert svc.load("user1") == []


# ═══════════════════════════════════════════════════════════════════════════
# ConnectorRegistry
# ═══════════════════════════════════════════════════════════════════════════

class TestConnectorRegistry:
    """Tests for the connector registry."""

    def _make_registry(self):
        from connectors.registry import ConnectorRegistry
        return ConnectorRegistry()

    def test_register_and_retrieve(self):
        """A registered connector is retrievable by ID."""
        registry = self._make_registry()
        connector = MagicMock(name="GitHub")
        registry.register("gh-123", connector)

        assert registry.get("gh-123") is connector

    def test_get_returns_none_for_unknown(self):
        """Requesting a non-existent connector returns None."""
        registry = self._make_registry()
        assert registry.get("nonexistent") is None

    def test_register_overwrites_existing(self):
        """Re-registering the same ID replaces the previous connector."""
        registry = self._make_registry()
        old = MagicMock(name="OldConnector")
        new = MagicMock(name="NewConnector")
        registry.register("id-1", old)
        registry.register("id-1", new)

        assert registry.get("id-1") is new

    def test_multiple_connectors(self):
        """Multiple connectors can coexist."""
        registry = self._make_registry()
        gh = MagicMock(name="GitHub")
        bb = MagicMock(name="Bitbucket")
        registry.register("gh", gh)
        registry.register("bb", bb)

        assert registry.get("gh") is gh
        assert registry.get("bb") is bb


# ═══════════════════════════════════════════════════════════════════════════
# WorkflowEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestWorkflowEngine:
    """Tests for the workflow execution engine."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self):
        """Patch ConnectorRegistry and SCMExecutor at import time."""
        with (
            patch("engine.workflow_engine.ConnectorRegistry") as mock_reg_cls,
            patch("engine.workflow_engine.SCMExecutor") as mock_exec_cls,
        ):
            self.mock_registry = MagicMock()
            mock_reg_cls.return_value = self.mock_registry
            self.mock_scm_executor = MagicMock()
            mock_exec_cls.return_value = self.mock_scm_executor
            yield

    def _make_engine(self):
        from engine.workflow_engine import WorkflowEngine
        return WorkflowEngine()

    def test_execute_processes_all_nodes(self, sample_workflow):
        """A valid workflow executes all nodes and returns 'completed'."""
        self.mock_scm_executor.execute.return_value = {"status": "success", "result": "done"}
        engine = self._make_engine()

        result = engine.execute(sample_workflow)

        assert result["status"] == "completed"
        assert len(result["steps"]) == len(sample_workflow["structure"])

    def test_execute_handles_missing_data_entry(self):
        """If a node has no matching data entry, execution fails fast."""
        engine = self._make_engine()
        workflow = {
            "structure": [
                {"id": "n1", "name": "orphan-node", "type": "normal", "content": {}, "position": {"x": 0, "y": 0}},
            ],
            "data": [],  # no matching data
        }

        result = engine.execute(workflow)

        assert result["status"] == "failed"
        assert result["steps"][0]["reason"] == "No matching data"

    def test_execute_returns_failed_on_exception(self):
        """An unhandled error during execution returns status='failed'."""
        engine = self._make_engine()
        # Pass a malformed workflow that will cause a KeyError
        result = engine.execute({"structure": [{}], "data": []})

        assert result["status"] == "failed"
        assert "error" in result

    def test_register_executor(self):
        """Custom executors can be registered for new action types."""
        engine = self._make_engine()
        custom_executor = MagicMock()
        engine.register_executor("CUSTOM_ACTION", custom_executor)

        assert engine.executors["CUSTOM_ACTION"] is custom_executor

    def test_scm_executor_called_for_scm_action(self, sample_workflow):
        """Nodes with type SCM_ACTION are dispatched to the SCM executor."""
        self.mock_scm_executor.execute.return_value = {"status": "success", "result": "SCM done"}
        engine = self._make_engine()

        # Keep only the SCM_ACTION node
        scm_only_workflow = {
            "structure": [sample_workflow["structure"][0]],
            "data": [sample_workflow["data"][0]],
        }
        result = engine.execute(scm_only_workflow)

        assert result["status"] == "completed"

    def test_external_source_returns_triggered(self, sample_workflow):
        """Nodes with type EXTERNAL_SOURCE return status='triggered'."""
        engine = self._make_engine()

        # Keep only the EXTERNAL_SOURCE node
        ext_only_workflow = {
            "structure": [sample_workflow["structure"][1]],
            "data": [sample_workflow["data"][1]],
        }
        result = engine.execute(ext_only_workflow)

        assert result["status"] == "completed"
        assert result["steps"][0]["status"] == "triggered"
