"""
Tests for the LangGraph-based workflow state machine (agents/workflow_graph.py).

Covers:
  - Intent classification routing
  - Workflow generation (new + modify)
  - Fallback handlers (unclear, general)
  - Enterprise error-handling decorator
  - Retry / backoff logic

All LLM and Pinecone calls are fully mocked so these tests run offline
and deterministically.
"""

import sys
import os
import json
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.conftest import SAMPLE_WORKFLOW, SAMPLE_PINECONE_RESULTS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_graph(mock_llm, mock_history):
    """
    Import and instantiate WorkflowGraph with mocked Pinecone.

    We patch ``query_components_tool._run`` globally so the graph never
    hits a real Pinecone index.
    """
    with patch("agents.workflow_graph.query_components_tool") as mock_pc:
        mock_pc._run.return_value = SAMPLE_PINECONE_RESULTS
        from agents.workflow_graph import WorkflowGraph
        wg = WorkflowGraph(llm_service=mock_llm, history_service=mock_history)
    return wg, mock_pc


# ── Intent classification ────────────────────────────────────────────────────

class TestClassifyIntent:
    """Tests for the classify_intent node."""

    def test_classifies_new_workflow(self, base_agent_state, mock_llm_service, mock_history_service):
        """LLM returns 'new_workflow' -> state.intent == 'new_workflow'."""
        mock_llm_service.invoke.return_value = "new_workflow"
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "create a workflow for GitHub and Jira"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.classify_intent(state)

        assert result["intent"] == "new_workflow"

    def test_classifies_modify_workflow(self, base_agent_state, mock_llm_service, mock_history_service):
        """LLM returns 'modify_workflow' -> state.intent == 'modify_workflow'."""
        mock_llm_service.invoke.return_value = "modify_workflow"
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "add a step to the workflow"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.classify_intent(state)

        assert result["intent"] == "modify_workflow"

    def test_classifies_general(self, base_agent_state, mock_llm_service, mock_history_service):
        """LLM returns 'general' -> state.intent == 'general'."""
        mock_llm_service.invoke.return_value = "general"
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "what providers do you support?"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.classify_intent(state)

        assert result["intent"] == "general"

    def test_classifies_unclear(self, base_agent_state, mock_llm_service, mock_history_service):
        """LLM returns 'unclear' -> state.intent == 'unclear'."""
        mock_llm_service.invoke.return_value = "unclear"
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "hmm"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.classify_intent(state)

        assert result["intent"] == "unclear"

    def test_invalid_intent_falls_back_to_unclear(self, base_agent_state, mock_llm_service, mock_history_service):
        """If the LLM returns garbage, intent is normalized to 'unclear'."""
        mock_llm_service.invoke.return_value = "something_random"
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "xyzzy"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.classify_intent(state)

        assert result["intent"] == "unclear"

    def test_empty_prompt_returns_unclear(self, base_agent_state, mock_llm_service, mock_history_service):
        """Empty string prompt short-circuits to 'unclear' without calling the LLM."""
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": ""}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.classify_intent(state)

        assert result["intent"] == "unclear"
        mock_llm_service.invoke.assert_not_called()

    def test_none_prompt_returns_unclear(self, base_agent_state, mock_llm_service, mock_history_service):
        """None prompt short-circuits to 'unclear'."""
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": None}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.classify_intent(state)

        assert result["intent"] == "unclear"


# ── Workflow generation ──────────────────────────────────────────────────────

class TestGenerateWorkflow:
    """Tests for the generate_workflow node."""

    def test_generates_valid_workflow(self, base_agent_state, mock_llm_service, mock_history_service):
        """When the LLM returns valid JSON, state.workflow is populated."""
        mock_llm_service.invoke.return_value = json.dumps(SAMPLE_WORKFLOW)
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "create a workflow for GitHub and Jira"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.generate_workflow(state)

        assert result["workflow"]["structure"]
        assert result["workflow"]["data"]
        assert result["awaiting_input"] is True
        assert "workflow" in result["response"].lower() or "build" in result["response"].lower()

    def test_handles_invalid_json_response(self, base_agent_state, mock_llm_service, mock_history_service):
        """If the LLM response is not parseable JSON, the user gets a clarification prompt."""
        mock_llm_service.invoke.side_effect = Exception("LLM parsing error")
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "create a workflow"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.generate_workflow(state)

        assert result["workflow"] == {}
        assert result["awaiting_input"] is True

    def test_handles_empty_structure_data(self, base_agent_state, mock_llm_service, mock_history_service):
        """If the LLM returns valid JSON but with empty structure/data, prompt for details."""
        mock_llm_service.invoke.return_value = json.dumps({"structure": [], "data": []})
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "create a workflow"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.generate_workflow(state)

        # Empty structure/data means workflow should be cleared
        assert result["workflow"] == {}
        assert "more details" in result["response"].lower() or "specific" in result["response"].lower()


# ── Modify workflow ──────────────────────────────────────────────────────────

class TestModifyWorkflow:
    """Tests for the modify_workflow node."""

    def test_modifies_existing_workflow(self, base_agent_state, mock_llm_service, mock_history_service, sample_workflow):
        """Successfully updates the workflow when LLM returns valid JSON."""
        modified = {**SAMPLE_WORKFLOW}
        modified["structure"].append({
            "id": "node-3",
            "name": "slack-notification",
            "type": "normal",
            "content": {},
            "position": {"x": 258, "y": 261},
        })
        mock_llm_service.invoke.return_value = json.dumps(modified)
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "add slack notification", "workflow": sample_workflow}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.modify_workflow(state)

        assert len(result["workflow"]["structure"]) == 3
        assert result["awaiting_input"] is True
        assert "updated" in result["response"].lower() or "got it" in result["response"].lower()

    def test_modify_handles_llm_failure(self, base_agent_state, mock_llm_service, mock_history_service, sample_workflow):
        """When the LLM fails during modification, a helpful error message is returned."""
        mock_llm_service.invoke.side_effect = Exception("timeout")
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "change something", "workflow": sample_workflow}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.modify_workflow(state)

        assert "couldn't" in result["response"].lower() or "change" in result["response"].lower()
        assert result["awaiting_input"] is True


# ── Handle unclear ───────────────────────────────────────────────────────────

class TestHandleUnclear:
    """Tests for the handle_unclear node."""

    def test_unclear_returns_clarification(self, base_agent_state, mock_llm_service, mock_history_service):
        """The unclear handler asks the user to clarify."""
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "???"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.handle_unclear(state)

        assert "clarify" in result["response"].lower()
        assert result["awaiting_input"] is True


# ── Handle general ───────────────────────────────────────────────────────────

class TestHandleGeneral:
    """Tests for the handle_general node."""

    def test_general_responds_with_llm(self, base_agent_state, mock_llm_service, mock_history_service):
        """General queries produce a conversational LLM response."""
        mock_llm_service.invoke.return_value = "We support GitHub, Bitbucket, and GitLab."
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "what providers do you support?"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.handle_general(state)

        assert result["response"]
        assert result["awaiting_input"] is True

    def test_general_detects_user_name(self, base_agent_state, mock_llm_service, mock_history_service):
        """If the prompt contains 'my name is X', the response greets the user."""
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "my name is Alice"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.handle_general(state)

        assert "Alice" in result["response"]
        assert result["awaiting_input"] is True

    def test_general_start_new_workflow_asks_for_requirements(self, base_agent_state, mock_llm_service, mock_history_service):
        """'start new workflow' without specifics should ask what the user needs."""
        mock_llm_service.invoke.return_value = "Sure! What kind of workflow would you like to create?"
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "start new workflow"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.handle_general(state)

        assert result["awaiting_input"] is True
        assert "next" in result["next_question"].lower() or "do" in result["next_question"].lower()


# ── Error handling decorator ─────────────────────────────────────────────────

class TestEnterpriseErrorHandler:
    """Tests for the enterprise_error_handler decorator."""

    def test_decorator_catches_exceptions(self, base_agent_state, mock_llm_service, mock_history_service):
        """
        When a decorated method raises, the decorator should populate
        state['error'] and set intent to 'unclear'.
        """
        mock_llm_service.invoke.side_effect = RuntimeError("kaboom")
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "create a workflow for GitHub"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.classify_intent(state)

        assert result["intent"] == "unclear"
        assert result["error"]["message"] == "kaboom"
        assert result["awaiting_input"] is True

    def test_decorator_preserves_traceback(self, base_agent_state, mock_llm_service, mock_history_service):
        """The error dict should contain a traceback string."""
        mock_llm_service.invoke.side_effect = ValueError("bad value")
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        state = {**base_agent_state, "prompt": "do something"}
        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            result = wg.classify_intent(state)

        assert "traceback" in result["error"]
        assert "ValueError" in result["error"]["traceback"]


# ── Retry logic ──────────────────────────────────────────────────────────────

class TestRetryLogic:
    """Tests for _invoke_with_retry."""

    def test_retry_succeeds_on_second_attempt(self, mock_llm_service, mock_history_service):
        """If the first call fails but the second succeeds, we get the result."""
        call_count = 0

        def _flaky_invoke(template, structured=False, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("transient")
            return "success"

        mock_llm_service.invoke.side_effect = _flaky_invoke
        wg, _ = _build_graph(mock_llm_service, mock_history_service)

        result = wg._invoke_with_retry("template", structured=False, prompt="test")
        assert result == "success"
        assert call_count == 2

    def test_retry_exhausts_attempts(self, mock_llm_service, mock_history_service):
        """If every attempt fails, the exception propagates."""
        mock_llm_service.invoke.side_effect = ConnectionError("permanent failure")
        wg, _ = _build_graph(mock_llm_service, mock_history_service)

        with pytest.raises(ConnectionError, match="permanent failure"):
            wg._invoke_with_retry("template", structured=False, retries=2, prompt="test")


# ── Pinecone context retrieval ───────────────────────────────────────────────

class TestPineconeContext:
    """Tests for _get_pinecone_context."""

    def test_returns_context_string(self, mock_llm_service, mock_history_service):
        """Successful Pinecone query produces a formatted context string."""
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)

        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            ctx = wg._get_pinecone_context("github jira")

        assert "GitHub Enterprise" in ctx
        assert "Bitbucket" in ctx

    def test_returns_fallback_on_pinecone_failure(self, mock_llm_service, mock_history_service):
        """When Pinecone raises, the fallback 'No context found' is returned."""
        wg, mock_pc = _build_graph(mock_llm_service, mock_history_service)
        mock_pc._run.side_effect = Exception("Pinecone down")

        with patch("agents.workflow_graph.query_components_tool", mock_pc):
            ctx = wg._get_pinecone_context("anything")

        assert ctx == "No context found"


# ── Graph routing ────────────────────────────────────────────────────────────

class TestRouteIntent:
    """Tests for the conditional edge router."""

    @pytest.mark.parametrize(
        "intent,expected_route",
        [
            ("new_workflow", "new_workflow"),
            ("modify_workflow", "modify_workflow"),
            ("unclear", "unclear"),
            ("general", "general"),
        ],
    )
    def test_routes_correctly(self, intent, expected_route, base_agent_state, mock_llm_service, mock_history_service):
        wg, _ = _build_graph(mock_llm_service, mock_history_service)
        state = {**base_agent_state, "intent": intent}
        assert wg._route_intent(state) == expected_route

    def test_none_intent_defaults_to_unclear(self, base_agent_state, mock_llm_service, mock_history_service):
        wg, _ = _build_graph(mock_llm_service, mock_history_service)
        state = {**base_agent_state, "intent": None}
        assert wg._route_intent(state) == "unclear"
