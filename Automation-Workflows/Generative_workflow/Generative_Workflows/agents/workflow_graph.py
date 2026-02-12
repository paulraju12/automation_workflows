import traceback
from langgraph.graph import StateGraph, END
from services.llm_service import LLMService
from services.history_service import HistoryService
from agents.tools import query_components_tool
from utils.logger import logger
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict
import json
import time
import os
from functools import wraps
import re


class AgentState(TypedDict):
    """State schema for the workflow graph."""
    prompt: str
    history: Annotated[List[Tuple[str, str]], "User-agent interaction history"]
    workflow: Dict[str, Any]
    intent: Literal["new_workflow", "modify_workflow", "unclear", "general"]
    response: str
    awaiting_input: bool
    next_question: str
    error: Annotated[Dict[str, Any], "Error details if any"]


def enterprise_error_handler(func: Callable[..., AgentState]) -> Callable[..., AgentState]:
    """Decorator for enterprise-grade error handling and logging."""
    @wraps(func)
    def wrapper(self: "WorkflowGraph", state: AgentState) -> AgentState:
        try:
            return func(self, state)
        except Exception as e:
            error_details: Dict[str, str] = {
                "message": str(e),
                "traceback": traceback.format_exc(),
                "function": func.__name__,
            }
            logger.error(f"Error in {func.__name__}: {error_details}", extra=error_details)
            state["response"] = "An error occurred. Please try again or clarify."
            state["intent"] = "unclear"
            state["awaiting_input"] = True
            state["error"] = error_details
            return state
    return wrapper


class WorkflowGraph:
    """Manages the workflow state graph for multi-agent interactions."""

    llm_service: LLMService
    history_service: HistoryService
    max_retries: int
    timeout_seconds: int
    graph: Any  # Compiled LangGraph

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        history_service: Optional[HistoryService] = None,
    ) -> None:
        self.llm_service: LLMService = llm_service or LLMService()
        self.history_service: HistoryService = history_service or HistoryService()
        self.max_retries: int = int(os.getenv("MAX_WORKFLOW_RETRIES", 3))
        self.timeout_seconds: int = int(os.getenv("WORKFLOW_TIMEOUT_SECONDS", 30))
        self.graph = self._build_graph()
        logger.debug("Initialized WorkflowGraph")

    def _build_graph(self) -> StateGraph:
        """Build the state graph for workflow processing."""
        workflow = StateGraph(AgentState)
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("generate_workflow", self.generate_workflow)
        workflow.add_node("modify_workflow", self.modify_workflow)
        workflow.add_node("handle_unclear", self.handle_unclear)
        workflow.add_node("handle_general", self.handle_general)

        workflow.set_entry_point("classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {
                "new_workflow": "generate_workflow",
                "modify_workflow": "modify_workflow",
                "unclear": "handle_unclear",
                "general": "handle_general",
            }
        )
        workflow.add_edge("generate_workflow", END)
        workflow.add_edge("modify_workflow", END)
        workflow.add_edge("handle_unclear", END)
        workflow.add_edge("handle_general", END)
        return workflow.compile()

    def _route_intent(self, state: AgentState) -> str:
        """Route the workflow based on classified intent."""
        intent = state["intent"] or "unclear"
        logger.debug(f"Routing intent: {intent}")
        return intent


    @enterprise_error_handler
    def classify_intent(self, state: AgentState) -> AgentState:
        """Classify the user's intent based on the prompt."""
        logger.info(f"Classifying intent for prompt: {state['prompt']}")
        if not state["prompt"] or not isinstance(state["prompt"], str):
            state["intent"] = "unclear"
            state["response"] = "I’m not sure what you mean. Can you clarify?"
            state["next_question"] = "Can you clarify?"
            return state

        history_str = "\n".join(f"{r}: {t}" for r, t in state["history"])
        workflow_str = json.dumps(state["workflow"], indent=2) if state["workflow"].get("structure") else "No workflow"
        enriched_query = f"Prompt: {state['prompt']}\nHistory: {history_str}\nWorkflow: {workflow_str}"
        pinecone_context = self._get_pinecone_context(enriched_query)

        template = (
            "Classify the user’s intent based on the prompt, history, and context:\n"
            "Options:\n"
            "- 'new_workflow': User wants to create a specific new workflow (e.g., 'create a workflow for Jira').\n"
            "- 'modify_workflow': User wants to modify an existing workflow (e.g., 'add to workflow').\n"
            "- 'general': User asks a question, seeks info, or initiates a workflow process without specifics (e.g., 'start new workflow', 'what are providers').\n"
            "- 'unclear': Intent is ambiguous.\n"
            "Prompt: {prompt}\nHistory: {history}\nContext: {pinecone_context}\n"
            "Examples:\n"
            "- 'create a workflow for Jira with GitHub' → 'new_workflow'\n"
            "- 'add a step to the workflow' → 'modify_workflow'\n"
            "- 'forgot above all conversation let start new workflow' → 'general'\n"
            "- 'start new workflow' → 'general'\n"
            "- 'what is this?' → 'general'\n"
            "Rules:\n"
            "- Focus on the user’s action: 'create' implies 'new_workflow', 'start' without 'create' implies 'general'.\n"
            "- If the prompt is ambiguous but mentions 'workflow', lean toward 'general' unless it’s clearly a creation or modification request.\n"
            "- Return exactly one of: 'new_workflow', 'modify_workflow', 'general', 'unclear'.\n"
            "Return plain text."
        )
        intent = self.llm_service.invoke(template, structured=False, prompt=state["prompt"], history=history_str, pinecone_context=pinecone_context)
        intent = intent.strip().strip("'\"")
        state["intent"] = intent if intent in {"new_workflow", "modify_workflow", "general", "unclear"} else "unclear"
        logger.info(f"Classified intent: {state['intent']}")
        return state

    @enterprise_error_handler
    def generate_workflow(self, state: AgentState) -> AgentState:
        """Generate a new workflow based on the prompt."""
        logger.info("Generating workflow")
        history_str = "\n".join(f"{r}: {t}" for r, t in state["history"])
        pinecone_context = self._get_pinecone_context(state["prompt"], history_str)


        template = (
            "Generate a workflow JSON based on the user’s prompt:\n"
            "Prompt: {prompt}\nHistory: {history}\nContext (SCM providers and connectors): {pinecone_context}\n"
            "Rules:\n"
            "- Identify SCM providers (e.g., GitHub, Bitbucket) and ticketing systems (e.g., Jira) in the prompt.\n"
            "- Use 'scm_id' from context for SCM providers (e.g., 'adf1f67b-e369-4701-af47-d9733ef27326' for GitLab).\n"
            "- Use 'ticketing_id' for ticketing systems (e.g., 'ticket-jira-placeholder' if no ID).\n"
            "- For conditional logic (e.g., 'based on type'), include a decision node with branches."
            "- 'structure': List of nodes with id (e.g., 'node-1'), name (hyphenated lowercase), type ('normal'), content (empty dict), position (x:58, y:261, increment x by 100).\n"
            "- 'data': List of entries with id, name, type ('SCM_ACTION' or 'EXTERNAL_SOURCE'), version '1.0', properties (dict with action), metadata (title, connector), and 'scm_id' or 'ticketing_id'.\n"
            "Return a valid JSON object with 'structure' and 'data' keys, without markdown code blocks."
        )
        try:
            response = self._invoke_with_retry(template, structured=True, prompt=state["prompt"], history=history_str, pinecone_context=pinecone_context)
            workflow = json.loads(response)
            if not workflow or not workflow.get("structure") or not workflow.get("data"):
                state["response"] = "I need more details to create a workflow. What specific actions or conditions do you want?"
                state["next_question"] = "What specific actions or conditions do you want?"
                state["workflow"] = {}
            else:
                state["workflow"] = workflow
                state["response"] = "Let’s build it! I’ve created a workflow based on your request."                        #state["response"] = f"Let’s build it! Here’s the workflow: {json.dumps(workflow, indent=2)}"
                state["next_question"] = "Anything else to add?"
        except Exception as e:
            logger.error(f"Workflow generation failed: {e}")
            state["response"] = "I couldn’t generate a workflow yet. What specific actions or conditions do you want?"
            state["next_question"] = "What specific actions or conditions do you want?"
            state["workflow"] = {}
        state["awaiting_input"] = True
        return state

    @enterprise_error_handler
    def modify_workflow(self, state: AgentState) -> AgentState:
        """Modify an existing workflow based on the prompt."""
        logger.info("Modifying workflow")
        history_str = "\n".join(f"{r}: {t}" for r, t in state["history"])
        pinecone_context = self._get_pinecone_context(state["prompt"], history_str)

        template = (
            "Modify the existing workflow JSON based on the prompt:\n"
            "Prompt: {prompt}\nHistory: {history}\nContext: {pinecone_context}\nExisting: {existing_workflow}\n"
            "Rules:\n"
            "- Update the JSON object with 'structure' and 'data' keys based on the prompt.\n"
            "- If no existing workflow, start fresh but consider the prompt.\n"
            "Return a valid JSON object with 'structure' and 'data' keys."
        )
        try:
            response = self._invoke_with_retry(template, structured=True, prompt=state["prompt"], history=history_str, pinecone_context=pinecone_context, existing_workflow=json.dumps(state["workflow"]))
            workflow = json.loads(response)
            state["workflow"] = workflow
            state["response"] = "Got it. I’ve updated the workflow for you."   #state["response"] = f"Got it. Workflow updated: {json.dumps(workflow, indent=2)}"
            state["next_question"] = "Anything else to add?"
        except Exception as e:
            logger.error(f"Workflow modification failed: {e}")
            state["response"] = "I couldn’t update the workflow. What do you want to change?"
            state["next_question"] = "What do you want to change?"
        state["awaiting_input"] = True
        return state

    @enterprise_error_handler
    def handle_unclear(self, state: AgentState) -> AgentState:
        """Handle cases where intent is unclear."""
        logger.info("Handling unclear intent")
        state["response"] = "I’m not sure what you mean. Can you clarify?"
        state["next_question"] = "Can you clarify?"
        state["awaiting_input"] = True
        return state

    @enterprise_error_handler
    def handle_general(self, state: AgentState) -> AgentState:
        """Handle general queries or non-specific requests."""
        logger.info("Handling general query")
        history_str = "\n".join(f"{r}: {t}" for r, t in state["history"])
        pinecone_context = self._get_pinecone_context(state["prompt"], history_str)

        name_match = re.search(r"my name is (\w+)", state["prompt"], re.IGNORECASE)
        if name_match:
            name = name_match.group(1).capitalize()
            state["response"] = f"Hi {name}! How can I assist you today?"
            state["next_question"] = "How can I assist you today?"
        else:
            template = (
                "Respond to the user’s prompt dynamically:\n"
                "Prompt: {prompt}\nHistory: {history}\nContext: {pinecone_context}\n"
                "Rules:\n"
                "- If asking to start a workflow without specifics (e.g., 'start new workflow'), ask for requirements.\n"
                "- If asking about providers or info, provide relevant details from context.\n"
                "- Keep responses concise, friendly, and conversational.\n"
                "Return plain text."
            )
            state["response"] = self._invoke_with_retry(template, structured=False, prompt=state["prompt"], history=history_str, pinecone_context=pinecone_context)
            state["next_question"] = "What do you want to do next?" if "start new workflow" in state["prompt"].lower() else "How can I assist you further?"
        state["awaiting_input"] = True
        return state

    def _invoke_with_retry(
        self, template: str, structured: bool, retries: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Invoke LLM with retry logic and exponential backoff."""
        effective_retries: int = retries if retries is not None else self.max_retries
        for attempt in range(effective_retries):
            try:
                return self.llm_service.invoke(template, structured, **kwargs)
            except Exception as e:
                if attempt == effective_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1}/{effective_retries} failed: {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError("Retry loop exited unexpectedly")

    def _get_pinecone_context(self, query: str, history: str = "") -> str:
        """Retrieve context from Pinecone vector store."""
        try:
            results: List[Dict[str, Any]] = query_components_tool._run(prompt=query, top_k=5)
            context: str = "\n".join(
                f"Name: {r.get('metadata', {}).get('name', 'unknown')}, "
                f"Type: {r.get('metadata', {}).get('type', 'unknown')}, "
                f"ID: {r.get('metadata', {}).get('id', 'N/A')}"
                for r in results
            ) or "No context found"
            logger.info("Pinecone query successful, retrieved %d results", len(results))
            return context
        except Exception as e:
            logger.warning(f"Pinecone query failed: {e}")
            return "No context found"

graph = WorkflowGraph().graph