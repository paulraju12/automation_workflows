from typing import List, Dict, Any
from functools import wraps
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from embeddings.vector_store import VectorStore
from utils.logger import logger
from utils.config import ActiveConfig
from connectors.registry import ConnectorRegistry
import traceback


class QueryComponentsInput(BaseModel):
    """Input schema for querying components."""
    prompt: str = Field(..., description="Search query for components")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of top results to retrieve")


class SCMActionInput(BaseModel):
    """Input schema for executing SCM actions."""
    action: Dict[str, Any] = Field(..., description="SCM action details")


def tool_error_handler(func):
    """
    Decorator for handling errors in tools with consistent logging and error response.

    Args:
        func (callable): Tool function to wrap

    Returns:
        callable: Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_details = {"error": str(e), "traceback": traceback.format_exc()}
            logger.error(f"Tool Error in {func.__name__}: {error_details}", extra=error_details)
            return {"status": "error", "message": str(e), "details": error_details}
    return wrapper


class QueryComponentsTool(BaseTool):
    """Tool for querying components from a vector store."""
    name: str = "query_components"
    description: str = "Query vector store for relevant connectors and SCM providers"
    args_schema: type[BaseModel] = QueryComponentsInput

    def __init__(self):
        super().__init__()
        logger.debug("Initialized QueryComponentsTool")

    @tool_error_handler
    def _run(self, prompt: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Execute a query against the vector store.

        Args:
            prompt (str): The search query string
            top_k (int, optional): Number of top results to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: List of matching components with metadata
        """
        logger.debug(f"Querying components: prompt='{prompt}', top_k={top_k}")
        vector_store = VectorStore()
        components = vector_store.query(prompt, top_k)
        for component in components:
            metadata = component.get("metadata", {})
            logger.info(f"Queried Component: Name={metadata.get('name', 'Unknown')}, "
                        f"Type={metadata.get('type', 'Unknown')}, ID={metadata.get('id', 'N/A')}")
        return components


class SCMActionTool(BaseTool):
    """Tool for executing SCM actions via registered connectors."""
    name: str = "execute_scm_action"
    description: str = "Execute Source Control Management actions via connectors"
    args_schema: type[BaseModel] = SCMActionInput
    connector_registry: ConnectorRegistry = Field(default_factory=ConnectorRegistry)

    def __init__(self):
        super().__init__()
        logger.debug("Initialized SCMActionTool")

    @tool_error_handler
    def _run(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an SCM action using the specified connector.

        Args:
            action (Dict[str, Any]): SCM action details including metadata and properties

        Returns:
            Dict[str, Any]: Result of the action execution
        """
        logger.debug(f"Executing SCM action: {action}")
        if not action or "metadata" not in action or "connector" not in action["metadata"]:
            raise ValueError("Invalid SCM action: Missing metadata or connector")

        connector_metadata = action["metadata"]["connector"]
        connector_name = connector_metadata.get("name")
        connector_id = connector_metadata.get("id", "default")

        connector = self.connector_registry.get(connector_id)
        if not connector:
            raise ValueError(f"Connector not found: {connector_name} (ID: {connector_id})")

        logger.info(f"Executing SCM Action: Action={action.get('properties', {}).get('action', 'Unknown')} "
                    f"Connector={connector_name} (ID={connector_id})")

        result = connector.validate_action(action.get("properties", {}).get("action", ""))
        return {"status": "success", "action": action, "result": result}


query_components_tool = QueryComponentsTool()
execute_scm_action_tool = SCMActionTool()