from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from utils.logger import logger
from connectors.registry import ConnectorRegistry
from engine.executors import SCMExecutor


class WorkflowEngine:
    """Engine for executing workflows by walking the node DAG sequentially."""

    registry: ConnectorRegistry
    executors: Dict[str, Any]

    def __init__(self) -> None:
        """Initialize the workflow engine with a connector registry and executors."""
        self.registry: ConnectorRegistry = ConnectorRegistry()
        self.executors: Dict[str, Any] = {"SCM_ACTION": SCMExecutor(self.registry)}
        logger.debug("Initialized WorkflowEngine")

    def register_executor(self, action_type: str, executor: Any) -> None:
        """
        Register an executor for a specific action type.

        Args:
            action_type: Action type key (e.g., ``'SCM_ACTION'``).
            executor: Executor instance implementing an ``execute(data)`` method.
        """
        self.executors[action_type] = executor
        logger.info(f"Registered executor for {action_type}")

    def execute(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the provided workflow.

        Args:
            workflow: Dict containing ``structure`` (list of nodes) and ``data``
                      (list of action descriptors).

        Returns:
            A result dict with ``status`` (``'completed'`` | ``'failed'``) and a
            ``steps`` list summarising each node outcome.
        """
        logger.info(f"Executing workflow with {len(workflow['structure'])} nodes")
        try:
            result: Dict[str, Any] = self._process_nodes(workflow["structure"], workflow)
            logger.info("Workflow execution completed successfully")
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def _process_nodes(self, nodes: List[Dict[str, Any]], workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process workflow nodes sequentially.

        Args:
            nodes: Ordered list of node dicts (``id``, ``name``, ``type``, ...).
            workflow: Full workflow dict (needed for ``data`` lookups).

        Returns:
            Accumulated execution result.
        """
        acc: Dict[str, Any] = {"status": "completed", "steps": []}
        for node in nodes:
            node_name: str = node["name"]
            node_type: str = node["type"].upper()
            data_entry: Optional[Dict[str, Any]] = next(
                (d for d in workflow["data"] if d["name"] == node_name), None
            )

            if not data_entry:
                logger.error(f"No data for node: {node_name}")
                acc["status"] = "failed"
                acc["steps"].append({"node": node_name, "status": "failed", "reason": "No matching data"})
                return acc

            step_result: Dict[str, Any] = {"node": node_name, "type": node_type}
            executor_map: Dict[Tuple[str, Optional[str]], Callable[..., Dict[str, Any]]] = {
                ("NORMAL", "EXTERNAL_SOURCE"): lambda: {"status": "triggered"},
                ("NORMAL", "SCM_ACTION"): self._execute_scm_action,
                ("BRANCH", None): self._evaluate_branch,
            }

            key: Tuple[str, Optional[str]] = (node_type, data_entry["type"] if node_type == "NORMAL" else None)
            executor: Callable[..., Dict[str, Any]] = executor_map.get(
                key, lambda: {"status": "failed", "result": "Invalid node type"}
            )
            step_result.update(executor() if key[0] != "NORMAL" else executor(data_entry))
            acc["steps"].append(step_result)
            logger.debug(f"Processed node: {step_result}")
        return acc

    def _execute_scm_action(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an SCM action using the registered executor.

        Args:
            data: Action descriptor containing ``scm_id`` and ``properties``.

        Returns:
            Execution result from the SCM executor.
        """
        executor: Optional[Any] = self.executors.get("SCM_ACTION")
        if not executor:
            logger.error("No SCM executor available")
            return {"status": "failed", "result": "No executor"}
        return executor.execute(data)

    def _evaluate_branch(self) -> Dict[str, str]:
        """Evaluate a branch node (placeholder for conditional logic)."""
        logger.debug("Evaluating branch node")
        return {"status": "branched to true"}