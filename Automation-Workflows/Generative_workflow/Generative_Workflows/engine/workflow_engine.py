from typing import Dict, List
from utils.logger import logger
from connectors.registry import ConnectorRegistry
from engine.executors import SCMExecutor


class WorkflowEngine:
    """Engine for executing workflows."""

    def __init__(self):
        """Initialize the workflow engine with a connector registry and executors."""
        self.registry = ConnectorRegistry()
        self.executors = {"SCM_ACTION": SCMExecutor(self.registry)}
        logger.debug("Initialized WorkflowEngine")

    def register_executor(self, action_type: str, executor) -> None:
        """
        Register an executor for a specific action type.

        Args:
            action_type (str): Type of action (e.g., 'SCM_ACTION')
            executor: Executor instance
        """
        self.executors[action_type] = executor
        logger.info(f"Registered executor for {action_type}")

    def execute(self, workflow: Dict) -> Dict:
        """
        Execute the provided workflow.

        Args:
            workflow (Dict): Workflow structure and data

        Returns:
            Dict: Execution result
        """
        logger.info(f"Executing workflow with {len(workflow['structure'])} nodes")
        try:
            result = self._process_nodes(workflow["structure"], workflow)
            logger.info("Workflow execution completed successfully")
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def _process_nodes(self, nodes: List[Dict], workflow: Dict) -> Dict:
        """
        Process workflow nodes sequentially.

        Args:
            nodes (List[Dict]): List of workflow nodes
            workflow (Dict): Full workflow data

        Returns:
            Dict: Accumulated result of node processing
        """
        acc = {"status": "completed", "steps": []}
        for node in nodes:
            node_name, node_type = node["name"], node["type"].upper()
            data_entry = next((d for d in workflow["data"] if d["name"] == node_name), None)

            if not data_entry:
                logger.error(f"No data for node: {node_name}")
                acc["status"] = "failed"
                acc["steps"].append({"node": node_name, "status": "failed", "reason": "No matching data"})
                return acc

            step_result = {"node": node_name, "type": node_type}
            executor_map = {
                ("NORMAL", "EXTERNAL_SOURCE"): lambda: {"status": "triggered"},
                ("NORMAL", "SCM_ACTION"): self._execute_scm_action,
                ("BRANCH", None): self._evaluate_branch
            }

            key = (node_type, data_entry["type"] if node_type == "NORMAL" else None)
            executor = executor_map.get(key, lambda: {"status": "failed", "result": "Invalid node type"})
            step_result.update(executor() if key[0] != "NORMAL" else executor(data_entry))
            acc["steps"].append(step_result)
            logger.debug(f"Processed node: {step_result}")
        return acc

    def _execute_scm_action(self, data: Dict) -> Dict:
        """
        Execute an SCM action using the registered executor.

        Args:
            data (Dict): Action data

        Returns:
            Dict: Execution result
        """
        executor = self.executors.get("SCM_ACTION")
        if not executor:
            logger.error("No SCM executor available")
            return {"status": "failed", "result": "No executor"}
        return executor.execute(data)

    def _evaluate_branch(self) -> Dict:
        """Evaluate a branch node (placeholder)."""
        logger.debug("Evaluating branch node")
        return {"status": "branched to true"}