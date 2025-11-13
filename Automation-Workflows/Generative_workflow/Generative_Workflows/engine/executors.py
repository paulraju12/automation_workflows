from connectors.registry import ConnectorRegistry
from utils.logger import logger


class SCMExecutor:
    """Executor for SCM actions."""

    def __init__(self, registry: ConnectorRegistry):
        """
        Initialize the SCM executor with a connector registry.

        Args:
            registry (ConnectorRegistry): Registry of connectors
        """
        self.registry = registry
        logger.debug("Initialized SCMExecutor")

    def execute(self, data: dict) -> dict:
        """
        Execute an SCM action using the specified connector.

        Args:
            data (dict): Action data including scm_id and properties

        Returns:
            dict: Execution result
        """
        logger.debug(f"Executing SCM action: {data}")
        connector_id = data.get("scm_id")
        connector = self.registry.get(connector_id)
        if not connector:
            logger.error(f"Connector {connector_id} not found")
            return {"status": "failed", "result": f"Connector {connector_id} not found"}
        result = connector.validate_action(data.get("properties", {}).get("action", ""))
        logger.info(f"SCM action executed: {result}")
        return {"status": "success", "result": "SCM action executed"}