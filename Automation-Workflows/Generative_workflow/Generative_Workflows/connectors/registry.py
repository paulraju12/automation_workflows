from utils.logger import logger
from typing import Dict, Any


class ConnectorRegistry:
    """Registry for managing SCM connectors."""

    def __init__(self):
        """Initialize an empty connector registry."""
        self.connectors: Dict[str, Any] = {}
        logger.debug("Initialized ConnectorRegistry")

    def register(self, connector_id: str, connector: Any) -> None:
        """
        Register a connector with a unique ID.

        Args:
            connector_id (str): Unique identifier for the connector
            connector (Any): Connector instance
        """
        logger.info(f"Registering connector: {connector.name} (id: {connector_id})")
        self.connectors[connector_id] = connector

    def get(self, connector_id: str) -> Any:
        """
        Retrieve a connector by ID.

        Args:
            connector_id (str): Connector identifier

        Returns:
            Any: Connector instance or None if not found
        """
        connector = self.connectors.get(connector_id)
        logger.debug(f"Retrieving connector: {connector_id}, Found: {connector is not None}")
        return connector