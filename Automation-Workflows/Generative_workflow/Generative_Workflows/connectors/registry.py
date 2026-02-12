from utils.logger import logger
from typing import Dict, Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class Connector(Protocol):
    """Structural type for anything registered as a connector."""
    name: str

    def validate_action(self, action: str) -> bool:
        ...


class ConnectorRegistry:
    """Thread-safe registry for managing SCM and ticketing connectors."""

    connectors: Dict[str, Any]

    def __init__(self) -> None:
        """Initialize an empty connector registry."""
        self.connectors: Dict[str, Any] = {}
        logger.debug("Initialized ConnectorRegistry")

    def register(self, connector_id: str, connector: Any) -> None:
        """
        Register a connector with a unique ID.

        Args:
            connector_id: Unique identifier for the connector.
            connector: Connector instance exposing at minimum a ``name`` attribute.
        """
        logger.info(f"Registering connector: {connector.name} (id: {connector_id})")
        self.connectors[connector_id] = connector

    def get(self, connector_id: str) -> Optional[Any]:
        """
        Retrieve a connector by ID.

        Args:
            connector_id: Connector identifier.

        Returns:
            The connector instance, or ``None`` if no connector is registered
            under the given ID.
        """
        connector: Optional[Any] = self.connectors.get(connector_id)
        logger.debug(f"Retrieving connector: {connector_id}, Found: {connector is not None}")
        return connector

    def list_ids(self) -> list[str]:
        """Return a sorted list of all registered connector IDs."""
        return sorted(self.connectors.keys())

    def __contains__(self, connector_id: str) -> bool:
        """Support ``'id' in registry`` syntax."""
        return connector_id in self.connectors

    def __len__(self) -> int:
        """Return the number of registered connectors."""
        return len(self.connectors)