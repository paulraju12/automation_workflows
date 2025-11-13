from utils.logger import logger


class SCMConnector:
    """Base class for SCM connectors."""

    def __init__(self, provider_id: str, name: str):
        """
        Initialize an SCM connector.

        Args:
            provider_id (str): Unique provider identifier
            name (str): Connector name
        """
        self.id = provider_id
        self.name = name
        logger.info(f"Initialized SCM connector: {name} ({provider_id})")

    def validate_action(self, action: str) -> bool:
        """
        Validate if an action is supported by the connector.

        Args:
            action (str): Action to validate

        Returns:
            bool: True if valid, False otherwise
        """
        logger.debug(f"Validating action '{action}' for {self.name}")
        valid_actions = {"commit", "push", "pull_request"}
        is_valid = action in valid_actions
        logger.info(f"Action validation result: {is_valid} for action '{action}'")
        return is_valid