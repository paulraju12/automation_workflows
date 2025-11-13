from utils.logger import logger


class WorkflowError(Exception):
    """Base exception for workflow-related errors."""

    def __init__(self, message: str):
        """
        Initialize the workflow error.

        Args:
            message (str): Error message
        """
        super().__init__(message)
        logger.error(f"WorkflowError: {message}")


class IntentClassificationError(WorkflowError):
    """Exception raised when intent classification fails."""
    pass


class WorkflowExecutionError(WorkflowError):
    """Exception raised when workflow execution fails."""
    pass