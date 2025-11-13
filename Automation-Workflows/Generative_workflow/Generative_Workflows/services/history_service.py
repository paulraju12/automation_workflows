import json
import os
from utils.logger import logger


class HistoryService:
    """Service for managing conversation history persistence."""

    def __init__(self, filename: str = "history.json"):
        """
        Initialize the history service.

        Args:
            filename (str, optional): File path for history storage. Defaults to "history.json".
        """
        self.filename = filename
        logger.debug(f"Initialized HistoryService with filename: {filename}")

    def load(self, user_id: str) -> list:
        """
        Load conversation history for a user.

        Args:
            user_id (str): Unique user identifier

        Returns:
            list: List of history entries
        """
        if not os.path.exists(self.filename):
            logger.debug(f"History file {self.filename} not found, returning empty list")
            return []
        try:
            with open(self.filename, "r") as f:
                content = f.read().strip()
                history = json.loads(content).get(user_id, []) if content else []
                logger.info(f"Loaded history for user {user_id}: {len(history)} entries")
                return history
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load history: {e}", exc_info=True)
            return []

    def save(self, user_id: str, history: list) -> None:
        """
        Save conversation history for a user.

        Args:
            user_id (str): Unique user identifier
            history (list): List of history entries
        """
        data = {user_id: history}
        try:
            with open(self.filename, "w") as f:
                json.dump(data, f)
            logger.info(f"History saved for user {user_id}: {len(history)} entries")
        except IOError as e:
            logger.error(f"Failed to save history: {e}", exc_info=True)