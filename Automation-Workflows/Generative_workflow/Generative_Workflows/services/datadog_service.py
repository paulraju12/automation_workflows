from datadog import initialize, statsd
from utils.config import ActiveConfig
from utils.logger import logger
import os


class DataDogService:
    """Service for integrating with DataDog for error tracking and metrics."""

    @staticmethod
    def init():
        """
        Initialize DataDog with API key and configuration.

        Raises:
            ValueError: If DATADOG_API_KEY is not set
        """
        api_key = os.getenv("DATADOG_API_KEY")
        if not api_key:
            logger.error("DATADOG_API_KEY not set in environment")
            raise ValueError("DATADOG_API_KEY is required for DataDog integration")

        options = {
            "api_key": api_key,
            "app_key": os.getenv("DATADOG_APP_KEY", ""),
            "statsd_host": os.getenv("DATADOG_HOST", "datadog-agent"),
            "statsd_port": 8125
        }
        initialize(**options)
        logger.info("DataDog initialized successfully")

    @staticmethod
    def capture_exception(e: Exception):
        """
        Capture an exception in DataDog.

        Args:
            e (Exception): Exception to report
        """
        tags = {"environment": os.getenv("ENVIRONMENT", "production")}
        statsd.increment("workflow_agent.errors", tags=tags)
        logger.debug(f"Captured exception in DataDog: {str(e)}")

    @staticmethod
    def increment_metric(metric_name: str, tags: dict = None):
        """
        Increment a custom metric in DataDog.

        Args:
            metric_name (str): Name of the metric
            tags (dict, optional): Additional tags for the metric
        """
        statsd.increment(metric_name, tags=tags or {})
        logger.debug(f"Incremented metric: {metric_name}")