import logging
from logging.handlers import RotatingFileHandler
import os
from utils.config import ActiveConfig

# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logger
logger = logging.getLogger("WorkflowAgent")
logger.setLevel(getattr(logging, ActiveConfig.LOG_LEVEL))

# Clear existing handlers
if logger.handlers:
    logger.handlers.clear()

# File handler with rotation
file_handler = RotatingFileHandler(
    filename=os.path.join(log_dir, "workflow_agent.log"),
    maxBytes=10 * 1024 * 1024,  # 10MB per file
    backupCount=18  # ~6 months of logs
)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
file_handler.setLevel(getattr(logging, ActiveConfig.LOG_LEVEL))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(getattr(logging, ActiveConfig.LOG_LEVEL))
logger.addHandler(console_handler)