from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

PROFILE= os.getenv("PROFILE","development")

#@dataclass(frozen=True)
class Config:
    DEBUG = False
    LOG_LEVEL = "INFO"
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "workflow_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB = "workflow_db"
    POSTGRES_HOST = "postgres"
    POSTGRES_PORT = 5432
    REDIS_HOST = "redis"
    REDIS_PORT = 6379
    MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", 3))
    MAX_WORKFLOW_RETRIES = int(os.getenv("MAX_WORKFLOW_RETRIES", 3))
    WORKFLOW_TIMEOUT_SECONDS = int(os.getenv("WORKFLOW_TIMEOUT_SECONDS", 30))


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class TestingConfig(Config):
    DEBUG = True
    LOG_LEVEL = "WARNING"

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "ERROR"

# Profile Mapping
config_map = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}

# Get the active config
ActiveConfig = config_map.get(PROFILE, DevelopmentConfig)()