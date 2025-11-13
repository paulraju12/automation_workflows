from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from utils.config import ActiveConfig
from utils.logger import logger
import time


class VectorStore:
    """Manages vector storage and querying with Pinecone."""

    def __init__(self):
        """Initialize the vector store with embedding model and Pinecone index."""
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.pc = Pinecone(api_key=ActiveConfig.PINECONE_API_KEY)
        self.index = self.pc.Index("workflow-components")
        logger.debug("Initialized VectorStore")

    def query(self, text: str, top_k: int = 10, retries: int = 3) -> list:
        """
        Query the Pinecone index for similar vectors.

        Args:
            text (str): Query text to embed
            top_k (int, optional): Number of top results. Defaults to 10.
            retries (int, optional): Number of retry attempts. Defaults to 3.

        Returns:
            list: List of matching vectors with metadata
        """
        logger.debug(f"Querying Pinecone: text='{text}', top_k={top_k}")
        embedding = self.model.encode([text])[0].tolist()
        for attempt in range(retries):
            try:
                results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)["matches"]
                logger.info(f"Pinecone query successful, retrieved {len(results)} results")
                return results
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Pinecone query failed after {retries} retries: {e}", exc_info=True)
                    return []
                logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
                time.sleep(2 ** attempt)