from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.config import ActiveConfig
from utils.logger import logger
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import json
import re
import time
import os


class WorkflowOutput(BaseModel):
    """Schema for structured workflow output."""
    structure: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of workflow nodes")
    data: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of workflow data entries")


class LLMService:
    """Service for interacting with the language model."""

    llm: ChatOpenAI
    max_retries: int

    def __init__(self) -> None:
        """Initialize the LLM service with OpenAI configuration."""
        self.llm: ChatOpenAI = ChatOpenAI(
            model="gpt-4o",
            api_key=ActiveConfig.OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=1000
        )
        self.max_retries: int = ActiveConfig.MAX_LLM_RETRIES
        logger.debug("Initialized LLMService")

    def invoke(self, template: str, structured: bool = False, retries: Optional[int] = None, **kwargs: Any) -> str:
        """
        Invoke the LLM with a prompt template.

        Args:
            template: Prompt template string with ``{placeholder}`` variables.
            structured: Whether to expect and validate JSON output.
            retries: Number of retries. Falls back to ``self.max_retries``.
            **kwargs: Template variable substitutions.

        Returns:
            The LLM response as a string. For structured calls the string is
            re-serialized JSON guaranteed to contain ``structure`` and ``data``.

        Raises:
            Exception: After all retries are exhausted.
        """
        prompt: PromptTemplate = PromptTemplate(input_variables=list(kwargs.keys()), template=template)
        effective_retries: int = retries if retries is not None else self.max_retries
        logger.debug(f"Invoking LLM with template: {template[:50]}..., structured: {structured}")

        for attempt in range(effective_retries):
            try:
                response: str = self.llm.invoke(prompt.format(**kwargs)).content.strip()
                if structured:
                    json_match: Optional[re.Match[str]] = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
                    json_str: str = json_match.group(1) if json_match else response
                    json_response: Dict[str, Any] = json.loads(json_str)
                    if not json_response.get("structure") or not json_response.get("data"):
                        raise ValueError("Missing 'structure' or 'data' in workflow")
                    logger.info("LLM structured response generated")
                    return json.dumps(json_response)
                logger.info(f"LLM plain text response: {response[:50]}...")
                return response
            except Exception as e:
                if attempt == effective_retries - 1:
                    logger.error(f"LLM invocation failed after {effective_retries} retries: {e}", exc_info=True)
                    raise Exception(f"LLM error: {str(e)}")
                logger.warning(f"Attempt {attempt + 1}/{effective_retries} failed: {e}")
                time.sleep(2 ** attempt)
        # Unreachable, but satisfies type checkers.
        raise RuntimeError("Retry loop exited unexpectedly")