from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.config import ActiveConfig
from utils.logger import logger
from pydantic import BaseModel, Field
from typing import Dict, Optional
import json
import re
import time
import os


class WorkflowOutput(BaseModel):
    """Schema for structured workflow output."""
    structure: Optional[list[Dict]] = Field(default=None, description="List of workflow nodes")
    data: Optional[list[Dict]] = Field(default=None, description="List of workflow data entries")


class LLMService:
    """Service for interacting with the language model."""

    def __init__(self):
        """Initialize the LLM service with OpenAI configuration."""
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=ActiveConfig.OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=1000
        )
        self.max_retries = ActiveConfig.MAX_LLM_RETRIES
        logger.debug("Initialized LLMService")

    def invoke(self, template: str, structured: bool = False, retries: int = None, **kwargs) -> str:
        """
        Invoke the LLM with a prompt template.

        Args:
            template (str): Prompt template
            structured (bool, optional): Whether to expect JSON output. Defaults to False.
            retries (int, optional): Number of retries. Defaults to self.max_retries.
            **kwargs: Template variables

        Returns:
            str: LLM response
        """
        prompt = PromptTemplate(input_variables=list(kwargs.keys()), template=template)
        retries = retries if retries is not None else self.max_retries
        logger.debug(f"Invoking LLM with template: {template[:50]}..., structured: {structured}")

        for attempt in range(retries):
            try:
                response = self.llm.invoke(prompt.format(**kwargs)).content.strip()
                if structured:
                    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
                    json_str = json_match.group(1) if json_match else response
                    json_response = json.loads(json_str)
                    if not json_response.get("structure") or not json_response.get("data"):
                        raise ValueError("Missing 'structure' or 'data' in workflow")
                    logger.info(f"LLM structured response generated")
                    return json.dumps(json_response)
                logger.info(f"LLM plain text response: {response[:50]}...")
                return response
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"LLM invocation failed after {retries} retries: {e}", exc_info=True)
                    raise Exception(f"LLM error: {str(e)}")
                logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
                time.sleep(2 ** attempt)