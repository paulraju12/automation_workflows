import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import hashlib
from agents.workflow_graph import graph
from engine.workflow_engine import WorkflowEngine
from utils.logger import logger
import redis.asyncio as redis
import asyncpg
import uuid
from typing import Optional, Dict
import os
from utils.config import ActiveConfig
from dotenv import load_dotenv

load_dotenv()

redis_client = redis.Redis(host=ActiveConfig.REDIS_HOST, port=ActiveConfig.REDIS_PORT, db=0)
db_pool = None
engine = WorkflowEngine()

try:
    with open("templates/workflow_template.json", "r") as f:
        WORKFLOW_TEMPLATE = json.load(f)
except Exception as e:
    logger.error(f"Failed to load workflow_template.json: {str(e)}")
    raise


class PromptRequest(BaseModel):
    """Request schema for processing a prompt."""
    prompt: str
    session_id: Optional[str] = None


class WorkflowResponse(BaseModel):
    """Response schema for workflow processing."""
    conversation: str
    session_id: str
    workflow: Optional[Dict] = None
    next_question: Optional[str] = None
    interaction_id: Optional[int] = None


async def get_db_connection(max_retries: int = 5, delay: int = 5) -> asyncpg.Pool:
    """Establish a connection pool to PostgreSQL with retry logic."""
    global db_pool
    if db_pool is None:
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting PostgreSQL connection (attempt {attempt + 1}/{max_retries})")
                db_pool = await asyncpg.create_pool(
                    database="workflow_db",
                    user=os.getenv("POSTGRES_USER", "workflow_user"),
                    password=os.getenv("POSTGRES_PASSWORD"),
                    host="postgres",
                    port=5432
                )
                logger.info("Database pool initialized successfully")
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=500, detail=f"DB connection failed: {str(e)}")
                await asyncio.sleep(delay)
    return db_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for managing startup and shutdown events."""
    await get_db_connection()
    logger.info("Application started with DB and Redis connections")

    yield

    if db_pool is not None:
        await db_pool.close()
        logger.info("Database pool closed")
    await redis_client.close()
    logger.info("Redis connection closed")


app = FastAPI(title="Workflow Agent API", lifespan=lifespan)


@app.post("/api/v1/workflow", response_model=WorkflowResponse)
async def process_prompt(request: PromptRequest):
    """Process a user prompt and generate a workflow or response, maintaining conversation flow."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()
        cache_key = f"cache:{session_id}:{prompt_hash}"
        state_key = f"state:{session_id}"

        logger.info(f"Processing request for session_id: {session_id}, prompt: {request.prompt}")

        # Check for cached response
        cached = await redis_client.get(cache_key)
        if cached is not None:
            logger.info(f"Cache hit for key: {cache_key}")
            return WorkflowResponse(**json.loads(cached.decode()))

        # Get database pool connection
        pool = await get_db_connection()

        # Check for cached state in Redis
        cached_state = await redis_client.get(state_key)
        if cached_state is not None:
            state = json.loads(cached_state.decode())
            logger.debug(f"Loaded state from Redis for session {session_id}")
        else:
            async with pool.acquire() as conn:
                logger.info(f"Fetching history for session_id: {session_id}")
                history = await conn.fetch(
                    "SELECT prompt, response FROM interactions WHERE session_id = $1 ORDER BY timestamp",
                    session_id
                )
                latest_state = await conn.fetchrow(
                    "SELECT state FROM interactions WHERE session_id = $1 ORDER BY timestamp DESC LIMIT 1",
                    session_id
                )

            state = json.loads(latest_state["state"]) if latest_state and latest_state["state"] is not None else {
                "prompt": "",
                "history": [(r["prompt"], r["response"]) for r in history],
                "workflow": {"structure": [], "data": []},
                "intent": None,
                "response": "",
                "awaiting_input": False,
                "next_question": "",
                "error": {}
            }
            logger.debug(f"Initialized state for session {session_id} from DB")

        # Update state with current prompt
        state["prompt"] = request.prompt

        # Invoke the workflow graph
        logger.info("Invoking workflow graph")
        result = await graph.ainvoke(state)
        state.update(result)

        # Set conversation and workflow based on graph output
        workflow = state["workflow"] if state["workflow"].get("structure") else None
        conversation = state["response"] or "Here’s your response."

        # Store updated state and interaction in DB
        async with pool.acquire() as conn:
            logger.info("Inserting interaction into database")
            interaction_id = await conn.fetchval(
                "INSERT INTO interactions (session_id, prompt, response, workflow, state, timestamp) "
                "VALUES ($1, $2, $3, $4, $5, NOW()) RETURNING id",
                session_id, request.prompt, conversation, json.dumps(state["workflow"]), json.dumps(state)
            )

        # Update state in Redis
        await redis_client.setex(state_key, 86400, json.dumps(state))
        logger.debug(f"Updated state in Redis for session {session_id}")

        # Prepare response
        response_data = WorkflowResponse(
            conversation=conversation,
            session_id=session_id,
            workflow=workflow,
            next_question=state["next_question"] or "Anything else to add?",
            interaction_id=interaction_id
        )
        await redis_client.setex(cache_key, 86400, json.dumps(response_data.dict()))
        logger.info(f"Processed prompt successfully for session {session_id}")
        return response_data
    except Exception as e:
        logger.error(f"Unexpected error in process_prompt: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)