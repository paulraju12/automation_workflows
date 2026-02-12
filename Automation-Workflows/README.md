# Workflow Automation Agent

**AI-powered workflow generation engine that classifies user intent and produces executable JSON workflows for SCM and ticketing tools, backed by a LangGraph state machine, Pinecone vector retrieval, and Redis-accelerated caching.**

---

## Architecture

```
                                 Workflow Automation Agent
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                                                                              │
 │  User ──► FastAPI (/api/v1/workflow)                                         │
 │              │                                                               │
 │              ├─── Redis Cache ──► HIT? ──► Return cached response            │
 │              │                                                               │
 │              ▼ MISS                                                          │
 │        ┌───────────────────────────────────────────────┐                     │
 │        │         LangGraph State Machine               │                     │
 │        │                                               │                     │
 │        │  ┌────────────────┐                           │                     │
 │        │  │ classify_intent│ ◄── LLM (GPT-4o)         │                     │
 │        │  └───────┬────────┘     + Pinecone context    │                     │
 │        │          │                                    │                     │
 │        │    ┌─────┴──────────────────────┐             │                     │
 │        │    │         Router             │             │                     │
 │        │    ├─────────┬─────────┬────────┤             │                     │
 │        │    ▼         ▼         ▼        ▼             │                     │
 │        │ generate  modify   handle    handle           │                     │
 │        │ workflow  workflow  unclear   general          │                     │
 │        │    │         │         │        │              │                     │
 │        │    └─────────┴─────────┴────────┘              │                     │
 │        │                   │                           │                     │
 │        └───────────────────┼───────────────────────────┘                     │
 │                            ▼                                                 │
 │                    Workflow Engine                                            │
 │                       │      │                                               │
 │                       ▼      ▼                                               │
 │                 SCM Exec  Ext Source                                          │
 │                       │                                                      │
 │              ┌────────┴────────┐                                             │
 │              ▼                 ▼                                              │
 │         PostgreSQL         Redis                                             │
 │        (interactions)    (state cache)                                        │
 │                                                                              │
 └──────────────────────────────────────────────────────────────────────────────┘
```

### Data flow

1. **User** sends a natural-language prompt via the REST API.
2. **FastAPI** checks **Redis** for a cached response (SHA-256 of prompt + session).
3. On a cache miss the request enters the **LangGraph state machine**.
4. The **classify_intent** node calls **GPT-4o** with **Pinecone** vector context to determine intent (`new_workflow` | `modify_workflow` | `general` | `unclear`).
5. The **conditional router** dispatches to the appropriate handler node.
6. Workflow JSON is generated/modified or a conversational reply is produced.
7. The interaction is persisted to **PostgreSQL** and the state is cached in **Redis** (24 h TTL).

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| API | FastAPI + Uvicorn | Async HTTP, auto-generated OpenAPI docs |
| Orchestration | LangGraph (StateGraph) | Deterministic, debuggable state machine for multi-step AI flows |
| LLM | OpenAI GPT-4o via LangChain | Intent classification and JSON workflow generation |
| Vector Store | Pinecone | Semantic retrieval of SCM provider/connector metadata |
| Embeddings | sentence-transformers (MiniLM-L6-v2) | 384-dim embeddings for Pinecone upsert and query |
| Caching | Redis 7 | Response and session-state caching with TTL |
| Persistence | PostgreSQL 15 | Interaction history, feedback, session state |
| Observability | DataDog + RotatingFileHandler | Metrics, error tracking, structured logs |
| Containerization | Docker + docker-compose | One-command local environment |

---

## Features

- **Intent classification** -- LLM-powered four-way classifier with Pinecone-enriched context.
- **Workflow generation** -- Produces JSON workflows with `structure` (DAG nodes) and `data` (action metadata) conforming to a published schema.
- **Workflow modification** -- Iteratively refines an existing workflow based on follow-up prompts.
- **Conversational fallback** -- Handles general questions and ambiguous inputs gracefully.
- **Session management** -- UUID-based sessions with history loaded from PostgreSQL and cached in Redis.
- **Retry with exponential backoff** -- LLM and Pinecone calls are retried automatically.
- **Enterprise error handling** -- Decorator-based error capture with full traceback logging.
- **Connector registry** -- Pluggable SCM and ticketing connectors (GitHub, Bitbucket, Jira, etc.).
- **CI pipeline** -- GitHub Actions with pytest and mypy on every push/PR.

---

## Getting Started

### Prerequisites

- Docker and docker-compose
- An OpenAI API key
- A Pinecone API key and environment

### 1. Clone and configure

```bash
git clone <repo-url>
cd Automation-Workflows/Generative_workflow/Generative_Workflows

cp .env.example .env
# Fill in OPENAI_API_KEY, PINECONE_API_KEY, POSTGRES_PASSWORD, etc.
```

### 2. Start everything

```bash
docker-compose up --build
```

This launches three containers:

| Service | Port |
|---|---|
| `workflow-agent` (FastAPI) | `localhost:8000` |
| `postgres` | `localhost:5432` |
| `redis` | `localhost:6379` |

### 3. Seed the vector store (first run only)

```bash
docker-compose exec workflow-agent python embeddings/generate_embeddings.py
```

### 4. Verify

```bash
curl http://localhost:8000/docs
```

---

## Running Tests

```bash
# From the Generative_Workflows directory:
pip install pytest pytest-asyncio httpx
pytest tests/ -v --tb=short
```

All tests mock external services (OpenAI, Pinecone, Redis, PostgreSQL), so they run offline and complete in seconds.

### Test structure

```
tests/
  conftest.py            # Shared fixtures, sample data
  test_workflow_graph.py  # LangGraph nodes, routing, error handling
  test_api.py            # FastAPI endpoint, caching, DB persistence
  test_services.py       # LLMService, HistoryService, ConnectorRegistry, WorkflowEngine
```

---

## API Reference

### `POST /api/v1/workflow`

Process a natural-language prompt and return a workflow or conversational response.

**Request body**

```json
{
  "prompt": "create a workflow where a GitHub issue triggers a Jira ticket",
  "session_id": "optional-uuid"
}
```

**Response** (`200 OK`)

```json
{
  "conversation": "Let's build it! I've created a workflow based on your request.",
  "session_id": "c1a2b3d4-...",
  "workflow": {
    "structure": [
      { "id": "node-1", "name": "github-issue-created", "type": "normal", "content": {}, "position": { "x": 58, "y": 261 } },
      { "id": "node-2", "name": "jira-create-ticket", "type": "normal", "content": {}, "position": { "x": 158, "y": 261 } }
    ],
    "data": [
      { "id": "node-1", "name": "github-issue-created", "type": "SCM_ACTION", "version": "1.0", "properties": { "action": "issue_created" }, "metadata": { "title": "GitHub Issue Created", "connector": { "name": "GitHub" } }, "scm_id": "adf1f67b-..." }
    ]
  },
  "next_question": "Anything else to add?",
  "interaction_id": 42
}
```

**Error responses**

| Status | Reason |
|---|---|
| `422` | Missing or invalid `prompt` field |
| `500` | Internal server error (DB or LLM failure) |

---

## Design Decisions

### Why LangGraph?

LangGraph gives us a **typed, compiled state machine** rather than an ad-hoc chain of LLM calls. Each node is a pure function over `AgentState`, making the flow testable, deterministic, and easy to extend with new intent branches (e.g., `delete_workflow`, `share_workflow`) without touching the router.

### Why Pinecone?

SCM providers and connector metadata change slowly but the search space is large. Pinecone lets us do **semantic nearest-neighbor lookup** (cosine similarity on 384-dim MiniLM embeddings) so the LLM receives the most relevant context regardless of exact keyword matches. This beats a keyword filter on a SQL table for recall quality.

### Why Redis for caching?

Two caching layers share the same Redis instance:

1. **Response cache** -- keyed by `session_id + SHA-256(prompt)`. Identical prompts within the same session skip the LLM entirely.
2. **State cache** -- keyed by `session_id`. Avoids a PostgreSQL round-trip to reconstruct conversation history on every request.

Both use a 24-hour TTL, which is a deliberate tradeoff (see below).

---

## Tradeoffs

| Decision | Upside | Downside |
|---|---|---|
| GPT-4o at temperature 0.3 | High accuracy for classification and structured JSON output | ~$0.005/request; latency ~1-3 s per LLM call |
| 24 h Redis cache TTL | Eliminates redundant LLM calls within a session | Stale responses if the user changes their mind semantically without changing the literal prompt |
| Pinecone as the sole context source | Fast, scalable semantic search | Adds an external dependency; cold starts if the index is empty |
| Synchronous LLM calls inside LangGraph nodes | Simpler reasoning and retry logic | Blocks the async event loop; limits throughput under high concurrency |
| File-based HistoryService fallback | Works without PostgreSQL for local dev | Not suitable for multi-instance production deployments |

---

## What I Would Improve

Given more time, the following changes would move this from a strong MVP to a production-grade platform:

1. **Async LLM calls** -- Swap `llm.invoke()` for `llm.ainvoke()` inside the LangGraph nodes so the FastAPI event loop is never blocked. This alone would roughly double throughput.

2. **WebSocket streaming** -- Stream partial workflow JSON and conversational tokens to the client via WebSockets instead of waiting for the full LLM response. This cuts perceived latency from seconds to milliseconds.

3. **Granular cache invalidation** -- Replace the flat 24 h TTL with an event-driven strategy: invalidate cache entries when the user modifies the workflow or when the Pinecone index is updated, preserving freshness without sacrificing hit rate.

4. **Structured output mode** -- Use OpenAI's function-calling / JSON mode instead of regex-parsing the response. This eliminates an entire class of parsing bugs.

5. **Workflow versioning** -- Store workflow snapshots in PostgreSQL with a `version` column so users can diff, rollback, and audit changes.

6. **Rate limiting and auth** -- Add JWT-based authentication and per-user rate limiting before exposing the API externally.

7. **Observability** -- Integrate OpenTelemetry traces across LangGraph nodes for end-to-end latency profiling, beyond the current DataDog metrics.

8. **Horizontal scaling** -- Move session state from in-memory Redis to a Redis Cluster or DynamoDB so the service can scale horizontally behind a load balancer without sticky sessions.

---

## Project Structure

```
Generative_Workflows/
├── agents/
│   ├── workflow_graph.py    # LangGraph state machine (classify, generate, modify, fallback)
│   └── tools.py             # Pinecone query tool, SCM action tool
├── connectors/
│   ├── registry.py          # Pluggable connector registry
│   └── scm_connectors.py    # Base SCM connector with action validation
├── embeddings/
│   ├── generate_embeddings.py  # One-time Pinecone index seeding script
│   └── vector_store.py      # Pinecone query wrapper with retry logic
├── engine/
│   ├── executors.py         # SCM action executor
│   └── workflow_engine.py   # DAG node processor
├── models/
│   └── workflow_state.py    # Legacy state dataclass
├── services/
│   ├── datadog_service.py   # DataDog metrics integration
│   ├── history_service.py   # File-based conversation persistence
│   └── llm_service.py       # OpenAI GPT-4o wrapper with retry + JSON parsing
├── templates/
│   └── workflow_template.json  # JSON schema for workflow output
├── tests/
│   ├── conftest.py          # Shared fixtures, sample data
│   ├── test_workflow_graph.py
│   ├── test_api.py
│   └── test_services.py
├── utils/
│   ├── config.py            # Profile-based configuration (dev/test/prod)
│   ├── exceptions.py        # Custom exception hierarchy
│   └── logger.py            # Rotating file + console logger
├── app.py                   # FastAPI application entry point
├── main.py                  # CLI entry point for local testing
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
├── .env.example
└── .github/workflows/ci.yml
```

---

## License

Internal use. See repository root for license details.
