# Workflow Automation Agent
# Overview:

This is an advanced, production-ready workflow management system designed for enterprise applications. It leverages AI-powered intent classification, workflow generation, and modification.

# Architecture

The agent is built around a state machine using LangGraph, with the following components:

State Management: AgentState tracks prompt, history, workflow, intent, and responses.

Intent Classifier: Uses an LLM to categorize user intent (new_workflow, modify_workflow, unclear, general).

Tool Detector: LLM-based extraction of tools/providers from prompts, validated against Pinecone.

Workflow Generator: Constructs JSON workflows with nodes and actions based on prompt and context.

Knowledge Base: Pinecone stores SCM tools and connectors (e.g., GitHub Enterprise, Bitbucket).

Error Handling: Decorators log errors and manage retries.

#PINECONE_API_KEY: Your Pinecone API key.
#LLM_API_KEY: API key for your LLM service.

# Features:

-Workflow Generation

-AI-driven workflow intent classification

-Dynamic workflow generation

-Robust error handling

-Centralized configuration management

-Comprehensive logging

-Flexible service architecture


# Installation

Clone the repository
Create a virtual environment

```bash
   python -m venv venv
   source venv/bin/activate
```

# On Windows, 
use`venv\Scripts\activate`

#Install dependencies

```bash
pip install -r requirements.txt
```

#Set up environment variables

```bash
export MAX_WORKFLOW_RETRIES=3
export WORKFLOW_TIMEOUT_SECONDS=30
```
#Usage
python
```bash
from workflow_graph import WorkflowGraph
workflow_graph = WorkflowGraph()
# Use workflow_graph for managing workflows
```

#Set Up Environment Variables: Create a .env file in the root directory:
```bash
PINECONE_API_KEY=your-pinecone-api-key
LLM_API_KEY=your-llm-api-key
MAX_WORKFLOW_RETRIES=3
WORKFLOW_TIMEOUT_SECONDS=30
```
#Run with Docker:
```bash
docker-compose up --build
```
This starts the agent on http://localhost:8000.
Usage
API Endpoint
Send a POST request to the workflow endpoint with a JSON payload containing your prompt:

```bash
curl -X POST "http://localhost:8000/api/v1/workflow" 
     -H "Content-Type: application/json" \
     -d '{"prompt": "design a workflow where github issue:created event triggers a slack message", "session_id": "123"}'
```
Response
Success: Returns a workflow JSON and a conversational message.


#MAX_WORKFLOW_RETRIES: Number of retries for LLM calls (default: 3).
#Update Pinecone with your tools:

