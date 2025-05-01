# EAG-v1-Assignment-7

This project implements a Retrieval-Augmented Generation (RAG) agent specifically designed to answer questions about the Indian Premier League (IPL). The agent uses a combination of document processing, semantic search, and large language models to provide accurate and contextually relevant answers.

## Architecture

The system follows a Perception -> Memory -> Decision -> Action framework:

1. **Perception**: Understands and structures the user's query
2. **Memory**: Maintains context and history of interactions
3. **Decision**: Determines the next action based on current state
4. **Action**: Executes the chosen action and processes results

## Components

### Core Components

- `perception.py`: Processes and structures user queries
- `memory.py`: Manages context and interaction history
- `decision.py`: Makes decisions about next actions
- `action.py`: Executes chosen actions
- `utils.py`: Contains utility functions for document processing and embeddings
- `mcp_client.py`: Main client for interacting with the MCP server
- `mcp_action_server.py`: Server implementation for tool execution

### Data Processing

The system uses two document chunking strategies:
1. **Semantic Chunking**: Divides documents based on semantic meaning
2. **Naive Chunking**: Simple fixed-size chunking with overlap

Documents are processed and stored in a FAISS index for efficient similarity search.

## Setup

1. Set up environment variables:
```bash
touch .env
```

2. Add your OPENAI_API_KEY in the environmene variable
```bash
OPENAI_API_KEY="skllll"
```

3. Process documents (Create the index beforehand):
```bash
python utils.py
```

## Usage
1. Run the client:
```bash
python mcp_client.py
```

3. Enter your IPL-related question when prompted.

## Features

- Semantic document chunking for better context preservation
- FAISS-based efficient similarity search
- Context-aware answer generation
- Memory management for maintaining conversation context
- Tool-based approach for flexible action execution

## Data Sources

The system uses pre-scraped IPL match reports and related documents. These documents are processed and indexed before the agent can answer questions.