# ML-AI Framework

Production-grade multi-agent ML framework with CrewAI and LangGraph orchestration.

## Features

- **Multi-Agent Workflows**: CrewAI and LangGraph implementations
- **Pydantic V2 Models**: Strict validation with comprehensive schemas
- **Custom Tools**: Data loading, cleaning, EDA, and model training
- **Error Handling**: Retry logic, circuit breakers, and fallback models
- **Structured Logging**: JSON logging with structlog
- **AG-UI Server**: FastAPI server with streaming support
- **Type Safety**: Full type hints and mypy compliance

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running Examples

```bash
# CrewAI workflow
python examples/simple_workflow.py

# LangGraph workflow
python examples/langgraph_workflow.py
```

### Starting AG-UI Server

```bash
# Start server
python -m src.ag_ui_server

# Or with uvicorn
uvicorn src.ag_ui_server:app --reload
```

## Architecture

### Core Components

1. **Models** (`src/models/`): Pydantic schemas with validation
2. **Tools** (`src/tools/`): Custom agent tools
3. **Workflows** (`src/workflows/`): CrewAI and LangGraph implementations
4. **Utils** (`src/utils/`): Logging and error handling
5. **AG-UI Server** (`src/ag_ui_server.py`): FastAPI streaming server

### Workflows

#### CrewAI Workflow
- 5 specialized agents
- Sequential or hierarchical execution
- Tool-based collaboration

#### LangGraph Workflow
- State-based execution
- Graph-based flow control
- Async support

## API Usage

### Execute Workflow (Non-streaming)

```bash
curl -X POST http://localhost:8000/workflow/execute \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/dataset.csv",
    "target_column": "target",
    "workflow_type": "crewai"
  }'
```

### Stream Workflow (AG-UI Protocol)

```bash
curl -X POST http://localhost:8000/workflow/stream \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/dataset.csv",
    "target_column": "target",
    "workflow_type": "langgraph"
  }'
```

## Development

### Testing

```bash
pytest tests/ --cov=src
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Configuration

Environment variables (see `.env.example`):

- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_MODEL`: Default model (gpt-4)
- `LOG_LEVEL`: Logging level
- `SERVER_PORT`: AG-UI server port

## License

MIT