# Quick Reference Guide

**ML-AI Framework v0.1.0** - Fast access to common commands and configurations

---

## Installation

```bash
# Quick setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## Common Commands

### Running the Application

```bash
# Start AG-UI server
uvicorn src.ag_ui_server:app --reload

# Or via Python module
python -m src.ag_ui_server

# Production mode (no reload)
uvicorn src.ag_ui_server:app --host 0.0.0.0 --port 8000
```

### Running Examples

```bash
# Simple CrewAI workflow
python examples/simple_workflow.py

# LangGraph workflow
python examples/langgraph_workflow.py

# Quickstart examples
python examples/quickstart/01_basic_workflow.py
python examples/quickstart/02_custom_tools.py
python examples/quickstart/03_error_handling.py
python examples/quickstart/04_testing.py

# Use case examples
python examples/iris_analysis/run_crewai.py
python examples/iris_analysis/run_langgraph.py
python examples/housing_regression/run_crewai.py
python examples/housing_regression/run_langgraph.py
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_tools.py

# Run specific test
pytest tests/unit/test_tools.py::test_load_data

# Run by marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m llm          # LLM-related tests

# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x
```

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Check formatting (no changes)
black --check src/ tests/

# Lint code
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/

# Type checking
mypy src/

# Run all quality checks
black src/ tests/ && ruff check src/ tests/ && mypy src/
```

---

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Execute Workflow (Non-Streaming)

**Endpoint**: `POST /workflow/execute`

**Request**:
```bash
curl -X POST http://localhost:8000/workflow/execute \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/iris.csv",
    "target_column": "species",
    "workflow_type": "crewai"
  }'
```

**Response**:
```json
{
  "status": "success",
  "workflow_type": "crewai",
  "results": {
    "data_summary": {...},
    "model_metrics": {...},
    "insights": [...],
    "recommendations": [...]
  },
  "execution_time": 45.2
}
```

### Stream Workflow (AG-UI Protocol)

**Endpoint**: `POST /workflow/stream`

**Request**:
```bash
curl -X POST http://localhost:8000/workflow/stream \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/housing.csv",
    "target_column": "price",
    "workflow_type": "langgraph"
  }'
```

**Response** (Server-Sent Events):
```
data: {"type": "status", "message": "Starting workflow"}
data: {"type": "agent", "agent": "Data Loader", "status": "working"}
data: {"type": "result", "data": {...}}
data: {"type": "complete"}
```

### Health Check (Recommended Addition)

**Endpoint**: `GET /health`

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-05T10:30:00Z",
  "version": "0.1.0"
}
```

---

## Configuration Quick Reference

### Environment Variables

**Required**:
```bash
OPENAI_API_KEY=sk-your-key-here
```

**Common Customizations**:
```bash
# Model Selection
OPENAI_MODEL=gpt-4              # or gpt-3.5-turbo for faster/cheaper
OPENAI_TEMPERATURE=0.7          # 0.0-2.0 (lower = more deterministic)

# Server
SERVER_PORT=8000                # Change port
SERVER_RELOAD=true              # Auto-reload on code changes (dev only)

# Logging
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
LOG_JSON=true                   # JSON format for production

# Performance
MAX_WORKERS=4                   # Thread pool size
ASYNC_ENABLED=true              # Enable async operations

# Workflow
MAX_AGENT_ITERATIONS=10         # Max iterations per agent
DEFAULT_WORKFLOW_TYPE=crewai    # crewai or langgraph

# Error Handling
RETRY_MAX_ATTEMPTS=3            # Number of retries
CIRCUIT_BREAKER_THRESHOLD=5     # Failures before circuit opens
```

### Accessing Settings in Code

```python
from config.settings import get_settings

settings = get_settings()
api_key = settings.openai_api_key
model = settings.openai_model
port = settings.server_port
```

---

## Common Workflows

### Data Analysis Workflow

```python
from src.workflows.crew_system import create_ml_crew

# Create and run workflow
crew = create_ml_crew()
result = crew.kickoff({
    "data_path": "data/dataset.csv",
    "target_column": "target"
})
print(result)
```

### LangGraph State Workflow

```python
from src.workflows.langgraph_system import create_ml_workflow

# Create workflow
workflow = create_ml_workflow()

# Run workflow
result = workflow.invoke({
    "data_path": "data/dataset.csv",
    "target_column": "target",
    "messages": []
})
print(result)
```

### Custom Tool Usage

```python
from src.tools.data_tools import load_data, clean_data
from src.tools.ml_tools import train_model
from src.tools.analysis_tools import perform_eda

# Load and clean data
df = load_data("data/dataset.csv")
cleaned_df = clean_data(df)

# Perform EDA
eda_results = perform_eda(cleaned_df, "target")

# Train model
model_results = train_model(cleaned_df, "target", "classification")
```

---

## Troubleshooting Quick Fixes

### Issue: Module Not Found

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: OpenAI Authentication Error

```bash
# Check .env file exists
cat .env | grep OPENAI_API_KEY

# Verify key is loaded
python -c "from config.settings import get_settings; print(get_settings().openai_api_key[:10])"
```

### Issue: Port Already in Use

```bash
# Find and kill process
lsof -i :8000  # Get PID
kill -9 <PID>

# Or use different port
SERVER_PORT=8001 uvicorn src.ag_ui_server:app --port 8001
```

### Issue: Tests Failing

```bash
# Run tests with verbose output
pytest -v

# Run specific failing test
pytest tests/path/to/test.py::test_name -v

# Check test dependencies
pip install -r requirements-test.txt
```

### Issue: Import Errors in Tests

```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install in development mode
pip install -e .
```

---

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

```bash
# Edit code
vim src/your_module.py

# Run tests frequently
pytest -x  # Stop on first failure
```

### 3. Code Quality

```bash
# Format and lint
black src/ tests/
ruff check --fix src/ tests/

# Type check
mypy src/

# Run full test suite
pytest --cov=src
```

### 4. Commit

```bash
git add .
git commit -m "feat: add new feature"
```

### 5. Push and PR

```bash
git push origin feature/your-feature-name
# Create PR on GitHub
```

---

## Docker Quick Commands

### Build Image

```bash
docker build -t ml-ai-framework:latest .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  ml-ai-framework:latest
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

---

## Performance Optimization Tips

### 1. Use Faster Model for Development

```bash
OPENAI_MODEL=gpt-3.5-turbo python examples/simple_workflow.py
```

### 2. Reduce Agent Iterations

```bash
MAX_AGENT_ITERATIONS=5 python examples/simple_workflow.py
```

### 3. Enable Async Processing

```bash
ASYNC_ENABLED=true uvicorn src.ag_ui_server:app
```

### 4. Increase Workers

```bash
MAX_WORKERS=8 uvicorn src.ag_ui_server:app --workers 4
```

### 5. Use Caching (if implemented)

```python
# Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(param):
    # ... computation
    return result
```

---

## Monitoring & Debugging

### View Logs

```bash
# Systemd service
sudo journalctl -u ml-ai-framework -f

# Docker
docker logs -f <container-id>

# Application logs
tail -f logs/app.log
```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG uvicorn src.ag_ui_server:app --reload
```

### Monitor Resource Usage

```bash
# CPU and Memory
htop

# Disk usage
df -h

# Process specific
ps aux | grep uvicorn
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Execute workflow
curl -X POST http://localhost:8000/workflow/execute \
  -H "Content-Type: application/json" \
  -d @examples/request.json
```

---

## File Structure Reference

```
ml-ai-framework/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── data_tools.py       # Data loading/cleaning
│   │   ├── ml_tools.py         # Model training/evaluation
│   │   └── analysis_tools.py   # EDA and insights
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py          # Structured logging
│   │   └── error_handling.py   # Retry/circuit breaker
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── crew_system.py      # CrewAI implementation
│   │   └── langgraph_system.py # LangGraph implementation
│   └── ag_ui_server.py         # FastAPI server
├── config/
│   ├── __init__.py
│   └── settings.py             # Configuration management
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test data and mocks
├── examples/
│   ├── quickstart/             # Getting started examples
│   ├── iris_analysis/          # Classification example
│   ├── housing_regression/     # Regression example
│   └── advanced_workflow/      # Advanced patterns
├── docs/
│   ├── PRODUCTION_READINESS.md # Validation report
│   ├── DEPLOYMENT.md          # Deployment guide
│   ├── QUICK_REFERENCE.md     # This file
│   └── CONTRIBUTING.md        # Development guide
├── .env.example               # Environment template
├── requirements.txt           # Production dependencies
├── requirements-test.txt      # Test dependencies
├── pyproject.toml            # Project metadata
├── pytest.ini                # Pytest configuration
└── README.md                 # Project overview
```

---

## Tool Functions Reference

### Data Tools

```python
from src.tools.data_tools import load_data, clean_data, validate_dataset

# Load data
df = load_data("path/to/data.csv")

# Clean data
cleaned_df = clean_data(df, missing_threshold=0.5, outlier_method="iqr")

# Validate dataset
is_valid = validate_dataset(df, required_columns=["feature1", "target"])
```

### ML Tools

```python
from src.tools.ml_tools import train_model, evaluate_model, analyze_model

# Train model
results = train_model(
    df=df,
    target_column="target",
    problem_type="classification",  # or "regression"
    test_size=0.2
)

# Evaluate model
metrics = evaluate_model(
    model=results["model"],
    X_test=results["X_test"],
    y_test=results["y_test"],
    problem_type="classification"
)

# Analyze model
analysis = analyze_model(
    model=results["model"],
    feature_names=results["feature_names"]
)
```

### Analysis Tools

```python
from src.tools.analysis_tools import perform_eda, calculate_statistics, generate_insights

# Perform EDA
eda_results = perform_eda(df, target_column="target")

# Calculate statistics
stats = calculate_statistics(df)

# Generate insights
insights = generate_insights(
    data_summary=stats,
    eda_results=eda_results,
    model_results=model_results
)
```

---

## Keyboard Shortcuts (Development)

### pytest

- `Ctrl+C` - Stop test execution
- `pytest -x` - Stop on first failure
- `pytest -k "test_name"` - Run tests matching pattern
- `pytest --lf` - Run last failed tests
- `pytest --ff` - Run failures first

### uvicorn

- `Ctrl+C` - Stop server
- `--reload` - Auto-reload on file changes
- `--log-level debug` - Verbose logging

---

## Useful Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# ML-AI Framework aliases
alias mlai-activate='source venv/bin/activate'
alias mlai-server='uvicorn src.ag_ui_server:app --reload'
alias mlai-test='pytest --cov=src -v'
alias mlai-format='black src/ tests/ && ruff check --fix src/ tests/'
alias mlai-check='black --check src/ tests/ && ruff check src/ tests/ && mypy src/'
```

---

## Quick Testing

### Test a Single Workflow

```python
# test_quick.py
from src.workflows.crew_system import create_ml_crew

crew = create_ml_crew()
result = crew.kickoff({
    "data_path": "data/iris.csv",
    "target_column": "species"
})
print(result)
```

```bash
python test_quick.py
```

### Test API Endpoint

```bash
# Save request payload
cat > /tmp/test_request.json << EOF
{
  "data_path": "data/iris.csv",
  "target_column": "species",
  "workflow_type": "crewai"
}
EOF

# Test endpoint
curl -X POST http://localhost:8000/workflow/execute \
  -H "Content-Type: application/json" \
  -d @/tmp/test_request.json
```

---

## Common Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'structlog'` | Dependencies not installed | `pip install -r requirements.txt` |
| `openai.error.AuthenticationError` | Invalid/missing API key | Check `.env` file, verify `OPENAI_API_KEY` |
| `OSError: [Errno 98] Address already in use` | Port 8000 in use | Kill process or use different port |
| `ValidationError` | Invalid input data | Check request payload matches schema |
| `FileNotFoundError: data/dataset.csv` | Data file missing | Verify file path, check permissions |
| `pydantic.error_wrappers.ValidationError` | Config validation failed | Check environment variables in `.env` |

---

## Getting Help

### Documentation
- **Full README**: `/README.md`
- **Deployment**: `/docs/DEPLOYMENT.md`
- **Production Readiness**: `/docs/PRODUCTION_READINESS.md`
- **Contributing**: `/docs/CONTRIBUTING.md`

### Examples
- **Quickstart**: `examples/quickstart/README.md`
- **Use Cases**: `examples/iris_analysis/README.md`, `examples/housing_regression/README.md`

### Testing
- **Test README**: `tests/README.md`

### External Resources
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **CrewAI Docs**: https://docs.crewai.com/
- **LangGraph Docs**: https://python.langchain.com/docs/langgraph
- **Pydantic Docs**: https://docs.pydantic.dev/

---

## Version Information

- **Framework Version**: 0.1.0
- **Python Required**: 3.10+
- **Last Updated**: 2025-10-05

---

**Quick Tip**: Bookmark this page for fast reference during development and operations.
