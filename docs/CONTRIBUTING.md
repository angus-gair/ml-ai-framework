# Contributing Guide

**ML-AI Framework** - Thank you for your interest in contributing!

This guide will help you set up your development environment, understand our code standards, and contribute effectively to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Code Standards](#code-standards)
5. [Testing Requirements](#testing-requirements)
6. [Pull Request Process](#pull-request-process)
7. [Project Structure](#project-structure)
8. [Common Tasks](#common-tasks)
9. [Release Process](#release-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. We pledge to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Our Standards

**Positive Behavior**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive feedback
- Focusing on community benefit
- Showing empathy and kindness

**Unacceptable Behavior**:
- Harassment or discriminatory language
- Trolling or insulting comments
- Publishing others' private information
- Any conduct inappropriate in a professional setting

### Enforcement

Violations may be reported to the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.10+** installed
- **Git** for version control
- **GitHub account** for pull requests
- **OpenAI API key** for testing (can use test mode)
- Basic understanding of:
  - Python programming
  - Multi-agent systems (helpful but not required)
  - Testing with pytest
  - Git workflow

### Fork and Clone

```bash
# 1. Fork the repository on GitHub
# Click "Fork" button at https://github.com/your-org/ml-ai-framework

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ml-ai-framework.git
cd ml-ai-framework

# 3. Add upstream remote
git remote add upstream https://github.com/your-org/ml-ai-framework.git

# 4. Verify remotes
git remote -v
```

### Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description

# Or for documentation
git checkout -b docs/what-you-are-documenting
```

---

## Development Setup

### 1. Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2. Install Dependencies

```bash
# Install all dependencies (production + development)
pip install -r requirements.txt
pip install -r requirements-test.txt

# Or install in development mode
pip install -e .

# Verify installation
python -c "from src.workflows import crew_system; print('✓ Installation successful')"
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env  # or vim, code, etc.
```

**Minimal .env for development**:
```bash
OPENAI_API_KEY=sk-your-test-key-here
OPENAI_MODEL=gpt-3.5-turbo  # Cheaper for development
LOG_LEVEL=DEBUG
SERVER_RELOAD=true
```

### 4. Install Pre-commit Hooks (Optional but Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

**Create .pre-commit-config.yaml** (if not present):
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
```

### 5. Verify Setup

```bash
# Run tests to verify everything works
pytest tests/ -v

# Check code quality
black --check src/ tests/
ruff check src/ tests/
mypy src/

# Run example
python examples/quickstart/01_basic_workflow.py
```

---

## Code Standards

### Python Style Guide

We follow **PEP 8** with some customizations:

**Line Length**: 100 characters (enforced by Black)
**Indentation**: 4 spaces (no tabs)
**Quotes**: Double quotes preferred for strings
**Imports**: Organized by isort/ruff

### Code Formatting

We use **Black** for automatic code formatting:

```bash
# Format all code
black src/ tests/ examples/

# Check formatting without changes
black --check src/

# Format specific file
black src/workflows/crew_system.py
```

**Black configuration** (in pyproject.toml):
```toml
[tool.black]
line-length = 100
target-version = ['py310', 'py311', 'py312']
include = '\.pyi?$'
```

### Linting

We use **Ruff** for fast Python linting:

```bash
# Lint all code
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Check specific file
ruff check src/tools/data_tools.py
```

**Ruff configuration** (in pyproject.toml):
```toml
[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
ignore = ["E501"]
```

**Selected rule categories**:
- **E**: Pycodestyle errors
- **F**: Pyflakes (unused imports, variables)
- **I**: Import sorting
- **N**: Naming conventions
- **W**: Warnings
- **UP**: Python upgrade syntax
- **B**: Bugbear (common bugs)
- **C4**: Comprehensions
- **SIM**: Simplifications

### Type Hints

We use **Mypy** for static type checking:

```bash
# Type check all code
mypy src/

# Check specific file
mypy src/models/schemas.py
```

**Requirements**:
- All functions must have type hints for parameters and return values
- Use `typing` module for complex types
- Use `Optional[T]` for nullable values
- Use `List[T]`, `Dict[K, V]`, etc. for collections

**Example**:
```python
from typing import List, Optional, Dict, Any
import pandas as pd

def load_data(
    file_path: str,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        file_path: Path to CSV file
        columns: Optional list of columns to load

    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(file_path, usecols=columns)
```

### Documentation

**Docstring Style**: Google style

**Required for**:
- All public functions and classes
- Module-level docstrings
- Complex private functions

**Example**:
```python
def train_model(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str = "classification",
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Train machine learning model on provided dataset.

    This function performs data splitting, model selection based on
    problem type, training, and returns comprehensive results.

    Args:
        df: Input DataFrame containing features and target
        target_column: Name of the target column
        problem_type: Type of ML problem ('classification' or 'regression')
        test_size: Proportion of data to use for testing (0.0-1.0)

    Returns:
        Dictionary containing:
            - model: Trained model object
            - X_train: Training features
            - X_test: Testing features
            - y_train: Training target
            - y_test: Testing target
            - feature_names: List of feature names

    Raises:
        ValueError: If target_column not in DataFrame
        ValueError: If problem_type is invalid
        ValueError: If test_size is out of range

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> results = train_model(df, 'target', 'classification', 0.2)
        >>> model = results['model']
        >>> accuracy = model.score(results['X_test'], results['y_test'])
    """
    # Implementation
    pass
```

### Naming Conventions

**Variables and Functions**: `snake_case`
```python
user_name = "John"
def calculate_total(): pass
```

**Classes**: `PascalCase`
```python
class DataLoader: pass
class MLWorkflow: pass
```

**Constants**: `UPPER_SNAKE_CASE`
```python
MAX_RETRIES = 3
DEFAULT_MODEL = "gpt-4"
```

**Private members**: Prefix with single underscore
```python
_internal_cache = {}
def _helper_function(): pass
```

---

## Testing Requirements

### Test Coverage

**Minimum coverage**: 80% (enforced by pytest)
**Target coverage**: 90%+

### Test Organization

```
tests/
├── unit/               # Fast, isolated unit tests
│   ├── test_tools.py
│   ├── test_models.py
│   └── test_mocked_llm.py
├── integration/        # Tests with external dependencies
│   └── test_llm_judge.py
├── fixtures/           # Shared test data and mocks
│   └── sample_datasets.py
└── test_data/          # Static test data files
```

### Writing Tests

**Use pytest conventions**:
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

**Example unit test**:
```python
import pytest
import pandas as pd
from src.tools.data_tools import load_data, clean_data

class TestDataTools:
    """Unit tests for data tools."""

    def test_load_data_valid_file(self):
        """Test loading valid CSV file."""
        df = load_data("tests/test_data/sample.csv")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_load_data_missing_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent.csv")

    @pytest.mark.parametrize("threshold,expected_columns", [
        (0.5, 8),
        (0.9, 10),
    ])
    def test_clean_data_missing_threshold(self, threshold, expected_columns):
        """Test cleaning with different missing value thresholds."""
        df = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [None, None, None, None],
            'c': [1, 2, 3, 4]
        })
        cleaned = clean_data(df, missing_threshold=threshold)
        assert len(cleaned.columns) == expected_columns
```

**Example integration test**:
```python
import pytest
from src.workflows.crew_system import create_ml_crew

@pytest.mark.integration
@pytest.mark.slow
class TestCrewWorkflow:
    """Integration tests for CrewAI workflow."""

    def test_full_workflow_execution(self):
        """Test complete workflow execution."""
        crew = create_ml_crew()
        result = crew.kickoff({
            "data_path": "tests/test_data/iris.csv",
            "target_column": "species"
        })

        assert result is not None
        assert "model_metrics" in result
        assert "insights" in result
```

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit
def test_unit_function(): pass

@pytest.mark.integration
def test_integration_function(): pass

@pytest.mark.slow
def test_slow_function(): pass

@pytest.mark.llm
def test_llm_function(): pass

@pytest.mark.parametrize
def test_parameterized_function(): pass
```

**Run specific markers**:
```bash
pytest -m unit           # Only unit tests
pytest -m integration    # Only integration tests
pytest -m "not slow"     # Skip slow tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_tools.py

# Run specific test
pytest tests/unit/test_tools.py::test_load_data

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run last failed tests
pytest --lf
```

### Test Fixtures

**Create reusable fixtures**:
```python
import pytest
import pandas as pd

@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [{
            "message": {"content": "Mocked response"}
        }]
    }

# Use in tests
def test_with_fixture(sample_dataframe):
    assert len(sample_dataframe) == 5
```

---

## Pull Request Process

### Before Submitting

**1. Update your branch**:
```bash
git checkout main
git pull upstream main
git checkout feature/your-feature
git rebase main
```

**2. Run all quality checks**:
```bash
# Format code
black src/ tests/

# Lint code
ruff check --fix src/ tests/

# Type check
mypy src/

# Run tests with coverage
pytest --cov=src --cov-fail-under=80
```

**3. Commit your changes**:
```bash
git add .
git commit -m "feat: add new feature"
```

**Commit message format**:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### Submitting PR

**1. Push to your fork**:
```bash
git push origin feature/your-feature
```

**2. Create Pull Request on GitHub**:
- Go to your fork on GitHub
- Click "New Pull Request"
- Select your feature branch
- Fill in PR template

**PR Template**:
```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added for new features
```

### Review Process

**What reviewers look for**:
1. Code quality and style compliance
2. Test coverage and quality
3. Documentation completeness
4. Breaking changes
5. Performance impact

**Addressing feedback**:
```bash
# Make requested changes
git add .
git commit -m "address review feedback"
git push origin feature/your-feature
```

**Squashing commits** (if requested):
```bash
git rebase -i HEAD~3  # Squash last 3 commits
git push origin feature/your-feature --force
```

---

## Project Structure

### Source Code Organization

```
src/
├── models/
│   ├── __init__.py
│   └── schemas.py          # Pydantic models and validation
├── tools/
│   ├── __init__.py
│   ├── data_tools.py       # Data loading and processing
│   ├── ml_tools.py         # Model training and evaluation
│   └── analysis_tools.py   # EDA and insights generation
├── utils/
│   ├── __init__.py
│   ├── logging.py          # Structured logging setup
│   └── error_handling.py   # Retry logic and circuit breakers
├── workflows/
│   ├── __init__.py
│   ├── crew_system.py      # CrewAI workflow implementation
│   └── langgraph_system.py # LangGraph workflow implementation
└── ag_ui_server.py         # FastAPI server with streaming
```

### Adding New Components

**New Tool**:
```python
# src/tools/new_tool.py
from typing import Any, Dict
import pandas as pd

def new_tool_function(df: pd.DataFrame, param: str) -> Dict[str, Any]:
    """
    Description of what this tool does.

    Args:
        df: Input DataFrame
        param: Tool parameter

    Returns:
        Results dictionary
    """
    # Implementation
    return {"result": "value"}

# Register in src/tools/__init__.py
from .new_tool import new_tool_function

__all__ = ["new_tool_function"]
```

**New Workflow**:
```python
# src/workflows/new_workflow.py
from typing import Dict, Any

def create_new_workflow() -> Any:
    """
    Create and configure new workflow.

    Returns:
        Configured workflow object
    """
    # Implementation
    pass

# Register in src/workflows/__init__.py
from .new_workflow import create_new_workflow

__all__ = ["create_new_workflow"]
```

---

## Common Tasks

### Adding a New Feature

```bash
# 1. Create feature branch
git checkout -b feature/my-new-feature

# 2. Implement feature
# - Add code to appropriate module
# - Add type hints
# - Add docstrings

# 3. Write tests
# - Create test file in tests/unit/ or tests/integration/
# - Aim for 90%+ coverage of new code

# 4. Update documentation
# - Update relevant README files
# - Add examples if applicable

# 5. Run quality checks
black src/ tests/
ruff check --fix src/ tests/
mypy src/
pytest --cov=src

# 6. Commit and push
git add .
git commit -m "feat: add new feature description"
git push origin feature/my-new-feature
```

### Fixing a Bug

```bash
# 1. Create fix branch
git checkout -b fix/bug-description

# 2. Write failing test that reproduces bug
# tests/unit/test_bug.py

# 3. Fix the bug

# 4. Verify test now passes
pytest tests/unit/test_bug.py

# 5. Run full test suite
pytest --cov=src

# 6. Commit and push
git add .
git commit -m "fix: resolve bug description"
git push origin fix/bug-description
```

### Updating Documentation

```bash
# 1. Create docs branch
git checkout -b docs/what-you-are-documenting

# 2. Update documentation files
# - README.md
# - docs/*.md
# - Docstrings

# 3. Verify formatting
# Check markdown links work

# 4. Commit and push
git add .
git commit -m "docs: update documentation for X"
git push origin docs/what-you-are-documenting
```

### Updating Dependencies

```bash
# 1. Check for outdated packages
pip list --outdated

# 2. Update specific package
pip install --upgrade package-name

# 3. Update requirements file
pip freeze > requirements.txt

# 4. Test thoroughly
pytest --cov=src

# 5. Commit changes
git add requirements.txt
git commit -m "chore: update package-name to version X.Y.Z"
```

---

## Release Process

### Version Numbering

We follow **Semantic Versioning** (SemVer):
- **MAJOR**: Breaking changes (1.0.0 → 2.0.0)
- **MINOR**: New features, backwards compatible (1.0.0 → 1.1.0)
- **PATCH**: Bug fixes, backwards compatible (1.0.0 → 1.0.1)

### Creating a Release

**Maintainers only**:

```bash
# 1. Update version in pyproject.toml
# version = "0.2.0"

# 2. Update CHANGELOG.md
# Add release notes

# 3. Commit version bump
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"

# 4. Create tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# 5. Push changes and tags
git push origin main
git push origin v0.2.0

# 6. Create GitHub release
# Use GitHub UI to create release from tag
```

---

## Getting Help

### Resources

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: Search existing issues on GitHub
- **Discussions**: Use GitHub Discussions for questions

### Asking Questions

**Before asking**:
1. Search existing issues and discussions
2. Read relevant documentation
3. Try to reproduce with minimal example

**When asking**:
- Provide clear description
- Include code snippets
- Share error messages
- Specify environment (OS, Python version, etc.)

### Reporting Bugs

**Bug report template**:
```markdown
**Describe the bug**
Clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run '...'
2. With input '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.12]
- Framework version: [e.g., 0.1.0]

**Additional context**
Any other context about the problem.
```

---

## Best Practices

### Code Quality

1. **Write self-documenting code** - Clear variable/function names
2. **Keep functions small** - Single responsibility principle
3. **Avoid magic numbers** - Use named constants
4. **Handle errors gracefully** - Use proper exception handling
5. **Log appropriately** - Use structured logging

### Testing

1. **Test behavior, not implementation** - Focus on outcomes
2. **Use descriptive test names** - Clear what is being tested
3. **Arrange-Act-Assert pattern** - Structure tests clearly
4. **Mock external dependencies** - Keep tests fast and isolated
5. **Test edge cases** - Don't just test happy path

### Git Workflow

1. **Small, focused commits** - One logical change per commit
2. **Clear commit messages** - Explain why, not what
3. **Keep branches up to date** - Rebase regularly
4. **Don't commit generated files** - Use .gitignore
5. **Review your own code first** - Self-review before PR

---

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

## Thank You!

Your contributions make this project better. We appreciate your time and effort!

**Questions?** Open an issue or start a discussion on GitHub.

---

**Last Updated**: 2025-10-05
**Version**: 1.0
