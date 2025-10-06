"""
Pytest configuration and shared fixtures for ML-AI framework tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, List
import tempfile
import shutil
import os


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_dataset():
    """Generate sample dataset for testing."""
    np.random.seed(42)
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randint(1, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def dataset_with_missing_values():
    """Dataset with missing values for testing cleaning logic."""
    np.random.seed(42)
    df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        'col2': [np.nan, 'B', 'C', np.nan, 'E', 'F', np.nan, 'H', 'I', 'J'],
        'col3': [1.1, 2.2, 3.3, np.nan, 5.5, 6.6, 7.7, np.nan, 9.9, 10.0],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    return df


@pytest.fixture
def dataset_with_duplicates():
    """Dataset with duplicate rows."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 1, 2],
        'b': [4, 5, 6, 4, 5],
        'c': [7, 8, 9, 7, 8]
    })
    return df


@pytest.fixture
def dataset_with_outliers():
    """Dataset with statistical outliers."""
    np.random.seed(42)
    normal_data = np.random.randn(95)
    outliers = np.array([10, -10, 15, -15, 20])
    combined = np.concatenate([normal_data, outliers])

    return pd.DataFrame({
        'values': combined,
        'target': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def iris_dataset_path(temp_dir, sample_dataset):
    """Create Iris-like dataset CSV file."""
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    file_path = temp_dir / "iris_dataset.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def mock_llm():
    """Mock LLM for deterministic testing."""
    llm = Mock()
    llm.generate = Mock(return_value="Mocked LLM response")
    llm.invoke = Mock(return_value=MagicMock(content="Mocked analysis result"))
    llm.model_name = "gpt-4o-mini-mock"
    return llm


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for API call testing."""
    client = Mock()

    # Mock chat completion
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mocked completion"
    mock_response.usage.total_tokens = 100
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 50

    client.chat.completions.create = Mock(return_value=mock_response)
    return client


@pytest.fixture
def mock_pydantic_model():
    """Mock Pydantic model validation."""
    from pydantic import BaseModel

    class MockModel(BaseModel):
        value: str = "mock_value"
        score: float = 0.95

    return MockModel


@pytest.fixture
def dataset_metadata_dict():
    """Sample dataset metadata dictionary."""
    return {
        "dataset_id": "test-123",
        "name": "test_dataset.csv",
        "source": "/path/to/test_dataset.csv",
        "rows": 100,
        "columns": 5,
        "dtypes": {
            "feature1": "float64",
            "feature2": "float64",
            "feature3": "int64",
            "category": "object",
            "target": "int64"
        },
        "missing_values": {
            "feature1": 0,
            "feature2": 2,
            "feature3": 0,
            "category": 1,
            "target": 0
        }
    }


@pytest.fixture
def cleaning_report_dict():
    """Sample cleaning report data."""
    return {
        "rows_before": 100,
        "rows_after": 95,
        "duplicates_removed": 5,
        "missing_values_handled": {
            "feature1": "median",
            "category": "mode"
        },
        "outliers_detected": {
            "feature1": 3,
            "feature2": 2
        },
        "feature_engineering": ["log_transform_feature1"],
        "cleaned_data_path": "/tmp/cleaned_data.csv"
    }


@pytest.fixture
def eda_findings_dict():
    """Sample EDA findings data."""
    return {
        "summary_statistics": {
            "feature1": {"mean": 0.5, "std": 1.2, "min": -2.0, "max": 3.5}
        },
        "strong_correlations": [
            {
                "feature_a": "feature1",
                "feature_b": "feature2",
                "correlation": 0.85,
                "significance": "strong"
            }
        ],
        "outliers_by_column": {"feature1": 3, "feature2": 2},
        "distribution_insights": [
            "feature1 is normally distributed",
            "feature2 shows right skewness"
        ],
        "recommended_features": ["feature1", "feature2", "feature3"],
        "visualizations_created": ["correlation_matrix.png", "distributions.png"]
    }


@pytest.fixture
def model_performance_dict():
    """Sample model performance metrics."""
    return {
        "algorithm": "RandomForestClassifier",
        "accuracy": 0.92,
        "precision": 0.90,
        "recall": 0.88,
        "f1_score": 0.89,
        "feature_importance": {
            "feature1": 0.35,
            "feature2": 0.30,
            "feature3": 0.25,
            "category": 0.10
        },
        "confusion_matrix": [[40, 5], [3, 42]],
        "model_path": "/models/trained_model.pkl",
        "training_time_seconds": 12.5
    }


@pytest.fixture
def mock_crew_agent():
    """Mock CrewAI agent."""
    agent = Mock()
    agent.role = "Test Agent"
    agent.goal = "Test goal"
    agent.backstory = "Test backstory"
    agent.verbose = True
    return agent


@pytest.fixture
def mock_crew_task():
    """Mock CrewAI task."""
    task = Mock()
    task.description = "Test task description"
    task.expected_output = "Test expected output"
    task.output = Mock()
    task.output.raw = "Task output"
    task.output.pydantic = None
    return task


# Performance testing fixtures
@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    np.random.seed(42)
    n_rows = 10000
    return pd.DataFrame({
        'feature1': np.random.randn(n_rows),
        'feature2': np.random.randn(n_rows),
        'feature3': np.random.randint(1, 100, n_rows),
        'feature4': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'target': np.random.randint(0, 2, n_rows)
    })


@pytest.fixture
def performance_threshold():
    """Performance thresholds for various operations."""
    return {
        'data_load': 1.0,  # seconds
        'data_clean': 2.0,
        'statistics': 1.5,
        'model_train': 10.0,
        'end_to_end': 30.0
    }


# LLM-as-judge fixtures
@pytest.fixture
def llm_judge_criteria():
    """Evaluation criteria for LLM-as-judge tests."""
    return {
        'accuracy': 'Does the output accurately reflect the data analysis?',
        'completeness': 'Are all required sections present?',
        'clarity': 'Is the output clear and well-structured?',
        'actionability': 'Does it provide actionable recommendations?'
    }


@pytest.fixture
def mock_langsmith_client():
    """Mock LangSmith client for tracing tests."""
    client = Mock()
    client.create_run = Mock(return_value={'id': 'test-run-123'})
    client.update_run = Mock()
    return client


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key-123')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key-456')
    monkeypatch.setenv('TESTING', 'true')


# Cleanup fixtures
@pytest.fixture
def cleanup_files():
    """Track and cleanup temporary files created during tests."""
    created_files = []

    def _add_file(filepath):
        created_files.append(filepath)

    yield _add_file

    # Cleanup
    for filepath in created_files:
        if os.path.exists(filepath):
            os.remove(filepath)
