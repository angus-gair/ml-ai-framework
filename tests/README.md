# ML-AI Framework Test Suite

Comprehensive test suite for the multi-agent AI data analysis framework.

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures and configuration
├── unit/                                # Unit tests
│   ├── test_models.py                  # Pydantic model validation
│   ├── test_tools.py                   # Tool function tests
│   └── test_mocked_llm.py              # Mocked LLM tests
├── integration/                         # Integration tests
│   ├── test_crew_workflow.py           # CrewAI workflow
│   ├── test_langgraph_workflow.py      # LangGraph workflow
│   ├── test_parameterized.py           # Parameterized scenarios
│   ├── test_llm_judge.py               # LLM-as-judge evaluation
│   └── test_performance.py             # Performance tests
├── fixtures/                            # Test data generators
│   └── sample_datasets.py
└── test_data/                           # Sample datasets
    └── README.md
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_models.py

# Specific test function
pytest tests/unit/test_models.py::TestDatasetMetadata::test_valid_dataset_metadata
```

### Run Tests by Marker
```bash
# Performance tests
pytest -m performance

# LLM-related tests
pytest -m llm

# Parameterized tests
pytest -m parametrize
```

### Coverage Report
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View coverage in terminal
pytest --cov=src --cov-report=term-missing
```

## Test Categories

### Unit Tests (tests/unit/)

**test_models.py** - Pydantic Model Validation
- DatasetMetadata validation
- CleaningReport validation with custom validators
- CorrelationPair boundary testing
- EDAFindings structure validation
- ModelPerformance metric validation
- AnalysisReport composite validation
- Edge cases: empty strings, negative values, out-of-bounds

**test_tools.py** - Tool Function Testing
- load_dataset with various CSV formats
- clean_data with missing values, duplicates, outliers
- calculate_statistics for descriptive stats and correlations
- train_model with classification tasks
- Error handling and edge cases
- Complete tool pipeline integration

**test_mocked_llm.py** - Mocked LLM Testing
- OpenAI completion mocking
- Anthropic Claude mocking
- Pydantic AI TestModel usage
- Deterministic test outputs
- Agent behavior with mocked responses
- Tool call mocking
- Cost tracking

### Integration Tests (tests/integration/)

**test_crew_workflow.py** - CrewAI Integration
- Crew initialization with all agents
- Sequential task execution
- Context passing between tasks
- Pydantic output validation
- Memory integration (short-term, long-term, entity)
- Tool invocation
- End-to-end Iris analysis

**test_langgraph_workflow.py** - LangGraph Integration
- State management with TypedDict
- State reducers and immutability
- Node routing logic
- Supervisor coordination
- Command pattern
- Checkpointing and recovery
- Error handling
- Complete workflow execution

**test_parameterized.py** - Parameterized Scenarios
- Dataset size variations (100-10000 rows)
- Different data types (numeric, mixed, categorical)
- Duplicate handling (0-10 duplicates)
- Outlier detection (varying magnitudes)
- Model training scenarios
- Validation edge cases
- End-to-end workflows

**test_llm_judge.py** - LLM-as-Judge Evaluation
- Evaluation rubric definition
- LLM judge scoring
- Output quality metrics
- Report completeness checking
- Batch evaluation
- Regression prevention
- Multi-judge consensus
- Continuous evaluation pipeline

**test_performance.py** - Performance Testing
- Data loading performance (<1s for 10k rows)
- Cleaning performance (<2s)
- Statistics calculation (<1.5s)
- Model training (<10s)
- End-to-end workflow (<30s)
- Memory usage monitoring
- Scalability with data size and features
- Concurrent execution
- Throughput testing (>10 ops/sec)

## Test Fixtures

### Core Fixtures (conftest.py)

**Dataset Fixtures**
- `sample_dataset` - Clean 100-row dataset
- `dataset_with_missing_values` - Dataset with NaN values
- `dataset_with_duplicates` - Dataset with duplicate rows
- `dataset_with_outliers` - Dataset with statistical outliers
- `iris_dataset_path` - Iris CSV file path
- `large_dataset` - 10k-row performance test dataset

**Mock Fixtures**
- `mock_llm` - Mocked LLM instance
- `mock_openai_client` - Mocked OpenAI client
- `mock_pydantic_model` - Mock Pydantic validation
- `mock_crew_agent` - Mock CrewAI agent
- `mock_crew_task` - Mock CrewAI task

**Data Fixtures**
- `dataset_metadata_dict` - Sample metadata
- `cleaning_report_dict` - Sample cleaning report
- `eda_findings_dict` - Sample EDA findings
- `model_performance_dict` - Sample model metrics

**Utility Fixtures**
- `temp_dir` - Temporary directory for test files
- `performance_threshold` - Performance time limits
- `llm_judge_criteria` - Evaluation criteria

## Writing New Tests

### Test Template
```python
import pytest

class TestMyFeature:
    """Test suite for my feature."""

    def test_basic_functionality(self):
        """Test basic feature behavior."""
        result = my_function()
        assert result is not None

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6)
    ])
    def test_parameterized(self, input, expected):
        """Test with multiple inputs."""
        assert my_function(input) == expected

    def test_with_fixture(self, sample_dataset):
        """Test using a fixture."""
        result = process(sample_dataset)
        assert len(result) > 0
```

### Best Practices

1. **One Assertion Per Test** - Each test should verify one behavior
2. **Descriptive Names** - Test names should explain what and why
3. **Arrange-Act-Assert** - Structure tests clearly
4. **Use Fixtures** - Reuse test data and setup
5. **Mock External Deps** - Keep tests isolated
6. **Test Edge Cases** - Boundary values, errors, empty inputs
7. **Performance Tests** - Mark slow tests with `@pytest.mark.slow`

## Test Coverage Goals

- **Unit Tests**: >90% coverage
- **Integration Tests**: All workflows covered
- **Edge Cases**: Comprehensive boundary testing
- **Performance**: All operations benchmarked
- **LLM Mocking**: All LLM calls tested deterministically

## Continuous Integration

Tests run automatically on:
- Every commit (unit tests)
- Pull requests (full suite)
- Nightly (performance regression)

## Troubleshooting

**Tests Fail Randomly**
- Check for LLM mocking - ensure deterministic outputs
- Verify no file system state dependencies
- Use fixtures for test isolation

**Performance Tests Fail**
- Check system load during test run
- Adjust thresholds in `performance_threshold` fixture
- Run performance tests separately: `pytest -m performance`

**Import Errors**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python path includes project root
- Verify virtual environment activated

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pydantic Testing Guide](https://docs.pydantic.dev/latest/concepts/testing/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
