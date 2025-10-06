# Quick Start Tutorial

Learn the ML-AI framework in 4 progressive examples. Each example builds on the previous one.

## Learning Path

```
01_basic_workflow.py      → Learn the fundamentals
         ↓
02_custom_tools.py        → Extend with custom tools
         ↓
03_error_handling.py      → Build robust pipelines
         ↓
04_testing.py             → Test your workflows
```

## Examples

### 01: Basic Workflow (Start Here!)

**Time:** 5 minutes
**Difficulty:** Beginner

The simplest possible ML workflow - load data, train model, get results.

**What you'll learn:**
- Loading datasets
- Running complete workflows
- Interpreting results

**Run:**
```bash
cd /home/thunder/projects/ml-ai-framework
python examples/quickstart/01_basic_workflow.py
```

**Expected Output:**
```
Step 1: Loading Iris dataset...
✓ Saved to: /tmp/quickstart_iris.csv
  Shape: (150, 5)

Step 2: Creating LangGraph workflow...
✓ Workflow initialized

Step 3: Running ML pipeline...
  (This will load → clean → analyze → train → report)

Step 4: Results
============================================================
✓ Status: success
⏱  Time: 45.23s

Model: random_forest
Accuracy: 96.67%

That's it! You just ran a complete ML pipeline.
```

**Key Concepts:**
- Workflows abstract the complexity
- Three simple steps: data → workflow → results
- LangGraph manages state automatically

---

### 02: Custom Tools

**Time:** 10 minutes
**Difficulty:** Intermediate

Learn to create and integrate custom tools into workflows.

**What you'll learn:**
- Creating custom data processing tools
- Feature engineering functions
- Outlier detection methods
- Feature importance analysis

**Run:**
```bash
python examples/quickstart/02_custom_tools.py
```

**Expected Output:**
```
Tool 1: Creating Polynomial Features
------------------------------------------------------------
Original features: 3
After polynomial: 6
New features: ['MedInc_power_2', 'HouseAge_power_2', 'AveRooms_power_2']

Tool 2: Detecting Outliers (IQR Method)
------------------------------------------------------------
MedInc:
  Outliers: 324 (1.57%)
  Bounds: -3.12 to 11.73

Tool 3: Feature Importance Analysis
------------------------------------------------------------
Top 5 Most Important Features:
  1. MedInc          ████████████████████ 0.5234
  2. AveOccup        ████████ 0.1567
  3. Latitude        ██████ 0.1234
```

**Key Concepts:**
- Tools are just Python functions
- Type hints improve clarity
- Return structured data (dicts, DataFrames)
- Include logging for debugging

---

### 03: Error Handling

**Time:** 15 minutes
**Difficulty:** Intermediate

Build robust ML pipelines with comprehensive error handling.

**What you'll learn:**
- Try-except patterns
- Retry with exponential backoff
- Circuit breaker pattern
- Graceful degradation
- Input validation

**Run:**
```bash
python examples/quickstart/03_error_handling.py
```

**Expected Output:**
```
Pattern 1: Basic Try-Except
------------------------------------------------------------
Loading nonexistent file: error
  Error: File not found

Pattern 2: Retry with Backoff
------------------------------------------------------------
  Attempt 1...
  Attempt 2...
  Attempt 3...
Result: Success!

Pattern 3: Circuit Breaker
------------------------------------------------------------
  Failure 1 - State: CLOSED
  Failure 2 - State: CLOSED
  Failure 3 - State: OPEN
  Request rejected: Circuit breaker OPEN - service unavailable...
```

**Key Patterns:**

**1. Retry Decorator:**
```python
@retry(max_attempts=3, delay_seconds=1.0)
def unstable_function():
    # Automatically retries on failure
    pass
```

**2. Circuit Breaker:**
```python
breaker = CircuitBreaker(failure_threshold=5)
breaker.call(risky_function)
```

**3. Graceful Degradation:**
```python
try:
    return complex_expensive_model()
except:
    return simple_fast_model()  # Fallback
```

---

### 04: Testing

**Time:** 15 minutes
**Difficulty:** Advanced

Learn testing patterns for ML workflows.

**What you'll learn:**
- Unit testing ML functions
- Integration testing workflows
- Mocking dependencies
- Property-based testing
- Performance benchmarks

**Run:**
```bash
python examples/quickstart/04_testing.py
```

**Expected Output:**
```
Running: Unit Test: Data Loading
✓ test_data_loading passed

Running: Unit Test: Data Cleaning
✓ test_data_cleaning passed

Running: Mock Test: External Dependencies
✓ test_workflow_with_mock passed

Running: Integration Test: Complete Workflow
✓ test_complete_workflow passed

============================================================
Test Results: 8 passed, 0 failed
```

**Testing Patterns:**

**1. Unit Test:**
```python
def test_data_loading():
    df, metadata = load_dataset(test_file)
    assert len(df) == expected_rows
    assert metadata.total_columns == expected_cols
```

**2. Mock Test:**
```python
with patch('src.tools.ml_tools.train_model') as mock:
    mock.return_value = (mock_model, mock_performance)
    result = workflow.run(data_path, target)
    assert mock.called
```

**3. Property Test:**
```python
def test_invariants():
    for _ in range(100):
        df_clean, report = clean_data(random_df())
        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) <= len(random_df())
```

## Progression Guide

### After Completing Quick Start

1. **Try Iris Analysis** (`../iris_analysis/`)
   - See full workflows in action
   - Compare CrewAI vs LangGraph
   - Understand agent coordination

2. **Explore Housing Regression** (`../housing_regression/`)
   - Work with larger datasets
   - Learn regression patterns
   - Handle real-world data issues

3. **Advanced Workflows** (`../advanced_workflow/`)
   - Create custom agents
   - Parallel execution
   - Streaming progress

## Common Patterns Reference

### Load and Prepare Data

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

df.to_csv('/tmp/data.csv', index=False)
```

### Run Workflow

```python
from src.workflows.langgraph_system import LangGraphWorkflow

workflow = LangGraphWorkflow(model_name="gpt-4")
result = workflow.run(
    data_path="/tmp/data.csv",
    target_column="target"
)
```

### Access Results

```python
if result["status"] == "success":
    report = result["report"]
    perf = report["model_performance"]

    print(f"Algorithm: {perf['algorithm']}")
    print(f"Accuracy: {perf['validation_score']:.2%}")
```

### Create Custom Tool

```python
from structlog import get_logger

logger = get_logger(__name__)

def my_custom_tool(df: pd.DataFrame) -> pd.DataFrame:
    """Custom data transformation tool."""
    logger.info("running_custom_tool", rows=len(df))

    # Your transformation logic
    df_transformed = df.copy()
    # ... transformations ...

    logger.info("custom_tool_completed")
    return df_transformed
```

### Error Handling Pattern

```python
try:
    result = risky_operation()
    logger.info("operation_succeeded")
    return result

except SpecificError as e:
    logger.error("specific_error", error=str(e))
    return fallback_value

except Exception as e:
    logger.error("unexpected_error", error=str(e))
    raise
```

## Tips for Success

### 1. Start Simple
- Run 01_basic_workflow.py first
- Understand each step before moving on
- Don't skip examples

### 2. Read the Code
- Each example is heavily commented
- Understand patterns, not just syntax
- Try modifying examples

### 3. Experiment
- Change parameters
- Try different datasets
- Break things intentionally

### 4. Use Logging
- Enable verbose logging to see what's happening
- Check logs when things fail
- Logs are in `/logs/` directory

### 5. Test Everything
- Write tests as you build
- Use the patterns from 04_testing.py
- Tests prevent future bugs

## Troubleshooting

### Issue: Import errors

**Solution:**
```bash
# Run from project root
cd /home/thunder/projects/ml-ai-framework
python examples/quickstart/01_basic_workflow.py
```

### Issue: OpenAI API errors

**Solution:**
```bash
export OPENAI_API_KEY="your-key-here"
```

Or use a different LLM provider:
```python
workflow = LangGraphWorkflow(model_name="gpt-3.5-turbo")
```

### Issue: Slow execution

**Reasons:**
- LLM API calls take time
- Large datasets
- Complex models

**Solutions:**
- Use smaller datasets for testing
- Cache results when possible
- Use faster models for development

### Issue: Tests failing

**Common causes:**
- Missing dependencies
- API rate limits
- Stale data

**Solutions:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check logs
tail -f logs/ml-ai-framework.log

# Run single test
python -c "from examples.quickstart.04_testing import test_data_loading; test_data_loading()"
```

## Next Steps

### Immediate Next Steps

1. ✅ Complete all 4 quick start examples
2. 📊 Try iris_analysis examples
3. 🏠 Explore housing_regression examples
4. 🚀 Build your own workflow

### Learning Resources

**Framework Documentation:**
- `/home/thunder/projects/ml-ai-framework/README.md` - Project overview
- `/home/thunder/projects/ml-ai-framework/docs/` - Detailed guides

**Example Projects:**
- `iris_analysis/` - Classification workflows
- `housing_regression/` - Regression workflows
- `advanced_workflow/` - Advanced patterns

**External Resources:**
- [CrewAI Docs](https://docs.crewai.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/)

## Feedback

Found an issue or have suggestions?
- Check existing examples for similar patterns
- Review framework documentation
- Experiment with modifications

## Summary

**You've Learned:**
- ✅ Basic workflow execution
- ✅ Creating custom tools
- ✅ Robust error handling
- ✅ Testing ML pipelines

**You Can Now:**
- Run complete ML workflows
- Extend the framework with custom tools
- Build production-ready pipelines
- Test your ML code properly

**Next Challenge:**
Build your own ML workflow using these patterns!

---

**Happy Learning! 🚀**
