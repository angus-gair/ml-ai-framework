# Iris Classification Example

This example demonstrates complete ML workflows using the ML-AI framework on the classic Iris dataset.

## Dataset Overview

**Iris Dataset:**
- **Samples:** 150
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Target:** species (setosa, versicolor, virginica)
- **Type:** Multi-class classification
- **Quality:** Clean, balanced, no missing values

## Examples Included

### 1. CrewAI Workflow (`run_crewai.py`)

Multi-agent workflow with 5 specialized agents:

**Agents:**
1. **Data Loading Specialist** - Loads and validates datasets
2. **Data Cleaning Specialist** - Handles data quality issues
3. **EDA Expert** - Performs statistical analysis
4. **Machine Learning Engineer** - Trains and tunes models
5. **Report Analyst** - Generates comprehensive reports

**Process:** Sequential task execution with agent collaboration

**Run:**
```bash
cd /home/thunder/projects/ml-ai-framework
python examples/iris_analysis/run_crewai.py
```

**Expected Output:**
- Complete workflow execution logs
- Agent task completion messages
- Final analysis report
- Results saved to `crewai_results.txt`

### 2. LangGraph Workflow (`run_langgraph.py`)

State-based workflow with graph execution:

**Graph Structure:**
```
START → load_data → clean_data → perform_eda → train_model → generate_report → END
```

**State Management:**
- Shared state across all nodes
- Automatic state updates
- Error tracking and recovery
- Message history

**Run:**
```bash
cd /home/thunder/projects/ml-ai-framework
python examples/iris_analysis/run_langgraph.py
```

**Expected Output:**
- State transition logs
- Node execution messages
- Comprehensive JSON report
- Results saved to `langgraph_results.json`

## Comparison: CrewAI vs LangGraph

| Aspect | CrewAI | LangGraph |
|--------|--------|-----------|
| **Architecture** | Agent-based | State-graph based |
| **Execution** | Sequential/Hierarchical | Graph-based flow |
| **State Management** | Agent context | Shared state object |
| **Coordination** | Task dependencies | Graph edges |
| **Flexibility** | High (agent roles) | High (graph structure) |
| **Debugging** | Agent logs | State snapshots |

## Expected Results

### Model Performance

Both workflows should achieve similar performance:

- **Accuracy:** ~95-97% on validation set
- **Precision/Recall/F1:** ~0.95+ (weighted average)
- **Training Time:** <5 seconds
- **Cross-validation:** 5-fold CV with ~95% average score

### Analysis Insights

**Feature Importance:**
1. Petal length (highest)
2. Petal width
3. Sepal length
4. Sepal width (lowest)

**Class Separability:**
- Setosa: Highly separable
- Versicolor & Virginica: Some overlap

**Correlations:**
- Strong: petal length ↔ petal width (>0.9)
- Moderate: sepal length ↔ petal length (~0.87)

## Files Generated

After running the examples:

```
iris_analysis/
├── run_crewai.py           # CrewAI workflow script
├── run_langgraph.py        # LangGraph workflow script
├── README.md               # This file
├── crewai_results.txt      # CrewAI execution results
└── langgraph_results.json  # LangGraph execution results
```

## Customization

### Modify Model Algorithm

```python
# In run_crewai.py or run_langgraph.py
workflow.run(
    data_path=dataset_path,
    target_column="species",
    # Add custom parameters (requires workflow modification)
)
```

### Change Hyperparameters

Edit the workflow files to pass custom hyperparameters to `train_model()`:

```python
hyperparameters = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": 42
}
```

### Try Different Algorithms

Supported algorithms:
- `random_forest` (default)
- `logistic_regression`

## Troubleshooting

### Issue: Import errors

**Solution:** Ensure you're running from the project root:
```bash
cd /home/thunder/projects/ml-ai-framework
python examples/iris_analysis/run_crewai.py
```

### Issue: Missing dependencies

**Solution:** Install requirements:
```bash
pip install -r /home/thunder/projects/ml-ai-framework/requirements.txt
```

### Issue: OpenAI API errors

**Solution:** Set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Next Steps

1. **Try Housing Regression:** See `../housing_regression/` for regression examples
2. **Advanced Patterns:** Check `../advanced_workflow/` for custom agents
3. **Create Custom Workflows:** Use these as templates for your own datasets

## Learning Objectives

After running these examples, you should understand:

1. How to prepare datasets for the framework
2. Differences between CrewAI and LangGraph approaches
3. Complete ML pipeline workflow steps
4. How agents coordinate to solve ML tasks
5. Model training and evaluation patterns
6. Error handling and logging best practices

## Performance Benchmarks

**Typical Execution Times:**
- CrewAI Workflow: 30-60 seconds
- LangGraph Workflow: 20-40 seconds

**Resource Usage:**
- Memory: <500 MB
- CPU: Moderate (depends on cross-validation)
- Disk: <1 MB for results

## References

- [Iris Dataset Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [CrewAI Documentation](https://docs.crewai.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
