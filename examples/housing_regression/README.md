# California Housing Regression Example

This example demonstrates regression workflows using the ML-AI framework on the California Housing dataset.

## Dataset Overview

**California Housing Dataset (1990 Census):**
- **Samples:** 20,640 California districts
- **Features:** 8 numerical features
- **Target:** Median house value (in $100,000s)
- **Type:** Regression
- **Challenges:** Outliers, geographical patterns, economic factors

### Features

1. **MedInc:** Median income in block group
2. **HouseAge:** Median house age in block group
3. **AveRooms:** Average number of rooms per household
4. **AveBedrms:** Average number of bedrooms per household
5. **Population:** Block group population
6. **AveOccup:** Average number of household members
7. **Latitude:** Block group latitude (geographical)
8. **Longitude:** Block group longitude (geographical)

### Target

**MedHouseVal:** Median house value for California districts (expressed in $100,000s)
- Range: $14,999 - $500,001
- Mean: ~$206,000
- Median: ~$180,000

## Examples Included

### 1. CrewAI Workflow (`run_crewai.py`)

Multi-agent regression workflow with specialized agents:

**Agents:**
1. **Data Loading Specialist** - Handles large dataset loading
2. **Data Cleaning Specialist** - Critical for outlier handling
3. **EDA Expert** - Feature correlations and distributions
4. **ML Engineer** - Regression model training and tuning
5. **Report Analyst** - R², RMSE, MAE metrics analysis

**Process:** Sequential execution with comprehensive regression metrics

**Run:**
```bash
cd /home/thunder/projects/ml-ai-framework
python examples/housing_regression/run_crewai.py
```

**Expected Duration:** 2-5 minutes (full dataset: 20,640 samples)

**Expected Output:**
- Complete regression pipeline execution
- Feature importance analysis
- Performance metrics (R², RMSE, MAE)
- Results saved to `crewai_results.txt`

### 2. LangGraph Workflow (`run_langgraph.py`)

State-based regression workflow with graph execution:

**Graph Structure:**
```
START → load_data → clean_data → perform_eda → train_model → generate_report → END
```

**State Management:**
- Dataset: 20k+ samples with 8 features
- Cleaning report: Outlier handling, missing values
- EDA findings: Correlations, distributions, patterns
- Model: Random Forest Regressor
- Metrics: R², RMSE, MAE, training time

**Run:**
```bash
cd /home/thunder/projects/ml-ai-framework
python examples/housing_regression/run_langgraph.py
```

**Expected Duration:** 2-5 minutes

**Expected Output:**
- State transition logs
- Node execution messages
- Comprehensive JSON report with metrics
- Results saved to `langgraph_results.json`

## Expected Performance

### Model Metrics

**Good Performance Targets:**
- **R² Score:** 0.75 - 0.85 (explains 75-85% of variance)
- **RMSE:** 0.5 - 0.6 ($50k - $60k prediction error)
- **MAE:** 0.35 - 0.45 ($35k - $45k average error)
- **Training Time:** 10-30 seconds (depends on hardware)

### Feature Importance

Typical feature importance ranking:

1. **MedInc** (0.45-0.55) - Strongest predictor
2. **Latitude** (0.15-0.20) - Location matters
3. **Longitude** (0.10-0.15) - Coastal vs inland
4. **AveOccup** (0.05-0.10) - Population density
5. **HouseAge** (0.03-0.08) - Age impact
6. **AveRooms** (0.02-0.05) - House size
7. **Population** (0.01-0.03) - District size
8. **AveBedrms** (0.01-0.02) - Bedroom count

### Insights

**Key Findings:**
- Median income is the dominant predictor
- Geographical location (lat/long) strongly influences prices
- Coastal areas typically have higher values
- House age has moderate negative correlation
- Outliers exist (values capped at $500k in original data)

## Comparison: CrewAI vs LangGraph

| Aspect | CrewAI | LangGraph |
|--------|--------|-----------|
| **Best For** | Complex agent coordination | Structured state flow |
| **Execution Style** | Agent task delegation | Graph node execution |
| **State Sharing** | Context passing | Centralized state object |
| **Debugging** | Agent logs, task outputs | State snapshots, messages |
| **Flexibility** | High (custom agents) | High (custom nodes) |
| **Performance** | Similar (~2-5 min) | Similar (~2-5 min) |

## Performance Optimization

### For Faster Execution

Modify the `prepare_housing_dataset()` call to use sampling:

```python
# In run_crewai.py or run_langgraph.py
dataset_path, df = prepare_housing_dataset(sample_size=5000)
```

**Sample Size Recommendations:**
- **Development/Testing:** 1,000 - 5,000 samples
- **Full Analysis:** 20,640 samples (complete dataset)

### Expected Performance by Sample Size

| Sample Size | Execution Time | R² Score |
|-------------|----------------|----------|
| 1,000 | 30-60 seconds | 0.70-0.75 |
| 5,000 | 1-2 minutes | 0.75-0.80 |
| 10,000 | 1.5-3 minutes | 0.78-0.82 |
| 20,640 | 2-5 minutes | 0.75-0.85 |

## Files Generated

After running the examples:

```
housing_regression/
├── run_crewai.py              # CrewAI workflow script
├── run_langgraph.py           # LangGraph workflow script
├── README.md                  # This file
├── crewai_results.txt         # CrewAI execution results
└── langgraph_results.json     # LangGraph execution results
```

## Customization Examples

### 1. Change Model Algorithm

```python
# Modify workflow to use Linear Regression
# In the framework's train_model call:
model, performance = train_model(
    X, y,
    model_type="regression",
    algorithm="linear_regression",  # Instead of random_forest
)
```

### 2. Custom Hyperparameters

```python
# For Random Forest Regressor
hyperparameters = {
    "n_estimators": 200,      # More trees (default: 100)
    "max_depth": 15,          # Deeper trees
    "min_samples_split": 10,  # Require more samples to split
    "min_samples_leaf": 5,    # Require more samples in leaf
    "random_state": 42
}
```

### 3. Different Train/Test Split

```python
# In train_model call
model, performance = train_model(
    X, y,
    model_type="regression",
    test_size=0.25,   # 25% test set (default: 0.2)
    val_size=0.15,    # 15% validation set (default: 0.1)
)
```

## Troubleshooting

### Issue: Long execution time

**Solution:** Use sampling to reduce dataset size:
```python
dataset_path, df = prepare_housing_dataset(sample_size=5000)
```

### Issue: Poor R² score (<0.7)

**Possible causes:**
- Outliers not properly handled
- Feature scaling issues
- Model underfitting

**Solution:** Check cleaning parameters and increase model complexity

### Issue: High RMSE (>0.7)

**Analysis:** RMSE of 0.7 means ~$70k error
- Check if outliers are skewing results
- Review feature importance
- Consider feature engineering

### Issue: Memory errors

**Solution:** Reduce sample size or use incremental learning:
```python
# Use smaller sample
dataset_path, df = prepare_housing_dataset(sample_size=10000)
```

## Learning Objectives

After running these examples, you should understand:

1. **Regression Workflows:** Complete pipeline from data to predictions
2. **Regression Metrics:** R², RMSE, MAE interpretation
3. **Feature Importance:** Identifying key predictors
4. **Outlier Handling:** Impact on regression performance
5. **Geographical Features:** Latitude/longitude as predictors
6. **Model Evaluation:** Validation strategies for regression
7. **Performance Tuning:** Balancing accuracy and speed

## Advanced Topics

### Feature Engineering Ideas

1. **Create derived features:**
   - Rooms per person: AveRooms / AveOccup
   - Bedroom ratio: AveBedrms / AveRooms
   - Population density: Population / (AveOccup * households)

2. **Geographical clustering:**
   - Coastal vs inland (based on longitude)
   - Urban vs rural (based on population)
   - Regional groupings (lat/long bins)

3. **Interaction features:**
   - Income × Location
   - Age × Rooms
   - Population × Occupancy

### Model Comparisons

Try different algorithms and compare:

```python
algorithms = [
    ("Random Forest", "random_forest"),
    ("Linear Regression", "linear_regression"),
]

for name, algo in algorithms:
    model, perf = train_model(X, y, algorithm=algo)
    print(f"{name} R²: {perf.validation_metrics.r2_score:.4f}")
```

## Performance Benchmarks

**Typical System Performance:**
- **CPU:** i7-9750H (6 cores)
- **RAM:** 16GB
- **Full Dataset:** 3-4 minutes
- **Sample (5k):** 1 minute

**Resource Usage:**
- Memory: 1-2 GB
- CPU: 60-80% (during training)
- Disk: ~5 MB for results

## Next Steps

1. **Try Iris Classification:** See `../iris_analysis/` for classification examples
2. **Advanced Patterns:** Check `../advanced_workflow/` for custom agents and parallel execution
3. **Quick Start:** See `../quickstart/` for building custom workflows
4. **Custom Dataset:** Adapt these examples for your own regression tasks

## References

- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [Regression Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Feature Importance](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

## Citation

If using this dataset in research:

```
Pace, R. Kelley and Ronald Barry, "Sparse Spatial Autoregressions",
Statistics and Probability Letters, 33 (1997) 291-297.
```
