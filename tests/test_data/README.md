# Test Data Directory

This directory contains sample datasets for testing the ML-AI framework.

## Dataset Types

### Classification Datasets
- `iris.csv` - Classic Iris dataset (150 rows, 5 columns, 3 classes)
- `binary_classification.csv` - Binary classification (1000 rows, 20 features)
- `imbalanced.csv` - Imbalanced dataset (10:1 ratio)

### Regression Datasets
- `regression_simple.csv` - Simple regression task (500 rows, 10 features)
- `regression_complex.csv` - Complex regression (1000 rows, 50 features)

### Data Quality Test Datasets
- `missing_values.csv` - Dataset with 20% missing values
- `duplicates.csv` - Dataset with duplicate rows
- `outliers.csv` - Dataset with statistical outliers
- `high_correlation.csv` - Features with known correlations

### Edge Case Datasets
- `single_feature.csv` - Only one feature
- `single_row.csv` - Only one sample
- `all_categorical.csv` - All categorical features
- `mixed_types.csv` - Mix of numeric and categorical

## Generation

Datasets are generated using `fixtures/sample_datasets.py`:

```python
from tests.fixtures.sample_datasets import generate_iris_dataset

df = generate_iris_dataset(output_path='tests/test_data/iris.csv')
```

## Usage in Tests

```python
import pandas as pd

def test_with_iris():
    df = pd.read_csv('tests/test_data/iris.csv')
    # Run tests...
```
