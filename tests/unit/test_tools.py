"""
Unit tests for custom tools with mocked dependencies.
Tests tool functionality without external API calls or file I/O.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import tempfile
import os


# Mock tool implementations for testing
# These will match the actual tool signatures from the spec


def load_dataset(file_path: str) -> str:
    """Load CSV dataset and return metadata JSON string."""
    try:
        df = pd.read_csv(file_path)
        metadata = {
            "name": file_path.split('/')[-1],
            "source": file_path,
            "rows": len(df),
            "columns": len(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(df[col].isna().sum()) for col in df.columns}
        }
        return str(metadata)
    except Exception as e:
        return f"Error loading dataset: {str(e)}"


def clean_data(input_path: str, output_path: str) -> str:
    """Clean dataset: handle missing values, remove duplicates, detect outliers."""
    df = pd.read_csv(input_path)
    rows_before = len(df)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Remove duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()

    # Detect outliers using IQR
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
        outliers[col] = int(outlier_mask.sum())

    df.to_csv(output_path, index=False)

    return f"Cleaned {rows_before} rows to {len(df)}. Removed {duplicates} duplicates. Outliers: {outliers}"


def calculate_statistics(file_path: str) -> str:
    """Generate comprehensive statistical summary."""
    df = pd.read_csv(file_path)
    stats = df.describe().to_dict()

    # Add correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr().to_dict()

    return f"Statistics: {stats}\n\nCorrelations: {correlations}"


def train_model(data_path: str, target_column: str, model_output_path: str) -> str:
    """Train classification model and return performance metrics."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import joblib

    df = pd.read_csv(data_path)

    # Prepare features and target
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
    y = df[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average='weighted')),
        "recall": float(recall_score(y_test, y_pred, average='weighted')),
        "f1_score": float(f1_score(y_test, y_pred, average='weighted')),
        "feature_importance": {col: float(imp) for col, imp in zip(X.columns, model.feature_importances_)}
    }

    # Save model
    joblib.dump(model, model_output_path)

    return str(metrics)


class TestLoadDatasetTool:
    """Test suite for load_dataset tool."""

    def test_load_valid_dataset(self, temp_dir, sample_dataset):
        """Test loading a valid CSV dataset."""
        file_path = temp_dir / "test_data.csv"
        sample_dataset.to_csv(file_path, index=False)

        result = load_dataset(str(file_path))

        assert "test_data.csv" in result
        assert "'rows': 100" in result
        assert "'columns': 5" in result

    def test_load_dataset_with_missing_values(self, temp_dir, dataset_with_missing_values):
        """Test loading dataset with missing values."""
        file_path = temp_dir / "missing_data.csv"
        dataset_with_missing_values.to_csv(file_path, index=False)

        result = load_dataset(str(file_path))

        assert "'rows': 10" in result
        assert "missing_values" in result

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        result = load_dataset("/nonexistent/path/data.csv")
        assert "Error loading dataset" in result

    def test_load_invalid_csv(self, temp_dir):
        """Test loading an invalid CSV file."""
        file_path = temp_dir / "invalid.csv"
        with open(file_path, 'w') as f:
            f.write("Invalid CSV content\nwith inconsistent\ncolumns")

        # Should handle gracefully
        result = load_dataset(str(file_path))
        # Pandas might still parse it, or return error
        assert result is not None

    def test_metadata_structure(self, temp_dir, sample_dataset):
        """Test that metadata contains all required fields."""
        file_path = temp_dir / "test.csv"
        sample_dataset.to_csv(file_path, index=False)

        result = load_dataset(str(file_path))

        # Check for required metadata fields
        assert "name" in result
        assert "source" in result
        assert "rows" in result
        assert "columns" in result
        assert "dtypes" in result
        assert "missing_values" in result

    def test_dtypes_detection(self, temp_dir):
        """Test correct dtype detection for various column types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        file_path = temp_dir / "types.csv"
        df.to_csv(file_path, index=False)

        result = load_dataset(str(file_path))

        assert "int_col" in result
        assert "float_col" in result
        assert "str_col" in result


class TestCleanDataTool:
    """Test suite for clean_data tool."""

    def test_clean_dataset_with_duplicates(self, temp_dir, dataset_with_duplicates):
        """Test duplicate removal."""
        input_path = temp_dir / "duplicates.csv"
        output_path = temp_dir / "cleaned.csv"
        dataset_with_duplicates.to_csv(input_path, index=False)

        result = clean_data(str(input_path), str(output_path))

        assert "Removed 2 duplicates" in result
        assert os.path.exists(output_path)

        # Verify cleaned data
        cleaned_df = pd.read_csv(output_path)
        assert len(cleaned_df) == 3  # Original 5 - 2 duplicates

    def test_clean_dataset_with_missing_values(self, temp_dir, dataset_with_missing_values):
        """Test missing value imputation."""
        input_path = temp_dir / "missing.csv"
        output_path = temp_dir / "cleaned.csv"
        dataset_with_missing_values.to_csv(input_path, index=False)

        result = clean_data(str(input_path), str(output_path))

        assert "Cleaned" in result

        # Verify no missing values remain
        cleaned_df = pd.read_csv(output_path)
        assert cleaned_df.isna().sum().sum() == 0

    def test_outlier_detection(self, temp_dir, dataset_with_outliers):
        """Test outlier detection using IQR method."""
        input_path = temp_dir / "outliers.csv"
        output_path = temp_dir / "cleaned.csv"
        dataset_with_outliers.to_csv(input_path, index=False)

        result = clean_data(str(input_path), str(output_path))

        assert "Outliers:" in result
        assert "'values':" in result  # Should detect outliers in 'values' column

    def test_preserve_data_without_issues(self, temp_dir, sample_dataset):
        """Test that clean dataset without issues is preserved."""
        input_path = temp_dir / "clean.csv"
        output_path = temp_dir / "output.csv"
        sample_dataset.to_csv(input_path, index=False)

        result = clean_data(str(input_path), str(output_path))

        cleaned_df = pd.read_csv(output_path)
        assert len(cleaned_df) > 0

    def test_numeric_vs_categorical_handling(self, temp_dir):
        """Test different imputation strategies for numeric vs categorical."""
        df = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'categorical': ['A', 'B', np.nan, 'A', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        input_path = temp_dir / "mixed.csv"
        output_path = temp_dir / "cleaned.csv"
        df.to_csv(input_path, index=False)

        clean_data(str(input_path), str(output_path))

        cleaned_df = pd.read_csv(output_path)
        # Numeric should use median, categorical should use mode
        assert pd.notna(cleaned_df['numeric']).all()
        assert pd.notna(cleaned_df['categorical']).all()


class TestCalculateStatisticsTool:
    """Test suite for calculate_statistics tool."""

    def test_calculate_basic_statistics(self, temp_dir, sample_dataset):
        """Test calculation of descriptive statistics."""
        file_path = temp_dir / "data.csv"
        sample_dataset.to_csv(file_path, index=False)

        result = calculate_statistics(str(file_path))

        assert "Statistics:" in result
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result

    def test_calculate_correlations(self, temp_dir):
        """Test correlation matrix calculation."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],  # Perfect correlation with a
            'c': [5, 4, 3, 2, 1]   # Perfect negative correlation with a
        })
        file_path = temp_dir / "corr.csv"
        df.to_csv(file_path, index=False)

        result = calculate_statistics(str(file_path))

        assert "Correlations:" in result
        # Should show strong correlations
        assert "'a':" in result
        assert "'b':" in result
        assert "'c':" in result

    def test_statistics_with_single_column(self, temp_dir):
        """Test statistics calculation with single numeric column."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        file_path = temp_dir / "single.csv"
        df.to_csv(file_path, index=False)

        result = calculate_statistics(str(file_path))

        assert "Statistics:" in result
        assert "value" in result

    def test_statistics_exclude_categorical(self, temp_dir):
        """Test that categorical columns are excluded from numeric statistics."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'C', 'A', 'B']
        })
        file_path = temp_dir / "mixed.csv"
        df.to_csv(file_path, index=False)

        result = calculate_statistics(str(file_path))

        # Numeric stats should be calculated
        assert "numeric" in result


class TestTrainModelTool:
    """Test suite for train_model tool."""

    def test_train_classification_model(self, temp_dir, iris_dataset_path):
        """Test training a classification model."""
        model_path = temp_dir / "model.pkl"

        result = train_model(str(iris_dataset_path), 'target', str(model_path))

        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "feature_importance" in result
        assert os.path.exists(model_path)

    def test_model_performance_metrics(self, temp_dir, iris_dataset_path):
        """Test that performance metrics are within valid ranges."""
        model_path = temp_dir / "model.pkl"

        result = train_model(str(iris_dataset_path), 'target', str(model_path))

        # Parse metrics
        metrics_dict = eval(result)

        # All metrics should be between 0 and 1
        assert 0.0 <= metrics_dict['accuracy'] <= 1.0
        assert 0.0 <= metrics_dict['precision'] <= 1.0
        assert 0.0 <= metrics_dict['recall'] <= 1.0
        assert 0.0 <= metrics_dict['f1_score'] <= 1.0

    def test_feature_importance_extraction(self, temp_dir, iris_dataset_path):
        """Test feature importance calculation."""
        model_path = temp_dir / "model.pkl"

        result = train_model(str(iris_dataset_path), 'target', str(model_path))
        metrics_dict = eval(result)

        assert 'feature_importance' in metrics_dict
        assert len(metrics_dict['feature_importance']) > 0

        # Feature importances should sum to approximately 1
        total_importance = sum(metrics_dict['feature_importance'].values())
        assert 0.99 <= total_importance <= 1.01

    def test_model_persistence(self, temp_dir, iris_dataset_path):
        """Test that trained model is saved correctly."""
        import joblib
        model_path = temp_dir / "model.pkl"

        train_model(str(iris_dataset_path), 'target', str(model_path))

        # Load and verify model
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, 'predict')
        assert hasattr(loaded_model, 'predict_proba')

    def test_reproducible_results(self, temp_dir, iris_dataset_path):
        """Test that results are reproducible with same random seed."""
        model_path1 = temp_dir / "model1.pkl"
        model_path2 = temp_dir / "model2.pkl"

        result1 = train_model(str(iris_dataset_path), 'target', str(model_path1))
        result2 = train_model(str(iris_dataset_path), 'target', str(model_path2))

        # Results should be identical due to random_state=42
        metrics1 = eval(result1)
        metrics2 = eval(result2)

        assert metrics1['accuracy'] == metrics2['accuracy']
        assert metrics1['f1_score'] == metrics2['f1_score']


class TestToolErrorHandling:
    """Test error handling across all tools."""

    def test_load_dataset_error_handling(self):
        """Test graceful error handling in load_dataset."""
        result = load_dataset("/invalid/path.csv")
        assert "Error" in result

    def test_clean_data_missing_input(self, temp_dir):
        """Test clean_data with missing input file."""
        output_path = temp_dir / "output.csv"

        with pytest.raises(Exception):
            clean_data("/nonexistent.csv", str(output_path))

    def test_calculate_statistics_invalid_file(self):
        """Test calculate_statistics with invalid file."""
        with pytest.raises(Exception):
            calculate_statistics("/invalid/path.csv")

    def test_train_model_missing_target(self, temp_dir, sample_dataset):
        """Test train_model with non-existent target column."""
        file_path = temp_dir / "data.csv"
        sample_dataset.to_csv(file_path, index=False)
        model_path = temp_dir / "model.pkl"

        with pytest.raises(Exception):
            train_model(str(file_path), 'nonexistent_target', str(model_path))


class TestToolIntegration:
    """Integration tests for tool workflows."""

    def test_complete_tool_pipeline(self, temp_dir, dataset_with_missing_values):
        """Test complete pipeline: load -> clean -> stats -> train."""
        # Step 1: Save initial dataset
        input_path = temp_dir / "raw.csv"
        dataset_with_missing_values.to_csv(input_path, index=False)

        # Step 2: Load dataset
        load_result = load_dataset(str(input_path))
        assert "'rows': 10" in load_result

        # Step 3: Clean data
        cleaned_path = temp_dir / "cleaned.csv"
        clean_result = clean_data(str(input_path), str(cleaned_path))
        assert "Cleaned" in clean_result

        # Step 4: Calculate statistics
        stats_result = calculate_statistics(str(cleaned_path))
        assert "Statistics:" in stats_result

        # Step 5: Train model
        model_path = temp_dir / "model.pkl"
        train_result = train_model(str(cleaned_path), 'target', str(model_path))
        assert "accuracy" in train_result

        # Verify all steps completed successfully
        assert os.path.exists(cleaned_path)
        assert os.path.exists(model_path)
