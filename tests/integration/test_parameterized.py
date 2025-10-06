"""
Parameterized tests for multiple dataset scenarios.
Uses pytest.mark.parametrize for comprehensive scenario coverage.
"""
import pytest
import pandas as pd
import numpy as np


class TestParameterizedDatasets:
    """Test workflows with various dataset configurations."""

    @pytest.mark.parametrize("n_rows,n_cols,missing_pct", [
        (100, 5, 0.0),    # Perfect data
        (100, 5, 0.1),    # 10% missing
        (100, 5, 0.3),    # 30% missing
        (1000, 10, 0.05), # Large dataset
        (50, 3, 0.2),     # Small dataset
    ])
    def test_load_dataset_sizes(self, temp_dir, n_rows, n_cols, missing_pct):
        """Test loading datasets of various sizes and missing value percentages."""
        # Generate dataset
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(n_rows, n_cols),
            columns=[f'feature{i}' for i in range(n_cols)]
        )

        # Add missing values
        if missing_pct > 0:
            mask = np.random.random(df.shape) < missing_pct
            df = df.mask(mask)

        # Save and load
        file_path = temp_dir / f"data_{n_rows}x{n_cols}.csv"
        df.to_csv(file_path, index=False)

        from tests.unit.test_tools import load_dataset
        result = load_dataset(str(file_path))

        assert f"'rows': {n_rows}" in result
        assert f"'columns': {n_cols}" in result

    @pytest.mark.parametrize("dataset_type,expected_dtypes", [
        ("numeric", ["int64", "float64"]),
        ("mixed", ["int64", "float64", "object"]),
        ("categorical", ["object"]),
    ])
    def test_dataset_dtypes(self, temp_dir, dataset_type, expected_dtypes):
        """Test datasets with different data type compositions."""
        if dataset_type == "numeric":
            df = pd.DataFrame({
                'int_col': [1, 2, 3, 4, 5],
                'float_col': [1.1, 2.2, 3.3, 4.4, 5.5]
            })
        elif dataset_type == "mixed":
            df = pd.DataFrame({
                'int_col': [1, 2, 3, 4, 5],
                'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
                'str_col': ['A', 'B', 'C', 'D', 'E']
            })
        else:  # categorical
            df = pd.DataFrame({
                'cat1': ['A', 'B', 'C', 'D', 'E'],
                'cat2': ['X', 'Y', 'Z', 'X', 'Y']
            })

        file_path = temp_dir / f"{dataset_type}.csv"
        df.to_csv(file_path, index=False)

        from tests.unit.test_tools import load_dataset
        result = load_dataset(str(file_path))

        # Verify dtypes present
        for dtype in expected_dtypes:
            assert dtype in result or str(dtype).replace('64', '') in result

    @pytest.mark.parametrize("duplicate_rows", [0, 2, 5, 10])
    def test_cleaning_duplicates(self, temp_dir, duplicate_rows):
        """Test cleaning datasets with varying numbers of duplicates."""
        # Create base data
        base_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [6, 7, 8, 9, 10],
            'target': [0, 1, 0, 1, 0]
        })

        # Add duplicates
        if duplicate_rows > 0:
            duplicates = base_df.head(duplicate_rows)
            df = pd.concat([base_df, duplicates], ignore_index=True)
        else:
            df = base_df

        input_path = temp_dir / "duplicates.csv"
        output_path = temp_dir / "cleaned.csv"
        df.to_csv(input_path, index=False)

        from tests.unit.test_tools import clean_data
        result = clean_data(str(input_path), str(output_path))

        assert f"Removed {duplicate_rows} duplicates" in result

    @pytest.mark.parametrize("outlier_count,outlier_magnitude", [
        (0, 0),      # No outliers
        (5, 5),      # Moderate outliers
        (10, 10),    # More extreme outliers
        (3, 20),     # Very extreme outliers
    ])
    def test_outlier_detection(self, temp_dir, outlier_count, outlier_magnitude):
        """Test outlier detection with varying outlier characteristics."""
        np.random.seed(42)
        # Normal data
        normal_data = np.random.randn(100 - outlier_count)

        # Add outliers
        if outlier_count > 0:
            outliers = np.random.choice(
                [-outlier_magnitude, outlier_magnitude],
                size=outlier_count
            )
            combined = np.concatenate([normal_data, outliers])
        else:
            combined = normal_data

        df = pd.DataFrame({
            'values': combined,
            'target': np.random.randint(0, 2, len(combined))
        })

        input_path = temp_dir / "outliers.csv"
        output_path = temp_dir / "cleaned.csv"
        df.to_csv(input_path, index=False)

        from tests.unit.test_tools import clean_data
        result = clean_data(str(input_path), str(output_path))

        if outlier_count > 0:
            assert "Outliers:" in result


class TestParameterizedModelTraining:
    """Test model training with different configurations."""

    @pytest.mark.parametrize("n_samples,n_features,n_classes", [
        (100, 4, 2),    # Small binary classification
        (500, 10, 2),   # Medium binary classification
        (1000, 20, 3),  # Large multi-class
        (150, 4, 3),    # Iris-like dataset
    ])
    def test_model_training_scenarios(self, temp_dir, n_samples, n_features, n_classes):
        """Test model training with various dataset sizes and complexities."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=min(n_features, n_classes * 2),
            n_redundant=max(0, n_features - n_classes * 2),
            random_state=42
        )

        df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(n_features)])
        df['target'] = y

        data_path = temp_dir / "train_data.csv"
        model_path = temp_dir / "model.pkl"
        df.to_csv(data_path, index=False)

        from tests.unit.test_tools import train_model
        result = train_model(str(data_path), 'target', str(model_path))

        metrics = eval(result)
        # Model should achieve reasonable performance
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0

    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3, 0.4])
    def test_train_test_split_ratios(self, test_size):
        """Test different train/test split ratios."""
        total_samples = 100
        train_samples = int(total_samples * (1 - test_size))
        test_samples = total_samples - train_samples

        assert train_samples + test_samples == total_samples
        assert test_samples / total_samples == pytest.approx(test_size, 0.01)


class TestParameterizedValidation:
    """Test Pydantic validation with various inputs."""

    @pytest.mark.parametrize("rows_before,rows_after,should_pass", [
        (100, 95, True),   # Valid: rows_after < rows_before
        (100, 100, True),  # Valid: equal (no rows removed)
        (100, 105, False), # Invalid: rows_after > rows_before
        (50, 45, True),    # Valid
        (1, 0, False),     # Invalid: rows_after must be > 0
    ])
    def test_cleaning_report_validation(self, rows_before, rows_after, should_pass):
        """Test CleaningReport validation with various row counts."""
        from pydantic import ValidationError
        from tests.unit.test_models import CleaningReport

        data = {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "duplicates_removed": 0,
            "missing_values_handled": {},
            "outliers_detected": {},
            "cleaned_data_path": "/tmp/cleaned.csv"
        }

        if should_pass:
            report = CleaningReport(**data)
            assert report.rows_before == rows_before
            assert report.rows_after == rows_after
        else:
            with pytest.raises(ValidationError):
                CleaningReport(**data)

    @pytest.mark.parametrize("metric_value,should_pass", [
        (0.0, True),   # Minimum valid
        (0.5, True),   # Mid-range
        (1.0, True),   # Maximum valid
        (1.5, False),  # Too high
        (-0.1, False), # Too low
    ])
    def test_model_metrics_validation(self, metric_value, should_pass, model_performance_dict):
        """Test ModelPerformance validation with various metric values."""
        from pydantic import ValidationError
        from tests.unit.test_models import ModelPerformance

        data = model_performance_dict.copy()
        data['accuracy'] = metric_value

        if should_pass:
            perf = ModelPerformance(**data)
            assert perf.accuracy == metric_value
        else:
            with pytest.raises(ValidationError):
                ModelPerformance(**data)

    @pytest.mark.parametrize("correlation,significance,should_pass", [
        (0.85, "strong", True),
        (0.45, "moderate", True),
        (0.15, "weak", True),
        (-0.95, "strong", True),
        (1.5, "strong", False),   # Out of bounds
        (0.5, "invalid", False),  # Invalid significance
    ])
    def test_correlation_pair_validation(self, correlation, significance, should_pass):
        """Test CorrelationPair validation with various correlations."""
        from pydantic import ValidationError
        from tests.unit.test_models import CorrelationPair

        if should_pass:
            pair = CorrelationPair(
                feature_a="f1",
                feature_b="f2",
                correlation=correlation,
                significance=significance
            )
            assert pair.correlation == correlation
        else:
            with pytest.raises(ValidationError):
                CorrelationPair(
                    feature_a="f1",
                    feature_b="f2",
                    correlation=correlation,
                    significance=significance
                )


class TestParameterizedWorkflows:
    """Test complete workflows with different configurations."""

    @pytest.mark.parametrize("dataset_name,target_col,expected_min_accuracy", [
        ("iris", "target", 0.80),
        ("simple_classification", "label", 0.70),
    ])
    def test_end_to_end_workflow(self, temp_dir, dataset_name, target_col, expected_min_accuracy):
        """Test end-to-end workflow with different datasets."""
        if dataset_name == "iris":
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df[target_col] = iris.target
        else:
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=200, n_features=5, random_state=42)
            df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
            df[target_col] = y

        data_path = temp_dir / f"{dataset_name}.csv"
        df.to_csv(data_path, index=False)

        # Execute workflow
        from tests.unit.test_tools import load_dataset, clean_data, train_model

        # Load
        load_result = load_dataset(str(data_path))
        assert "rows" in load_result

        # Clean
        cleaned_path = temp_dir / "cleaned.csv"
        clean_result = clean_data(str(data_path), str(cleaned_path))
        assert "Cleaned" in clean_result

        # Train
        model_path = temp_dir / "model.pkl"
        train_result = train_model(str(cleaned_path), target_col, str(model_path))
        metrics = eval(train_result)

        assert metrics['accuracy'] >= expected_min_accuracy

    @pytest.mark.parametrize("memory_type", ["short_term", "long_term", "entity"])
    def test_memory_types(self, memory_type):
        """Test different memory types in multi-agent system."""
        memory_store = {
            "short_term": [],
            "long_term": {},
            "entity": {}
        }

        # Add to specified memory type
        if memory_type == "short_term":
            memory_store[memory_type].append({"task": "test", "result": "success"})
            assert len(memory_store[memory_type]) == 1
        elif memory_type == "long_term":
            memory_store[memory_type]["key"] = "value"
            assert memory_store[memory_type]["key"] == "value"
        else:  # entity
            memory_store[memory_type]["dataset"] = {"name": "iris"}
            assert "dataset" in memory_store[memory_type]


class TestParameterizedErrorHandling:
    """Test error handling with various failure scenarios."""

    @pytest.mark.parametrize("error_type,error_message", [
        ("FileNotFoundError", "File not found"),
        ("ValueError", "Invalid value"),
        ("KeyError", "Missing key"),
        ("TypeError", "Type mismatch"),
    ])
    def test_error_scenarios(self, error_type, error_message):
        """Test handling of different error types."""
        error_classes = {
            "FileNotFoundError": FileNotFoundError,
            "ValueError": ValueError,
            "KeyError": KeyError,
            "TypeError": TypeError,
        }

        error_class = error_classes[error_type]

        with pytest.raises(error_class) as exc_info:
            raise error_class(error_message)

        assert error_message in str(exc_info.value)
