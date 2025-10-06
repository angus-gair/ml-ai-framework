"""Quick Start 04: Testing ML Workflows

Learn how to test ML workflows, tools, and agents.

What you'll learn:
- Unit testing ML functions
- Integration testing workflows
- Mocking external dependencies
- Testing with fixtures
- Assertion patterns for ML
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict
from unittest.mock import Mock, patch, MagicMock


# Example 1: Unit Test for Data Tool
def test_data_loading():
    """
    Test data loading function.

    Pattern: Test function inputs and outputs
    """
    from src.tools.data_tools import load_dataset

    # Create test data
    test_file = "/tmp/test_data.csv"
    df_test = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    df_test.to_csv(test_file, index=False)

    # Test loading
    df, metadata = load_dataset(test_file)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ['feature1', 'feature2', 'target']
    assert metadata.total_rows == 3
    assert metadata.total_columns == 3

    print("✓ test_data_loading passed")


# Example 2: Test Data Cleaning
def test_data_cleaning():
    """
    Test data cleaning function.

    Pattern: Test transformations and reports
    """
    from src.tools.data_tools import clean_data

    # Create messy data
    df_messy = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [1, 2, 2, 3, 4],  # Has duplicate
        'C': [10, 20, 30, 40, 1000],  # Has outlier
    })

    # Test cleaning
    df_clean, report = clean_data(
        df_messy,
        drop_duplicates=True,
        handle_missing="drop",
        outlier_method="iqr",
    )

    # Assertions
    assert len(df_clean) < len(df_messy)  # Rows removed
    assert report.rows_removed > 0
    assert len(report.operations) > 0
    assert report.duration_seconds > 0

    print("✓ test_data_cleaning passed")


# Example 3: Test Model Training
def test_model_training():
    """
    Test model training function.

    Pattern: Test ML pipeline with known data
    """
    from src.tools.ml_tools import train_model
    from sklearn.datasets import make_classification

    # Create synthetic data
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    y_series = pd.Series(y, name='target')

    # Train model
    model, performance = train_model(
        X_df,
        y_series,
        model_type="classification",
        algorithm="random_forest",
        hyperparameters={"n_estimators": 10, "random_state": 42},
    )

    # Assertions
    assert model is not None
    assert performance.model_type.value == "classification"
    assert performance.train_metrics.accuracy > 0.7
    assert performance.validation_metrics.accuracy > 0.5
    assert performance.training_duration_seconds > 0
    assert len(performance.feature_names) == 5

    print("✓ test_model_training passed")


# Example 4: Test with Mocking
def test_workflow_with_mock():
    """
    Test workflow using mocks.

    Pattern: Mock external dependencies
    """
    # Mock the expensive operations
    with patch('src.tools.ml_tools.train_model') as mock_train:
        # Setup mock return value
        mock_performance = Mock()
        mock_performance.model_type.value = "classification"
        mock_performance.validation_metrics.accuracy = 0.95

        mock_train.return_value = (Mock(), mock_performance)

        # Call function that uses train_model
        from src.tools.ml_tools import train_model

        X = pd.DataFrame([[1, 2], [3, 4]])
        y = pd.Series([0, 1])

        model, perf = train_model(X, y)

        # Verify mock was called
        assert mock_train.called
        assert perf.validation_metrics.accuracy == 0.95

    print("✓ test_workflow_with_mock passed")


# Example 5: Integration Test
def test_complete_workflow():
    """
    Test complete workflow end-to-end.

    Pattern: Integration test with real components
    """
    from sklearn.datasets import load_iris
    from src.workflows.langgraph_system import LangGraphWorkflow

    # Prepare test data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    test_file = "/tmp/test_iris.csv"
    df.to_csv(test_file, index=False)

    # Run workflow
    workflow = LangGraphWorkflow(model_name="gpt-4", temperature=0.7)

    result = workflow.run(
        data_path=test_file,
        target_column="species",
    )

    # Assertions
    assert result["status"] in ["success", "completed_with_errors"]
    assert "report" in result
    assert "execution_time_seconds" in result
    assert result["execution_time_seconds"] > 0

    print("✓ test_complete_workflow passed")


# Example 6: Fixture Pattern
class TestDataFixture:
    """
    Reusable test data fixture.

    Pattern: Setup and teardown for tests
    """

    @staticmethod
    def create_sample_classification_data():
        """Create sample classification dataset."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=3,
            random_state=42
        )

        return pd.DataFrame(X), pd.Series(y)

    @staticmethod
    def create_sample_regression_data():
        """Create sample regression dataset."""
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42
        )

        return pd.DataFrame(X), pd.Series(y)


def test_using_fixture():
    """
    Test using reusable fixtures.

    Pattern: DRY principle with fixtures
    """
    # Use fixture
    X, y = TestDataFixture.create_sample_classification_data()

    # Test assertions
    assert len(X) == 200
    assert X.shape[1] == 10
    assert len(y.unique()) == 3

    print("✓ test_using_fixture passed")


# Example 7: Property-Based Testing
def test_data_invariants():
    """
    Test data properties that should always hold.

    Pattern: Property-based testing
    """
    from src.tools.data_tools import clean_data

    # Create various datasets
    for _ in range(5):
        n_samples = np.random.randint(50, 200)
        n_features = np.random.randint(3, 10)

        df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'f{i}' for i in range(n_features)]
        )

        # Add some NaNs randomly
        mask = np.random.random(df.shape) < 0.1
        df = df.mask(mask)

        # Clean data
        df_clean, report = clean_data(df, handle_missing="drop")

        # Invariants that should always hold
        assert df_clean.isnull().sum().sum() == 0, "No nulls after cleaning"
        assert len(df_clean) <= len(df), "Can't add rows during cleaning"
        assert report.rows_before == len(df), "Report tracks original size"
        assert report.rows_after == len(df_clean), "Report tracks final size"

    print("✓ test_data_invariants passed")


# Example 8: Performance Test
def test_performance_benchmark():
    """
    Test performance benchmarks.

    Pattern: Performance regression testing
    """
    import time
    from src.tools.ml_tools import train_model
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)

    # Benchmark training time
    start = time.time()
    model, perf = train_model(
        X_df,
        y_series,
        algorithm="random_forest",
        hyperparameters={"n_estimators": 50, "random_state": 42},
    )
    duration = time.time() - start

    # Performance assertions
    assert duration < 30, f"Training took too long: {duration:.2f}s"
    assert perf.training_duration_seconds < 30

    print(f"✓ test_performance_benchmark passed (duration: {duration:.2f}s)")


def run_all_tests():
    """Run all test examples."""

    print("Quick Start 04: Testing ML Workflows")
    print("=" * 60)
    print()

    tests = [
        ("Unit Test: Data Loading", test_data_loading),
        ("Unit Test: Data Cleaning", test_data_cleaning),
        ("Unit Test: Model Training", test_model_training),
        ("Mock Test: External Dependencies", test_workflow_with_mock),
        ("Integration Test: Complete Workflow", test_complete_workflow),
        ("Fixture Test: Reusable Data", test_using_fixture),
        ("Property Test: Data Invariants", test_data_invariants),
        ("Performance Test: Benchmarks", test_performance_benchmark),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"Running: {name}")
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
            print()

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print()

    print("Key Takeaways:")
    print("  1. Unit tests verify individual functions")
    print("  2. Integration tests verify complete workflows")
    print("  3. Mocks isolate components for testing")
    print("  4. Fixtures provide reusable test data")
    print("  5. Property tests verify invariants")
    print("  6. Performance tests catch regressions")
    print()
    print("Next: Explore advanced_workflow/ for complex patterns")


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging(log_level="WARNING", log_to_file=False)

    run_all_tests()
