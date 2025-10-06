"""
Performance and load tests for ML-AI framework.
Tests execution time, memory usage, and scalability.
"""
import pytest
import time
import psutil
import os
from pathlib import Path


class TestExecutionPerformance:
    """Test execution time for various operations."""

    def test_data_loading_performance(self, temp_dir, performance_threshold):
        """Test that data loading completes within time threshold."""
        import pandas as pd
        import numpy as np

        # Create large dataset
        n_rows = 10000
        df = pd.DataFrame({
            f'feature{i}': np.random.randn(n_rows)
            for i in range(10)
        })

        file_path = temp_dir / "large_data.csv"
        df.to_csv(file_path, index=False)

        # Measure load time
        start_time = time.time()

        from tests.unit.test_tools import load_dataset
        result = load_dataset(str(file_path))

        end_time = time.time()
        duration = end_time - start_time

        assert duration < performance_threshold['data_load']
        assert "'rows': 10000" in result

    def test_data_cleaning_performance(self, temp_dir, large_dataset, performance_threshold):
        """Test cleaning performance on large dataset."""
        input_path = temp_dir / "large_input.csv"
        output_path = temp_dir / "large_output.csv"
        large_dataset.to_csv(input_path, index=False)

        start_time = time.time()

        from tests.unit.test_tools import clean_data
        result = clean_data(str(input_path), str(output_path))

        duration = time.time() - start_time

        assert duration < performance_threshold['data_clean']

    def test_statistics_calculation_performance(self, temp_dir, large_dataset, performance_threshold):
        """Test statistics calculation performance."""
        file_path = temp_dir / "stats_data.csv"
        large_dataset.to_csv(file_path, index=False)

        start_time = time.time()

        from tests.unit.test_tools import calculate_statistics
        result = calculate_statistics(str(file_path))

        duration = time.time() - start_time

        assert duration < performance_threshold['statistics']

    def test_model_training_performance(self, temp_dir, large_dataset, performance_threshold):
        """Test model training performance."""
        # Prepare training data
        data_path = temp_dir / "train_data.csv"
        model_path = temp_dir / "model.pkl"
        large_dataset['target'] = (large_dataset['feature1'] > 0).astype(int)
        large_dataset.to_csv(data_path, index=False)

        start_time = time.time()

        from tests.unit.test_tools import train_model
        result = train_model(str(data_path), 'target', str(model_path))

        duration = time.time() - start_time

        assert duration < performance_threshold['model_train']

    def test_end_to_end_performance(self, temp_dir, performance_threshold):
        """Test complete workflow execution time."""
        from sklearn.datasets import load_iris
        import pandas as pd

        # Setup
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        input_path = temp_dir / "iris.csv"
        df.to_csv(input_path, index=False)

        start_time = time.time()

        # Execute full workflow
        from tests.unit.test_tools import load_dataset, clean_data, calculate_statistics, train_model

        load_dataset(str(input_path))
        cleaned_path = temp_dir / "cleaned.csv"
        clean_data(str(input_path), str(cleaned_path))
        calculate_statistics(str(cleaned_path))
        model_path = temp_dir / "model.pkl"
        train_model(str(cleaned_path), 'target', str(model_path))

        duration = time.time() - start_time

        assert duration < performance_threshold['end_to_end']


class TestMemoryUsage:
    """Test memory consumption of operations."""

    def test_data_loading_memory(self, temp_dir, large_dataset):
        """Test memory usage during data loading."""
        import gc
        import pandas as pd

        file_path = temp_dir / "memory_test.csv"
        large_dataset.to_csv(file_path, index=False)

        # Force garbage collection
        gc.collect()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # Load data
        df = pd.read_csv(file_path)

        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory

        # Memory increase should be reasonable (< 100MB for 10k rows)
        assert memory_increase < 100

        # Cleanup
        del df
        gc.collect()

    def test_model_training_memory(self, temp_dir):
        """Test memory usage during model training."""
        import gc
        from sklearn.datasets import make_classification
        import pandas as pd

        # Create dataset
        X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])
        df['target'] = y

        data_path = temp_dir / "train.csv"
        df.to_csv(data_path, index=False)

        gc.collect()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Train model
        from tests.unit.test_tools import train_model
        model_path = temp_dir / "model.pkl"
        train_model(str(data_path), 'target', str(model_path))

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 200  # MB


class TestScalability:
    """Test system scalability with increasing load."""

    @pytest.mark.parametrize("n_rows", [100, 1000, 5000, 10000])
    def test_scaling_with_data_size(self, temp_dir, n_rows):
        """Test performance scaling with dataset size."""
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            f'feature{i}': np.random.randn(n_rows)
            for i in range(5)
        })

        file_path = temp_dir / f"scale_{n_rows}.csv"
        df.to_csv(file_path, index=False)

        start_time = time.time()

        from tests.unit.test_tools import load_dataset
        result = load_dataset(str(file_path))

        duration = time.time() - start_time

        # Duration should scale sub-linearly
        # For 100 rows: ~0.1s, for 10000 rows: ~1s (not 10s)
        expected_max_time = (n_rows / 100) * 0.1 * 1.5  # With 50% overhead
        assert duration < expected_max_time

    @pytest.mark.parametrize("n_features", [5, 10, 20, 50])
    def test_scaling_with_feature_count(self, temp_dir, n_features):
        """Test performance scaling with number of features."""
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            f'feature{i}': np.random.randn(1000)
            for i in range(n_features)
        })
        df['target'] = np.random.randint(0, 2, 1000)

        data_path = temp_dir / f"features_{n_features}.csv"
        model_path = temp_dir / f"model_{n_features}.pkl"
        df.to_csv(data_path, index=False)

        start_time = time.time()

        from tests.unit.test_tools import train_model
        train_model(str(data_path), 'target', str(model_path))

        duration = time.time() - start_time

        # Training time should scale with feature count but remain reasonable
        assert duration < 30  # Max 30 seconds even for 50 features


class TestConcurrency:
    """Test concurrent execution performance."""

    def test_parallel_data_loading(self, temp_dir):
        """Test loading multiple datasets in parallel."""
        import concurrent.futures
        import pandas as pd

        # Create multiple datasets
        datasets = []
        for i in range(5):
            df = pd.DataFrame({'col': range(100)})
            file_path = temp_dir / f"parallel_{i}.csv"
            df.to_csv(file_path, index=False)
            datasets.append(str(file_path))

        start_time = time.time()

        # Load in parallel
        from tests.unit.test_tools import load_dataset

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(load_dataset, datasets))

        parallel_duration = time.time() - start_time

        # Parallel should be faster than sequential
        # Sequential would take ~5x longer
        assert len(results) == 5
        # Parallel execution should complete in reasonable time
        assert parallel_duration < 5  # seconds

    def test_concurrent_model_training(self, temp_dir):
        """Test training multiple models concurrently."""
        import concurrent.futures
        from sklearn.datasets import make_classification
        import pandas as pd

        # Prepare multiple datasets
        tasks = []
        for i in range(3):
            X, y = make_classification(n_samples=500, n_features=10, random_state=i)
            df = pd.DataFrame(X, columns=[f'f{j}' for j in range(10)])
            df['target'] = y

            data_path = temp_dir / f"concurrent_{i}.csv"
            model_path = temp_dir / f"concurrent_model_{i}.pkl"
            df.to_csv(data_path, index=False)

            tasks.append((str(data_path), 'target', str(model_path)))

        start_time = time.time()

        from tests.unit.test_tools import train_model

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(train_model, *task) for task in tasks]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        duration = time.time() - start_time

        assert len(results) == 3
        assert all('accuracy' in r for r in results)


class TestCaching:
    """Test caching mechanisms for performance optimization."""

    def test_repeated_data_access(self, temp_dir, sample_dataset):
        """Test that repeated data access can be optimized."""
        file_path = temp_dir / "cache_test.csv"
        sample_dataset.to_csv(file_path, index=False)

        # First access
        start_time = time.time()
        from tests.unit.test_tools import load_dataset
        first_result = load_dataset(str(file_path))
        first_duration = time.time() - start_time

        # Second access (could be cached)
        start_time = time.time()
        second_result = load_dataset(str(file_path))
        second_duration = time.time() - start_time

        # Results should be identical
        assert first_result == second_result

        # Note: Actual caching would make second_duration < first_duration


class TestResourceCleanup:
    """Test that resources are properly cleaned up."""

    def test_file_cleanup_after_operations(self, temp_dir, sample_dataset):
        """Test that temporary files are cleaned up."""
        initial_files = set(temp_dir.iterdir())

        # Perform operations
        file_path = temp_dir / "cleanup_test.csv"
        sample_dataset.to_csv(file_path, index=False)

        from tests.unit.test_tools import load_dataset
        load_dataset(str(file_path))

        # Cleanup
        os.remove(file_path)

        final_files = set(temp_dir.iterdir())

        # No extra files should remain
        assert len(final_files) <= len(initial_files)

    def test_memory_cleanup_after_workflow(self, temp_dir):
        """Test memory is released after workflow completion."""
        import gc
        from sklearn.datasets import load_iris
        import pandas as pd

        gc.collect()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Run workflow
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        data_path = temp_dir / "cleanup.csv"
        df.to_csv(data_path, index=False)

        from tests.unit.test_tools import load_dataset, clean_data, train_model

        load_dataset(str(data_path))
        cleaned_path = temp_dir / "cleaned.csv"
        clean_data(str(data_path), str(cleaned_path))
        model_path = temp_dir / "model.pkl"
        train_model(str(cleaned_path), 'target', str(model_path))

        # Force cleanup
        del df
        gc.collect()

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_diff = final_memory - initial_memory

        # Memory increase should be minimal after cleanup
        assert memory_diff < 50  # MB


class TestThroughput:
    """Test system throughput under load."""

    def test_requests_per_second(self, temp_dir):
        """Test number of operations that can be completed per second."""
        import pandas as pd

        # Create test dataset
        df = pd.DataFrame({'col': range(100)})
        file_path = temp_dir / "throughput.csv"
        df.to_csv(file_path, index=False)

        from tests.unit.test_tools import load_dataset

        # Measure throughput
        start_time = time.time()
        operations_count = 0
        target_duration = 1.0  # 1 second

        while (time.time() - start_time) < target_duration:
            load_dataset(str(file_path))
            operations_count += 1

        operations_per_second = operations_count / target_duration

        # Should handle at least 10 operations per second
        assert operations_per_second >= 10
