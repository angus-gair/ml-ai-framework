"""
Sample datasets and test data generators.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def generate_iris_dataset(output_path: Path = None):
    """Generate Iris-like classification dataset."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def generate_regression_dataset(n_samples=500, n_features=10, output_path: Path = None):
    """Generate synthetic regression dataset."""
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(5, n_features // 2),
        noise=10.0,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(n_features)])
    df['target'] = y

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def generate_classification_dataset(n_samples=1000, n_features=20, n_classes=2, output_path: Path = None):
    """Generate synthetic classification dataset."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=min(n_features, n_classes * 3),
        n_redundant=max(0, n_features - n_classes * 3 - 2),
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(n_features)])
    df['target'] = y

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def generate_dataset_with_missing_values(missing_rate=0.2, output_path: Path = None):
    """Generate dataset with missing values."""
    np.random.seed(42)

    n_rows = 200
    df = pd.DataFrame({
        'numeric1': np.random.randn(n_rows),
        'numeric2': np.random.randn(n_rows),
        'categorical1': np.random.choice(['A', 'B', 'C'], n_rows),
        'categorical2': np.random.choice(['X', 'Y', 'Z'], n_rows),
        'target': np.random.randint(0, 2, n_rows)
    })

    # Inject missing values
    for col in df.columns:
        if col != 'target':
            mask = np.random.random(n_rows) < missing_rate
            df.loc[mask, col] = np.nan

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def generate_dataset_with_outliers(outlier_count=20, output_path: Path = None):
    """Generate dataset with statistical outliers."""
    np.random.seed(42)

    n_rows = 200
    normal_data = np.random.randn(n_rows - outlier_count, 5)

    # Add extreme outliers
    outlier_data = np.random.choice([-10, -8, 8, 10], size=(outlier_count, 5))

    combined_data = np.vstack([normal_data, outlier_data])
    np.random.shuffle(combined_data)

    df = pd.DataFrame(combined_data, columns=[f'feature{i}' for i in range(5)])
    df['target'] = np.random.randint(0, 2, n_rows)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def generate_imbalanced_dataset(imbalance_ratio=0.1, output_path: Path = None):
    """Generate imbalanced classification dataset."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(10)])
    df['target'] = y

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def generate_high_correlation_dataset(output_path: Path = None):
    """Generate dataset with known high correlations."""
    np.random.seed(42)

    n_rows = 500
    feature1 = np.random.randn(n_rows)
    feature2 = feature1 + np.random.randn(n_rows) * 0.1  # High correlation
    feature3 = -feature1 + np.random.randn(n_rows) * 0.1  # High negative correlation
    feature4 = np.random.randn(n_rows)  # Independent

    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'target': (feature1 + feature2 > 0).astype(int)
    })

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def generate_time_series_dataset(n_points=1000, output_path: Path = None):
    """Generate time series dataset."""
    np.random.seed(42)

    dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
    trend = np.linspace(0, 10, n_points)
    seasonality = 5 * np.sin(2 * np.pi * np.arange(n_points) / 365)
    noise = np.random.randn(n_points) * 2

    values = trend + seasonality + noise

    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'trend': trend,
        'seasonality': seasonality
    })

    if output_path:
        df.to_csv(output_path, index=False)

    return df
