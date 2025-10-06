"""Quick Start 02: Custom Tools

Learn how to create and use custom tools in your ML workflows.

What you'll learn:
- Creating custom data processing tools
- Integrating tools with agents
- Tool function signatures
- Error handling in tools
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Tuple
from structlog import get_logger

logger = get_logger(__name__)


# Custom Tool 1: Feature Engineering
def create_polynomial_features(
    df: pd.DataFrame,
    degree: int = 2,
    include_bias: bool = False,
) -> pd.DataFrame:
    """
    Create polynomial features from numeric columns.

    This is a custom tool that extends the framework's capabilities.

    Args:
        df: Input DataFrame
        degree: Polynomial degree (2 = squared features, 3 = cubed, etc.)
        include_bias: Whether to include bias term

    Returns:
        DataFrame with polynomial features added
    """
    logger.info("creating_polynomial_features", degree=degree)

    df_poly = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        for d in range(2, degree + 1):
            new_col = f"{col}_power_{d}"
            df_poly[new_col] = df[col] ** d

    logger.info(
        "polynomial_features_created",
        original_features=len(df.columns),
        new_features=len(df_poly.columns),
    )

    return df_poly


# Custom Tool 2: Outlier Detection
def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: list = None,
    multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, dict]:
    """
    Detect outliers using IQR method.

    Custom tool for advanced outlier detection.

    Args:
        df: Input DataFrame
        columns: Columns to check (None = all numeric)
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme)

    Returns:
        Tuple of (outlier flags DataFrame, statistics dict)
    """
    logger.info("detecting_outliers", method="iqr", multiplier=multiplier)

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_flags = pd.DataFrame(index=df.index)
    stats = {}

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_flags[f"{col}_outlier"] = outliers

        stats[col] = {
            "count": int(outliers.sum()),
            "percentage": float((outliers.sum() / len(df)) * 100),
            "bounds": (float(lower_bound), float(upper_bound)),
        }

    logger.info("outliers_detected", total_columns=len(columns))

    return outlier_flags, stats


# Custom Tool 3: Feature Importance Extractor
def extract_feature_importance(
    model: any,
    feature_names: list,
    top_n: int = 10,
) -> dict:
    """
    Extract and rank feature importance from trained model.

    Custom tool for analyzing model insights.

    Args:
        model: Trained sklearn model
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        Dict of {feature: importance} sorted by importance
    """
    logger.info("extracting_feature_importance", features=len(feature_names))

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        logger.warning("model_has_no_feature_importance")
        return {}

    # Create feature-importance pairs
    feature_importance = dict(zip(feature_names, importances))

    # Sort by importance
    sorted_features = dict(
        sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
    )

    logger.info("feature_importance_extracted", top_features=len(sorted_features))

    return sorted_features


def demonstrate_custom_tools():
    """Demonstrate using custom tools."""

    print("Quick Start 02: Custom Tools")
    print("=" * 60)
    print()

    # Create sample data
    print("Creating sample dataset...")
    from sklearn.datasets import load_boston
    import warnings
    warnings.filterwarnings('ignore')

    # Use California housing instead (Boston is deprecated)
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target

    print(f"✓ Loaded {len(df)} samples")
    print()

    # Tool 1: Polynomial Features
    print("Tool 1: Creating Polynomial Features")
    print("-" * 60)
    df_poly = create_polynomial_features(df.iloc[:, :3], degree=2)
    print(f"Original features: {len(df.iloc[:, :3].columns)}")
    print(f"After polynomial: {len(df_poly.columns)}")
    print(f"New features: {list(df_poly.columns[3:])}")
    print()

    # Tool 2: Outlier Detection
    print("Tool 2: Detecting Outliers (IQR Method)")
    print("-" * 60)
    outlier_flags, stats = detect_outliers_iqr(df, columns=['MedInc', 'HouseAge'])

    for col, stat in stats.items():
        print(f"{col}:")
        print(f"  Outliers: {stat['count']} ({stat['percentage']:.2f}%)")
        print(f"  Bounds: {stat['bounds'][0]:.2f} to {stat['bounds'][1]:.2f}")
    print()

    # Tool 3: Feature Importance (requires trained model)
    print("Tool 3: Feature Importance Analysis")
    print("-" * 60)
    print("Training a quick model for demonstration...")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    importance = extract_feature_importance(model, X.columns.tolist(), top_n=5)

    print("Top 5 Most Important Features:")
    for i, (feature, score) in enumerate(importance.items(), 1):
        bar = "█" * int(score * 50)
        print(f"  {i}. {feature:15s} {bar} {score:.4f}")
    print()

    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("  1. Custom tools are just Python functions")
    print("  2. Use type hints for clarity")
    print("  3. Include logging for debugging")
    print("  4. Return structured data (dicts, DataFrames)")
    print("  5. Handle errors gracefully")
    print()
    print("Next: Try 03_error_handling.py to learn error patterns")


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging(log_level="INFO", log_to_file=False)

    demonstrate_custom_tools()
