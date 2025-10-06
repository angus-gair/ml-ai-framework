"""Data manipulation and loading tools."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
from structlog import get_logger

from ..models.schemas import DatasetMetadata, CleaningReport, CleaningOperation, ColumnInfo, DataType
from ..utils.error_handling import with_retry, circuit_breaker

logger = get_logger(__name__)


@with_retry(max_attempts=3, backoff_seconds=2.0)
@circuit_breaker(failure_threshold=5, recovery_timeout=60)
def load_dataset(file_path: str, **kwargs: Any) -> tuple[pd.DataFrame, DatasetMetadata]:
    """
    Load dataset from various file formats with metadata extraction.

    Args:
        file_path: Path to the dataset file
        **kwargs: Additional parameters for pandas read functions

    Returns:
        Tuple of (DataFrame, DatasetMetadata)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
    """
    start_time = time.time()
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    logger.info("loading_dataset", path=str(path), size_bytes=path.stat().st_size)

    # Determine file type and load
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path, **kwargs)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, **kwargs)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path, **kwargs)
        elif suffix == ".json":
            df = pd.read_json(file_path, **kwargs)
        elif suffix == ".feather":
            df = pd.read_feather(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Generate metadata
        metadata = _generate_metadata(df, path.name, str(path))

        duration = time.time() - start_time
        logger.info(
            "dataset_loaded",
            rows=df.shape[0],
            columns=df.shape[1],
            duration_seconds=round(duration, 2),
        )

        return df, metadata

    except Exception as e:
        logger.error("dataset_load_failed", path=str(path), error=str(e))
        raise


def _generate_metadata(df: pd.DataFrame, name: str, source_path: str) -> DatasetMetadata:
    """Generate comprehensive dataset metadata."""
    columns_info = []

    for col in df.columns:
        col_data = df[col]
        null_count = int(col_data.isnull().sum())
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0.0

        # Determine data type
        if pd.api.types.is_numeric_dtype(col_data):
            dtype = DataType.NUMERIC
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            dtype = DataType.DATETIME
        elif pd.api.types.is_bool_dtype(col_data):
            dtype = DataType.BOOLEAN
        elif pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
            dtype = DataType.CATEGORICAL if unique_ratio < 0.5 else DataType.TEXT
        else:
            dtype = DataType.TEXT

        columns_info.append(
            ColumnInfo(
                name=col,
                dtype=dtype,
                null_count=null_count,
                null_percentage=null_pct,
                unique_count=int(col_data.nunique()),
                memory_usage_bytes=int(col_data.memory_usage(deep=True)),
            )
        )

    return DatasetMetadata(
        name=name,
        shape=(len(df), len(df.columns)),
        columns=columns_info,
        total_memory_bytes=int(df.memory_usage(deep=True).sum()),
        source_path=source_path,
    )


@with_retry(max_attempts=3, backoff_seconds=1.0)
def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    handle_missing: str = "drop",
    missing_threshold: float = 0.5,
    outlier_method: Optional[str] = None,
    outlier_threshold: float = 3.0,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean dataset with comprehensive reporting.

    Args:
        df: Input DataFrame
        drop_duplicates: Whether to drop duplicate rows
        handle_missing: Strategy for missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode')
        missing_threshold: Drop columns with missing% above this threshold
        outlier_method: Outlier detection method ('iqr', 'zscore', None)
        outlier_threshold: Threshold for outlier detection

    Returns:
        Tuple of (cleaned DataFrame, CleaningReport)
    """
    start_time = time.time()
    operations = []

    rows_before = len(df)
    cols_before = len(df.columns)
    df_clean = df.copy()

    logger.info("starting_data_cleaning", rows=rows_before, columns=cols_before)

    # 1. Drop duplicates
    if drop_duplicates:
        duplicates_before = df_clean.duplicated().sum()
        if duplicates_before > 0:
            df_clean = df_clean.drop_duplicates()
            operations.append(
                CleaningOperation(
                    column="all",
                    operation="drop_duplicates",
                    affected_rows=int(duplicates_before),
                    details={"duplicates_removed": int(duplicates_before)},
                )
            )
            logger.info("duplicates_removed", count=int(duplicates_before))

    # 2. Handle columns with high missing percentage
    columns_to_drop = []
    for col in df_clean.columns:
        missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100
        if missing_pct > (missing_threshold * 100):
            columns_to_drop.append(col)
            operations.append(
                CleaningOperation(
                    column=col,
                    operation="drop_column_high_missing",
                    affected_rows=len(df_clean),
                    details={"missing_percentage": round(missing_pct, 2)},
                )
            )

    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
        logger.info("high_missing_columns_dropped", columns=columns_to_drop)

    # 3. Handle missing values in remaining columns
    for col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            if handle_missing == "drop":
                df_clean = df_clean.dropna(subset=[col])
                operations.append(
                    CleaningOperation(
                        column=col,
                        operation="drop_missing",
                        affected_rows=int(missing_count),
                        details={"missing_values": int(missing_count)},
                    )
                )
            elif handle_missing == "fill_mean" and pd.api.types.is_numeric_dtype(df_clean[col]):
                fill_value = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(fill_value)
                operations.append(
                    CleaningOperation(
                        column=col,
                        operation="fill_mean",
                        affected_rows=int(missing_count),
                        details={"fill_value": float(fill_value)},
                    )
                )
            elif handle_missing == "fill_median" and pd.api.types.is_numeric_dtype(df_clean[col]):
                fill_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_value)
                operations.append(
                    CleaningOperation(
                        column=col,
                        operation="fill_median",
                        affected_rows=int(missing_count),
                        details={"fill_value": float(fill_value)},
                    )
                )
            elif handle_missing == "fill_mode":
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else None
                if fill_value is not None:
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    operations.append(
                        CleaningOperation(
                            column=col,
                            operation="fill_mode",
                            affected_rows=int(missing_count),
                            details={"fill_value": str(fill_value)},
                        )
                    )

    # 4. Handle outliers
    if outlier_method:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers_mask = _detect_outliers(
                df_clean[col], method=outlier_method, threshold=outlier_threshold
            )
            outliers_count = outliers_mask.sum()

            if outliers_count > 0:
                df_clean = df_clean[~outliers_mask]
                operations.append(
                    CleaningOperation(
                        column=col,
                        operation=f"remove_outliers_{outlier_method}",
                        affected_rows=int(outliers_count),
                        details={
                            "method": outlier_method,
                            "threshold": outlier_threshold,
                            "outliers_removed": int(outliers_count),
                        },
                    )
                )

    duration = time.time() - start_time

    report = CleaningReport(
        operations=operations,
        rows_before=rows_before,
        rows_after=len(df_clean),
        columns_before=cols_before,
        columns_after=len(df_clean.columns),
        duration_seconds=round(duration, 2),
    )

    logger.info(
        "cleaning_completed",
        rows_removed=report.rows_removed,
        columns_removed=cols_before - len(df_clean.columns),
        duration_seconds=round(duration, 2),
    )

    return df_clean, report


def _detect_outliers(series: pd.Series, method: str, threshold: float) -> pd.Series:
    """Detect outliers using specified method."""
    if method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    elif method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    else:
        return pd.Series([False] * len(series))


def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary of statistical measures
    """
    logger.info("calculating_statistics", rows=len(df), columns=len(df.columns))

    stats = {
        "shape": df.shape,
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(df.select_dtypes(include=["object", "category"]).columns),
        "total_missing": int(df.isnull().sum().sum()),
        "missing_percentage": round((df.isnull().sum().sum() / df.size) * 100, 2),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "duplicates": int(df.duplicated().sum()),
    }

    # Numeric column statistics
    numeric_stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        numeric_stats[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "skew": float(df[col].skew()),
            "kurtosis": float(df[col].kurtosis()),
        }

    stats["numeric_statistics"] = numeric_stats

    logger.info("statistics_calculated", total_missing=stats["total_missing"])

    return stats
