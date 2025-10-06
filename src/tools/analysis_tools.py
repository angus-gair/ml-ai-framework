"""Exploratory data analysis and statistical tools."""

from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from structlog import get_logger

from ..models.schemas import (
    EDAFindings,
    StatisticalSummary,
    CorrelationPair,
)
from ..utils.error_handling import with_retry

logger = get_logger(__name__)


@with_retry(max_attempts=2, backoff_seconds=1.0)
def perform_eda(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    correlation_threshold: float = 0.7,
) -> EDAFindings:
    """
    Perform comprehensive exploratory data analysis.

    Args:
        df: Input DataFrame
        target_column: Target variable for supervised learning
        correlation_threshold: Minimum correlation to report

    Returns:
        EDAFindings with comprehensive analysis
    """
    logger.info("starting_eda", rows=len(df), columns=len(df.columns))

    # Statistical summaries
    statistical_summary = []
    for col in df.columns:
        summary = _generate_column_summary(df[col], col)
        statistical_summary.append(summary)

    # Correlations
    correlations = _calculate_correlations(df, correlation_threshold)

    # Outlier detection
    outliers_detected = detect_outliers(df, method="iqr")

    # Distribution insights
    distribution_insights = _analyze_distributions(df)

    # Missing value patterns
    missing_patterns = _analyze_missing_patterns(df)

    # Preliminary feature importance (if target provided)
    feature_importance = None
    if target_column and target_column in df.columns:
        feature_importance = _calculate_preliminary_importance(df, target_column)

    findings = EDAFindings(
        statistical_summary=statistical_summary,
        correlations=correlations,
        outliers_detected=outliers_detected,
        distribution_insights=distribution_insights,
        missing_value_patterns=missing_patterns,
        feature_importance_preliminary=feature_importance,
    )

    logger.info(
        "eda_completed",
        total_outliers=findings.total_outliers,
        high_correlations=len(correlations),
    )

    return findings


def _generate_column_summary(series: pd.Series, col_name: str) -> StatisticalSummary:
    """Generate statistical summary for a column."""
    summary_dict = {
        "column": col_name,
        "count": int(series.count()),
    }

    if pd.api.types.is_numeric_dtype(series):
        summary_dict.update({
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "q25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        })

    # Mode for all types
    mode_values = series.mode()
    if not mode_values.empty:
        summary_dict["mode"] = mode_values.iloc[0]

    return StatisticalSummary(**summary_dict)


def _calculate_correlations(df: pd.DataFrame, threshold: float) -> List[CorrelationPair]:
    """Calculate significant correlations between numeric columns."""
    correlations = []
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        return correlations

    corr_matrix = numeric_df.corr()

    for i, col1 in enumerate(numeric_df.columns):
        for j, col2 in enumerate(numeric_df.columns):
            if i < j:  # Avoid duplicates and self-correlation
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) >= threshold:
                    # Calculate p-value
                    _, p_value = stats.pearsonr(numeric_df[col1].dropna(), numeric_df[col2].dropna())

                    correlations.append(
                        CorrelationPair(
                            feature1=col1,
                            feature2=col2,
                            correlation=round(float(corr_value), 4),
                            significance=round(float(p_value), 4),
                        )
                    )

    return sorted(correlations, key=lambda x: abs(x.correlation), reverse=True)


def detect_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> Dict[str, int]:
    """
    Detect outliers in numeric columns.

    Args:
        df: Input DataFrame
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        Dictionary mapping column names to outlier counts
    """
    logger.info("detecting_outliers", method=method, threshold=threshold)

    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_mask = z_scores > threshold
        else:
            continue

        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            outliers[col] = outlier_count

    total_outliers = sum(outliers.values())
    logger.info("outliers_detected", total=total_outliers, columns_affected=len(outliers))

    return outliers


def analyze_correlations(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    method: str = "pearson",
) -> Dict[str, float]:
    """
    Analyze correlations with target variable.

    Args:
        df: Input DataFrame
        target_column: Target variable name
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Dictionary of feature-target correlations
    """
    if target_column is None or target_column not in df.columns:
        logger.warning("target_column_not_found", target=target_column)
        return {}

    logger.info("analyzing_correlations", target=target_column, method=method)

    numeric_df = df.select_dtypes(include=[np.number])

    if target_column not in numeric_df.columns:
        logger.warning("target_not_numeric", target=target_column)
        return {}

    correlations = numeric_df.corr(method=method)[target_column].drop(target_column)

    return {col: round(float(val), 4) for col, val in correlations.items()}


def _analyze_distributions(df: pd.DataFrame) -> Dict[str, str]:
    """Analyze distribution characteristics of numeric columns."""
    insights = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()

        if abs(skewness) < 0.5:
            dist_type = "normal"
        elif skewness > 0.5:
            dist_type = "right_skewed"
        else:
            dist_type = "left_skewed"

        if kurtosis > 3:
            dist_type += "_heavy_tailed"
        elif kurtosis < 3:
            dist_type += "_light_tailed"

        insights[col] = dist_type

    return insights


def _analyze_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze patterns in missing data."""
    patterns = {
        "total_missing": int(df.isnull().sum().sum()),
        "columns_with_missing": [],
        "rows_with_missing": int(df.isnull().any(axis=1).sum()),
    }

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            patterns["columns_with_missing"].append({
                "column": col,
                "count": int(missing_count),
                "percentage": round((missing_count / len(df)) * 100, 2),
            })

    return patterns


def _calculate_preliminary_importance(df: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """Calculate preliminary feature importance using correlation."""
    numeric_df = df.select_dtypes(include=[np.number])

    if target_column not in numeric_df.columns:
        return {}

    correlations = numeric_df.corr()[target_column].drop(target_column)

    # Normalize absolute correlations to get importance scores
    abs_corr = correlations.abs()
    if abs_corr.sum() > 0:
        importance = abs_corr / abs_corr.sum()
        return {col: round(float(val), 4) for col, val in importance.items()}

    return {}
