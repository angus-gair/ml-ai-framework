"""Machine learning model training and evaluation tools."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from structlog import get_logger

from ..models.schemas import ModelPerformance, ModelMetrics, ModelType
from ..utils.error_handling import with_retry, circuit_breaker

logger = get_logger(__name__)


@with_retry(max_attempts=3, backoff_seconds=2.0)
@circuit_breaker(failure_threshold=5, recovery_timeout=60)
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "classification",
    algorithm: str = "random_forest",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    hyperparameters: Optional[Dict[str, Any]] = None,
    cross_validation: bool = True,
    cv_folds: int = 5,
) -> Tuple[Any, ModelPerformance]:
    """
    Train ML model with comprehensive performance tracking.

    Args:
        X: Feature matrix
        y: Target variable
        model_type: 'classification' or 'regression'
        algorithm: Model algorithm to use
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
        hyperparameters: Model hyperparameters
        cross_validation: Whether to perform cross-validation
        cv_folds: Number of CV folds

    Returns:
        Tuple of (trained model, ModelPerformance)
    """
    start_time = time.time()
    hyperparameters = hyperparameters or {}

    logger.info(
        "starting_model_training",
        model_type=model_type,
        algorithm=algorithm,
        features=X.shape[1],
        samples=X.shape[0],
    )

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    val_proportion = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_proportion, random_state=random_state
    )

    logger.info(
        "data_split",
        train_size=len(X_train),
        val_size=len(X_val),
        test_size=len(X_test),
    )

    # Initialize model
    model = _get_model(model_type, algorithm, hyperparameters)

    # Train model
    model.fit(X_train, y_train)

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_metrics = _calculate_metrics(y_train, y_train_pred, model_type, model, X_train)
    val_metrics = _calculate_metrics(y_val, y_val_pred, model_type, model, X_val)
    test_metrics = _calculate_metrics(y_test, y_test_pred, model_type, model, X_test)

    # Cross-validation
    cv_scores = None
    if cross_validation:
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring=_get_cv_scoring(model_type)
        )
        logger.info(
            "cross_validation_completed",
            mean_score=round(float(cv_scores.mean()), 4),
            std_score=round(float(cv_scores.std()), 4),
        )

    # Feature importance
    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = dict(zip(X.columns, model.feature_importances_.tolist()))
    elif hasattr(model, "coef_"):
        feature_importance = dict(zip(X.columns, np.abs(model.coef_).tolist()))

    duration = time.time() - start_time

    # Create performance report
    performance = ModelPerformance(
        model_type=ModelType.CLASSIFICATION if model_type == "classification" else ModelType.REGRESSION,
        model_name=algorithm,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        hyperparameters=hyperparameters,
        training_duration_seconds=round(duration, 2),
        feature_names=X.columns.tolist(),
        feature_importance=feature_importance,
        cross_validation_scores=[float(score) for score in cv_scores] if cv_scores is not None else None,
    )

    logger.info(
        "model_training_completed",
        algorithm=algorithm,
        train_score=train_metrics.accuracy or train_metrics.r2_score,
        val_score=val_metrics.accuracy or val_metrics.r2_score,
        duration_seconds=round(duration, 2),
    )

    return model, performance


def _get_model(model_type: str, algorithm: str, hyperparameters: Dict[str, Any]) -> Any:
    """Get model instance based on type and algorithm."""
    if model_type == "classification":
        if algorithm == "random_forest":
            return RandomForestClassifier(**hyperparameters)
        elif algorithm == "logistic_regression":
            return LogisticRegression(**hyperparameters)
        else:
            raise ValueError(f"Unsupported classification algorithm: {algorithm}")
    elif model_type == "regression":
        if algorithm == "random_forest":
            return RandomForestRegressor(**hyperparameters)
        elif algorithm == "linear_regression":
            return LinearRegression(**hyperparameters)
        else:
            raise ValueError(f"Unsupported regression algorithm: {algorithm}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: str,
    model: Any,
    X: pd.DataFrame,
) -> ModelMetrics:
    """Calculate metrics based on model type."""
    if model_type == "classification":
        # For binary classification
        try:
            y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
            auc = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
        except Exception:
            auc = None

        return ModelMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            f1_score=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            auc_roc=float(auc) if auc is not None else None,
        )
    else:  # regression
        mse = mean_squared_error(y_true, y_pred)
        return ModelMetrics(
            mse=float(mse),
            rmse=float(np.sqrt(mse)),
            mae=float(mean_absolute_error(y_true, y_pred)),
            r2_score=float(r2_score(y_true, y_pred)),
        )


def _get_cv_scoring(model_type: str) -> str:
    """Get scoring metric for cross-validation."""
    return "accuracy" if model_type == "classification" else "r2"


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "classification",
) -> ModelMetrics:
    """
    Evaluate trained model on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_type: 'classification' or 'regression'

    Returns:
        ModelMetrics with test performance
    """
    logger.info("evaluating_model", test_samples=len(X_test))

    y_pred = model.predict(X_test)
    metrics = _calculate_metrics(y_test, y_pred, model_type, model, X_test)

    logger.info(
        "evaluation_completed",
        primary_metric=metrics.accuracy or metrics.r2_score,
    )

    return metrics


def generate_predictions(
    model: Any,
    X: pd.DataFrame,
    return_probabilities: bool = False,
) -> np.ndarray:
    """
    Generate predictions from trained model.

    Args:
        model: Trained model
        X: Feature matrix
        return_probabilities: Whether to return probabilities (classification only)

    Returns:
        Predictions array
    """
    logger.info("generating_predictions", samples=len(X))

    if return_probabilities and hasattr(model, "predict_proba"):
        predictions = model.predict_proba(X)
    else:
        predictions = model.predict(X)

    logger.info("predictions_generated", count=len(predictions))

    return predictions
