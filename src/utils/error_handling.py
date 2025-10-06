"""Error handling utilities with retry logic and circuit breakers."""

import time
import functools
from typing import Any, Callable, Optional, Type, Tuple
from enum import Enum

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from pybreaker import CircuitBreaker, CircuitBreakerError
from structlog import get_logger

logger = get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class MLFrameworkError(Exception):
    """Base exception for ML framework errors."""
    pass


class DataLoadError(MLFrameworkError):
    """Error loading data."""
    pass


class DataValidationError(MLFrameworkError):
    """Error validating data."""
    pass


class ModelTrainingError(MLFrameworkError):
    """Error during model training."""
    pass


class PredictionError(MLFrameworkError):
    """Error during prediction."""
    pass


def with_retry(
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_seconds: Initial backoff time in seconds
        max_backoff_seconds: Maximum backoff time
        exceptions: Tuple of exceptions to retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=backoff_seconds,
                max=max_backoff_seconds,
            ),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except RetryError as e:
                logger.error(
                    "retry_exhausted",
                    function=func.__name__,
                    attempts=max_attempts,
                    error=str(e),
                )
                raise e.last_attempt.exception()

        return wrapper

    return decorator


# Global circuit breakers registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    exclude: Optional[Tuple[Type[Exception], ...]] = None,
    name: Optional[str] = None,
) -> Callable:
    """
    Decorator for circuit breaker pattern.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        exclude: Exception types to exclude from counting as failures
        name: Circuit breaker name (defaults to function name)

    Returns:
        Decorated function with circuit breaker
    """
    def decorator(func: Callable) -> Callable:
        breaker_name = name or func.__name__

        # Create or get existing circuit breaker
        if breaker_name not in _circuit_breakers:
            cb_kwargs = {
                "fail_max": failure_threshold,
                "reset_timeout": recovery_timeout,
                "name": breaker_name,
            }
            if exclude is not None:
                cb_kwargs["exclude"] = exclude
            _circuit_breakers[breaker_name] = CircuitBreaker(**cb_kwargs)

        breaker = _circuit_breakers[breaker_name]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = breaker.call(func, *args, **kwargs)
                logger.debug(
                    "circuit_breaker_success",
                    function=func.__name__,
                    state=breaker.current_state,
                )
                return result
            except CircuitBreakerError as e:
                logger.error(
                    "circuit_breaker_open",
                    function=func.__name__,
                    state=breaker.current_state,
                    failure_count=breaker.fail_counter,
                )
                raise MLFrameworkError(f"Circuit breaker open for {func.__name__}") from e

        return wrapper

    return decorator


def get_circuit_breaker_state(name: str) -> Optional[str]:
    """
    Get the current state of a circuit breaker.

    Args:
        name: Circuit breaker name

    Returns:
        Current state or None if not found
    """
    if name in _circuit_breakers:
        return _circuit_breakers[name].current_state
    return None


def reset_circuit_breaker(name: str) -> bool:
    """
    Reset a circuit breaker to closed state.

    Args:
        name: Circuit breaker name

    Returns:
        True if reset successful, False if not found
    """
    if name in _circuit_breakers:
        _circuit_breakers[name].call_count = 0
        _circuit_breakers[name].fail_counter = 0
        _circuit_breakers[name]._state = CircuitBreaker.STATE_CLOSED
        logger.info("circuit_breaker_reset", name=name)
        return True
    return False


class FallbackModel:
    """Fallback model when primary model fails."""

    def __init__(self, model_type: str = "mean"):
        """
        Initialize fallback model.

        Args:
            model_type: Type of fallback ('mean', 'median', 'mode', 'zero')
        """
        self.model_type = model_type
        self.fallback_value = None
        logger.info("fallback_model_initialized", type=model_type)

    def fit(self, y: Any) -> None:
        """Fit fallback model."""
        if self.model_type == "mean":
            self.fallback_value = y.mean()
        elif self.model_type == "median":
            self.fallback_value = y.median()
        elif self.model_type == "mode":
            self.fallback_value = y.mode()[0] if not y.mode().empty else 0
        else:
            self.fallback_value = 0

        logger.info("fallback_model_fitted", value=self.fallback_value)

    def predict(self, X: Any) -> Any:
        """Generate fallback predictions."""
        import numpy as np
        predictions = np.full(len(X), self.fallback_value)
        logger.warning(
            "using_fallback_predictions",
            count=len(predictions),
            value=self.fallback_value,
        )
        return predictions


def with_fallback_model(fallback_type: str = "mean") -> Callable:
    """
    Decorator to provide fallback model on failure.

    Args:
        fallback_type: Type of fallback model

    Returns:
        Decorated function with fallback
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "primary_model_failed_using_fallback",
                    function=func.__name__,
                    error=str(e),
                    fallback_type=fallback_type,
                )

                # Attempt to use fallback
                fallback = FallbackModel(model_type=fallback_type)

                # Try to extract training data from args/kwargs
                if 'y_train' in kwargs:
                    fallback.fit(kwargs['y_train'])
                elif len(args) > 1:
                    fallback.fit(args[1])  # Assume second arg is y

                return fallback

        return wrapper

    return decorator
