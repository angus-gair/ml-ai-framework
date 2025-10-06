"""Quick Start 03: Error Handling

Learn robust error handling patterns for ML workflows.

What you'll learn:
- Try-except patterns
- Retry mechanisms
- Circuit breaker pattern
- Logging errors
- Graceful degradation
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import time
from typing import Any, Dict
from functools import wraps
from structlog import get_logger

logger = get_logger(__name__)


# Pattern 1: Basic Try-Except
def safe_load_data(file_path: str) -> Dict[str, Any]:
    """
    Safely load data with error handling.

    Pattern: Basic try-except with specific exceptions
    """
    try:
        import pandas as pd
        df = pd.read_csv(file_path)

        logger.info("data_loaded_successfully", path=file_path, rows=len(df))

        return {"status": "success", "data": df}

    except FileNotFoundError:
        logger.error("file_not_found", path=file_path)
        return {"status": "error", "error": "File not found", "data": None}

    except pd.errors.EmptyDataError:
        logger.error("empty_data", path=file_path)
        return {"status": "error", "error": "Empty file", "data": None}

    except Exception as e:
        logger.error("unexpected_error", path=file_path, error=str(e))
        return {"status": "error", "error": str(e), "data": None}


# Pattern 2: Retry Decorator
def retry(max_attempts: int = 3, delay_seconds: float = 1.0):
    """
    Retry decorator for transient failures.

    Pattern: Exponential backoff with retry limit
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    return result

                except Exception as e:
                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error=str(e),
                    )

                    if attempt == max_attempts:
                        logger.error("max_retries_exceeded", function=func.__name__)
                        raise

                    # Exponential backoff
                    wait_time = delay_seconds * (2 ** (attempt - 1))
                    logger.info("retry_waiting", seconds=wait_time)
                    time.sleep(wait_time)

        return wrapper
    return decorator


# Pattern 3: Circuit Breaker
class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.

    States:
    - CLOSED: Normal operation
    - OPEN: Failures exceeded threshold, reject requests
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""

        # Check if we should attempt recovery
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info("circuit_breaker_half_open", function=func.__name__)
                self.state = "HALF_OPEN"
            else:
                logger.warning("circuit_breaker_open", function=func.__name__)
                raise Exception("Circuit breaker OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)

            # Success - reset if we were recovering
            if self.state == "HALF_OPEN":
                logger.info("circuit_breaker_recovered", function=func.__name__)
                self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            logger.error(
                "circuit_breaker_failure",
                function=func.__name__,
                failures=self.failure_count,
                threshold=self.failure_threshold,
            )

            # Open circuit if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning("circuit_breaker_opened", function=func.__name__)

            raise


# Pattern 4: Graceful Degradation
def train_model_with_fallback(X, y, preferred_model="complex", **kwargs):
    """
    Train model with fallback to simpler model if complex one fails.

    Pattern: Graceful degradation
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if preferred_model == "complex":
        try:
            logger.info("attempting_complex_model")
            model = RandomForestClassifier(n_estimators=200, max_depth=20, **kwargs)
            model.fit(X, y)
            logger.info("complex_model_succeeded")
            return model, "complex"

        except Exception as e:
            logger.warning("complex_model_failed", error=str(e))
            logger.info("falling_back_to_simple_model")

    # Fallback to simple model
    try:
        model = LogisticRegression(max_iter=1000, **kwargs)
        model.fit(X, y)
        logger.info("simple_model_succeeded")
        return model, "simple"

    except Exception as e:
        logger.error("all_models_failed", error=str(e))
        raise


# Pattern 5: Validation and Early Exit
def validate_and_process(df, required_columns: list):
    """
    Validate input before processing.

    Pattern: Fail fast with validation
    """
    import pandas as pd

    # Type validation
    if not isinstance(df, pd.DataFrame):
        error_msg = f"Expected DataFrame, got {type(df)}"
        logger.error("invalid_input_type", expected="DataFrame", got=type(df).__name__)
        raise TypeError(error_msg)

    # Column validation
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error("missing_columns", columns=list(missing_cols))
        raise ValueError(error_msg)

    # Empty check
    if len(df) == 0:
        logger.error("empty_dataframe")
        raise ValueError("DataFrame is empty")

    # Null check
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning("null_values_detected", counts=null_counts.to_dict())

    logger.info("validation_passed", rows=len(df), columns=len(df.columns))

    return True


def demonstrate_error_handling():
    """Demonstrate error handling patterns."""

    print("Quick Start 03: Error Handling Patterns")
    print("=" * 60)
    print()

    # Pattern 1: Basic Try-Except
    print("Pattern 1: Basic Try-Except")
    print("-" * 60)

    result = safe_load_data("/tmp/nonexistent.csv")
    print(f"Loading nonexistent file: {result['status']}")
    print(f"  Error: {result.get('error')}")
    print()

    # Pattern 2: Retry Decorator
    print("Pattern 2: Retry with Backoff")
    print("-" * 60)

    call_count = 0

    @retry(max_attempts=3, delay_seconds=0.5)
    def unstable_function():
        nonlocal call_count
        call_count += 1
        print(f"  Attempt {call_count}...")

        if call_count < 3:
            raise ConnectionError("Temporary failure")

        return "Success!"

    try:
        result = unstable_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed after retries: {e}")
    print()

    # Pattern 3: Circuit Breaker
    print("Pattern 3: Circuit Breaker")
    print("-" * 60)

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=2)

    def failing_service():
        raise Exception("Service unavailable")

    # Trip the circuit
    for i in range(1, 4):
        try:
            breaker.call(failing_service)
        except Exception:
            print(f"  Failure {i} - State: {breaker.state}")

    # Try again - should be rejected
    try:
        breaker.call(failing_service)
    except Exception as e:
        print(f"  Request rejected: {str(e)[:50]}...")
    print()

    # Pattern 4: Graceful Degradation
    print("Pattern 4: Graceful Degradation")
    print("-" * 60)

    from sklearn.datasets import make_classification
    import numpy as np

    # Create dataset that might challenge complex model
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    model, model_type = train_model_with_fallback(X, y, preferred_model="complex")
    print(f"Model used: {model_type}")
    print(f"  Accuracy on training: {model.score(X, y):.2%}")
    print()

    # Pattern 5: Validation
    print("Pattern 5: Input Validation")
    print("-" * 60)

    import pandas as pd

    df_valid = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })

    try:
        validate_and_process(df_valid, required_columns=['feature1', 'feature2', 'target'])
        print("  ✓ Validation passed")
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")

    # Test with invalid input
    try:
        validate_and_process(df_valid, required_columns=['feature1', 'missing_col'])
        print("  ✓ Validation passed")
    except ValueError as e:
        print(f"  ✗ Validation failed (expected): {e}")
    print()

    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("  1. Always handle expected exceptions explicitly")
    print("  2. Use retry for transient failures")
    print("  3. Circuit breakers prevent cascading failures")
    print("  4. Graceful degradation maintains functionality")
    print("  5. Validate inputs early (fail fast)")
    print("  6. Log errors with context for debugging")
    print()
    print("Next: Try 04_testing.py to learn testing patterns")


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging(log_level="INFO", log_to_file=False)

    demonstrate_error_handling()
