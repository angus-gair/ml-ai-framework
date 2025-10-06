"""Structured logging configuration with structlog."""

import sys
import logging
from typing import Any, Dict

import structlog
from structlog.processors import TimeStamper, JSONRenderer, add_log_level
from structlog.stdlib import BoundLogger


def setup_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    include_caller: bool = True,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output logs in JSON format
        include_caller: Whether to include caller information
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Build processor chain
    processors = [
        structlog.contextvars.merge_contextvars,
        add_log_level,
        TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if include_caller:
        processors.append(structlog.processors.CallsiteParameterAdder([
            structlog.processors.CallsiteParameter.FILENAME,
            structlog.processors.CallsiteParameter.FUNC_NAME,
            structlog.processors.CallsiteParameter.LINENO,
        ]))

    # Add appropriate renderer
    if json_logs:
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None, **initial_values: Any) -> BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)
        **initial_values: Initial context values to bind

    Returns:
        Configured structured logger
    """
    logger = structlog.get_logger(name)

    if initial_values:
        logger = logger.bind(**initial_values)

    return logger


def add_context(**kwargs: Any) -> None:
    """
    Add context variables to all subsequent log messages.

    Args:
        **kwargs: Context key-value pairs
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context(*keys: str) -> None:
    """
    Clear specific context variables.

    Args:
        *keys: Context keys to clear
    """
    structlog.contextvars.unbind_contextvars(*keys)


def log_task_execution(task_id: str, task_type: str, agent_id: str = None) -> Dict[str, Any]:
    """
    Create context for task execution logging.

    Args:
        task_id: Unique task identifier
        task_type: Type of task
        agent_id: Agent executing the task

    Returns:
        Context dictionary
    """
    context = {
        "task_id": task_id,
        "task_type": task_type,
    }

    if agent_id:
        context["agent_id"] = agent_id

    add_context(**context)
    return context
