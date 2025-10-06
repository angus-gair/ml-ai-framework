"""Application settings using Pydantic Settings."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="Default OpenAI model")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")

    # Server Configuration
    server_host: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    server_reload: bool = Field(default=True, description="Enable auto-reload")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_json: bool = Field(default=True, description="Use JSON logging")
    log_include_caller: bool = Field(default=True, description="Include caller info")

    # Workflow Configuration
    max_agent_iterations: int = Field(default=10, ge=1, description="Max agent iterations")
    default_workflow_type: str = Field(default="crewai", description="Default workflow type")

    # Error Handling Configuration
    retry_max_attempts: int = Field(default=3, ge=1, description="Max retry attempts")
    retry_backoff_seconds: float = Field(default=2.0, ge=0.1, description="Retry backoff")
    circuit_breaker_threshold: int = Field(default=5, ge=1, description="Circuit breaker threshold")
    circuit_breaker_timeout: int = Field(default=60, ge=1, description="Circuit breaker timeout")

    # Data Processing Configuration
    missing_value_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Missing value threshold")
    outlier_method: str = Field(default="iqr", description="Outlier detection method")
    outlier_threshold: float = Field(default=1.5, ge=0.1, description="Outlier threshold")

    # Model Training Configuration
    default_test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size")
    default_val_size: float = Field(default=0.1, ge=0.05, le=0.3, description="Validation set size")
    random_state: int = Field(default=42, description="Random seed")
    cross_validation_folds: int = Field(default=5, ge=2, le=10, description="CV folds")

    # Performance Configuration
    max_workers: int = Field(default=4, ge=1, description="Max worker threads")
    async_enabled: bool = Field(default=True, description="Enable async operations")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Application settings
    """
    return Settings()
