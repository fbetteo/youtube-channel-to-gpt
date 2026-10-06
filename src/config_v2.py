"""
Configuration settings for YouTube Transcript API (Pydantic v2 compatible)
"""

import os
from typing import Dict, List
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """API settings loaded from environment variables"""

    # API settings
    api_title: str = Field(default="YouTube Transcript API")
    api_version: str = Field(default="1.0.0")
    api_key: str = Field(
        default_factory=lambda: os.getenv("TRANSCRIPT_API_KEY", "default_dev_key")
    )
    api_base_url: str = Field(
        default_factory=lambda: os.getenv("API_BASE_URL", "http://localhost:8000")
    )

    # YouTube API settings
    youtube_api_key: str = Field(
        default_factory=lambda: os.getenv("YOUTUBE_API_KEY", "")
    )

    # Proxy settings for YouTube Transcript API
    webshare_proxy_username: str = Field(
        default_factory=lambda: os.getenv("WEBSHARE_PROXY_USERNAME", "")
    )
    webshare_proxy_password: str = Field(
        default_factory=lambda: os.getenv("WEBSHARE_PROXY_PASSWORD", "")
    )

    # File storage settings
    temp_dir: str = Field(
        default_factory=lambda: os.getenv("TEMP_DIR", "../build/transcripts")
    )

    # AWS/S3 settings for Lambda integration
    aws_access_key_id: str = Field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", "")
    )
    aws_secret_access_key: str = Field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", "")
    )
    aws_default_region: str = Field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    s3_bucket_name: str = Field(default_factory=lambda: os.getenv("S3_BUCKET_NAME", ""))

    # Lambda settings
    lambda_function_name: str = Field(
        default_factory=lambda: os.getenv(
            "LAMBDA_FUNCTION_NAME", "youtube-transcript-processor"
        )
    )
    lambda_results_queue_url: str = Field(
        default_factory=lambda: os.getenv("LAMBDA_RESULTS_QUEUE_URL", "")
    )

    # Legacy unauthenticated /internal/job/... result callbacks. Set to false once
    # Lambda results arrive through SQS; the worker then relies on SQS only.
    enable_http_result_callbacks: bool = Field(
        default_factory=lambda: os.getenv(
            "ENABLE_HTTP_RESULT_CALLBACKS", "true"
        ).strip().lower()
        not in {"false", "0", "no"}
    )

    # SQS consumer settings
    sqs_consumer_wait_time_seconds: int = Field(
        default_factory=lambda: int(os.getenv("SQS_CONSUMER_WAIT_TIME_SECONDS", "20"))
    )
    sqs_consumer_max_messages: int = Field(
        default_factory=lambda: int(os.getenv("SQS_CONSUMER_MAX_MESSAGES", "10"))
    )
    sqs_consumer_visibility_timeout: int = Field(
        default_factory=lambda: int(os.getenv("SQS_CONSUMER_VISIBILITY_TIMEOUT", "120"))
    )
    sqs_consumer_concurrency: int = Field(
        default_factory=lambda: int(os.getenv("SQS_CONSUMER_CONCURRENCY", "10"))
    )

    # Shared with the Next.js proxy. When it matches X-Proxy-Secret, the
    # X-Client-IP header is trusted for anonymous rate limiting.
    proxy_shared_secret: str = Field(
        default_factory=lambda: os.getenv("PROXY_SHARED_SECRET", "")
    )

    # Single-video AI summaries (OpenAI). Disabled while the key is empty.
    summary_openai_api_key: str = Field(
        default_factory=lambda: os.getenv("SUMMARY_OPENAI_API_KEY", "")
    )
    summary_model: str = Field(
        default_factory=lambda: os.getenv("SUMMARY_MODEL", "gpt-4.1-mini")
    )
    summary_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("SUMMARY_TIMEOUT_SECONDS", "25"))
    )
    summary_max_input_chars: int = Field(
        default_factory=lambda: int(os.getenv("SUMMARY_MAX_INPUT_CHARS", "240000"))
    )
    summary_max_output_tokens: int = Field(
        default_factory=lambda: int(os.getenv("SUMMARY_MAX_OUTPUT_TOKENS", "3000"))
    )

    # Job timeout settings
    job_timeout_minutes: int = Field(
        default_factory=lambda: int(os.getenv("JOB_TIMEOUT_MINUTES", "15"))
    )

    # Server settings
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    # CORS settings - comma-separated list of allowed origins
    cors_origins: str = Field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        )
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated origins string to a list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    # Load environment variables from .env file
    # Note: This requires python-dotenv to be installed
    def __init__(self, **data):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        super().__init__(**data)


# Create a global settings instance
settings = Settings()
