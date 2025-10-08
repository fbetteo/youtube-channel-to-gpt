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
