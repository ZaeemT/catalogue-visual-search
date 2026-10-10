from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """

    APP_NAME: str = "Catalogue Visual Search"
    APP_VERSION: str = "0.1.0"

    # MongoDB configuration
    MONGO_URI: str = Field(...)  # MongoDB URI
    DB_NAME: str = Field(...)  # Database name
    
    # JWT / Auth configuration
    JWT_SECRET: str = Field(...)
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)  # Token expiration time (minutes)

    # Visual search configuration
    # Switching the embedding model is this one setting: the index directory,
    # the loaded weights and the compatibility check all follow from it.
    EMBEDDING_MODEL: str = Field(default="marqo-fashionsiglip")
    INDEX_ROOT: Path = Field(default=Path("data/index"))
    CATALOGUE_DIR: Path = Field(default=Path("data/catalogue"))
    EMBEDDING_DEVICE: Optional[str] = Field(default=None)  # None = auto (mps/cuda/cpu)
    SEARCH_TOP_K_DEFAULT: int = Field(default=10)
    SEARCH_TOP_K_MAX: int = Field(default=50)
    SEARCH_MAX_UPLOAD_BYTES: int = Field(default=15 * 1024 * 1024)
    SEARCH_WARMUP_QUERIES: int = Field(default=2)

    # Explanations. The sentence is assembled in code from measured signals;
    # the LLM only translates the category name into English. Turning it off
    # keeps the explanation, with the category in its original wording.
    EXPLAIN_WITH_LLM: bool = Field(default=True)
    OLLAMA_MODEL: str = Field(default="llama3.2:1b")
    OLLAMA_HOST: str = Field(default="http://127.0.0.1:11434")
    OLLAMA_TIMEOUT: float = Field(default=10.0)

    # Email server
    MAIL_SERVER: Optional[str] = Field(default=None)
    MAIL_PORT: Optional[int] = Field(default=None)
    MAIL_USE_TLS: Optional[bool] = Field(default=None)
    MAIL_USERNAME: Optional[str] = Field(default=None)
    MAIL_PASSWORD: Optional[str] = Field(default=None)
    MAIL_DEFAULT_SENDER: Optional[str] = Field(default=None)


    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings, loading from environment variables if not already cached
    """
    return Settings()

    