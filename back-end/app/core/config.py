from datetime import time
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = str(Path(__file__).resolve().parents[2] / ".env")


class Settings(BaseSettings):
    # --- Application Metadata ---
    PROJECT_NAME: str = "Nova Desk Backend"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="development | staging | production")
    LOG_LEVEL: str = Field(default="INFO")

    # --- Required Secrets (loaded from .env) ---
    DATABASE_URL: str
    GROQ_API_KEY: str
    GOOGLE_GEMINI_API_KEY: str
    PINECONE_API_KEY: str

    # --- Optional Zero-Cost Fallback Provider Keys ---
    OPENROUTER_API_KEY: Optional[str] = Field(
        default=None,
        description="Optional OpenRouter API key for Tier-3 free model fallbacks",
    )
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    CEREBRAS_API_KEY: Optional[str] = Field(
        default=None,
        description="Optional Cerebras API key for high-daily-quota 120B tool-calling fallbacks",
    )
    CEREBRAS_BASE_URL: str = Field(default="https://api.cerebras.ai/v1")

    # --- Admin Security ---
    ADMIN_API_KEY: Optional[str] = Field(
        default=None,
        description="API key required to enable admin endpoints like /upload-faqs (disabled with 503 when unset)",
    )

    # --- Database Connection Pool Settings ---
    DB_POOL_SIZE: int = Field(default=5, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0)
    DB_POOL_RECYCLE_SECONDS: int = Field(default=300, ge=60)

    # --- Business & Scheduling Configuration ---
    SPA_TIMEZONE: str = Field(default="Asia/Karachi")
    BUSINESS_OPEN_HOUR: int = Field(default=9, ge=0, le=23)
    BUSINESS_CLOSE_HOUR: int = Field(default=17, ge=1, le=23)
    DEFAULT_SLOT_DURATION_MINUTES: int = Field(default=30, ge=5)
    BOOKING_HORIZON_DAYS: int = Field(
        default=14,
        ge=1,
        description="Guests can book from today up to this many days ahead (inclusive)",
    )
    LOOKAHEAD_DAYS_FOR_ALTERNATIVES: int = Field(default=7, ge=1)
    MAX_ALTERNATIVE_DAYS_RETURNED: int = Field(default=3, ge=1)
    FUZZY_MATCH_CUTOFF: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- Vector DB & RAG Settings ---
    PINECONE_INDEX_NAME: str = Field(default="nova-spa-faqs")
    EMBEDDING_MODEL_NAME: str = Field(default="models/gemini-embedding-001")
    EMBEDDING_DIMENSIONS: int = Field(default=768)
    RAG_CHUNK_SIZE: int = Field(default=1000)
    RAG_CHUNK_OVERLAP: int = Field(default=200)
    RAG_TOP_K: int = Field(default=3)
    SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.92, ge=0.0, le=1.0)

    # --- Multi-Bucket LLM Model Configuration ---
    # Groq Quota Buckets
    FAST_ROUTER_GROQ_MODEL: str = Field(
        default="openai/gpt-oss-20b",
        description="Lightweight fast Groq bucket (1,000 RPD, 30 RPM) for intent routing & FAQ synthesis",
    )
    TOOL_REASONING_GROQ_MODEL: str = Field(
        default="openai/gpt-oss-120b",
        description="Heavy Groq bucket (1,000 RPD, 30 RPM) reserved strictly for tool-calling nodes",
    )
    ALT_GROQ_MODEL: str = Field(
        default="qwen/qwen3.8-27b",
        description="Separate Groq model quota bucket for structured & tool fallbacks",
    )

    # High-Capacity Gemini Quota Buckets (15 RPM | 250K TPM | 500 RPD each)
    GEMINI_FLASH_LITE_3_5_MODEL: str = Field(
        default="gemini-3.5-flash-lite",
        description="High-RPD Gemini bucket #1 (500 RPD, 15 RPM, 250K TPM)",
    )
    GEMINI_FLASH_LITE_3_1_MODEL: str = Field(
        default="gemini-3.1-flash-lite",
        description="High-RPD Gemini bucket #2 (500 RPD, 15 RPM, 250K TPM)",
    )

    # Secondary Gemini Flash Quota Buckets (5 RPM | 250K TPM | 20 RPD each — reserved for Tier-2 tool fallbacks)
    GEMINI_FLASH_3_8_MODEL: str = Field(
        default="gemini-3.8-flash",
        description="Emergency tool-calling fallback bucket (20 RPD, 250K TPM)",
    )
    GEMINI_FLASH_3_6_MODEL: str = Field(
        default="gemini-3.6-flash",
        description="Emergency modification tool-calling fallback bucket (20 RPD, 250K TPM)",
    )

    # Optional OpenRouter & Cerebras Models
    OPENROUTER_FAST_MODEL: str = Field(default="openrouter/free")
    OPENROUTER_LIGHTNING_MODEL: str = Field(default="nvidia/nemotron-3.5-lightning:free")
    OPENROUTER_TOOL_MODEL: str = Field(default="nvidia/nemotron-3.5-super-vl:free")
    CEREBRAS_TOOL_MODEL: str = Field(default="gpt-oss-120b")

    # Fail-fast resiliency settings (max_retries=0 ensures immediate failover on 429)
    LLM_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0.0)
    LLM_MAX_RETRIES: int = Field(default=0, ge=0)

    # --- CORS Settings ---
    ALLOWED_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://nova-desk-ai-agent.vercel.app",
            "https://nova-desk-ai-agent-git-main-moiz-imrans-projects-80fe3802.vercel.app",
        ]
    )

    @property
    def business_open_time(self) -> time:
        return time(self.BUSINESS_OPEN_HOUR, 0)

    @property
    def business_close_time(self) -> time:
        return time(self.BUSINESS_CLOSE_HOUR, 0)

    model_config = SettingsConfigDict(
        env_file=(_BACKEND_ENV_FILE, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
