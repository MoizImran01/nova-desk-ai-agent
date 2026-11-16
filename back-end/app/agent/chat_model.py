import logging
from typing import Any, Literal, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

AgentModelRole = Literal[
    "intent_classification",
    "collect_appointment_details",
    "collect_reschedule_details",
    "handle_retreive_faqs",
]


class LLMRegistry:
    """
    Centralized Multi-Provider, Multi-Bucket LLM Registry.

    Solves the two critical LangGraph/LangChain fallback pitfalls:
    1. Fail-Fast on Rate Limits (`max_retries=0`, `timeout=15.0`):
       Disables slow internal SDK backoff loops so a 429/503 immediately triggers
       the next model bucket in milliseconds.
    2. Pre-Binding Schemas Before `.with_fallbacks()`:
       Always attaches `.with_structured_output(Schema)` to EACH candidate model
       individually (`ChatGroq`, `ChatGoogleGenerativeAI`, `ChatOpenAI`) before
       linking them with `.with_fallbacks(...)`.
    """

    @staticmethod
    def _groq(model_name: str, temperature: float = 0.0) -> BaseChatModel:
        return ChatGroq(
            model=model_name,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
            max_retries=settings.LLM_MAX_RETRIES,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _gemini(model_name: str, temperature: float = 0.0) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.GOOGLE_GEMINI_API_KEY,
            temperature=temperature,
            max_retries=settings.LLM_MAX_RETRIES,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _openrouter(model_name: str, temperature: float = 0.0) -> BaseChatModel:
        return ChatOpenAI(
            model=model_name,
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            temperature=temperature,
            max_retries=settings.LLM_MAX_RETRIES,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

    @classmethod
    def get_candidates(cls, role: AgentModelRole) -> list[BaseChatModel]:
        """
        Return the ordered list of `[primary, fallback_1, fallback_2, ...]` chat models
        for the given agent role across independent quota buckets.
        """
        has_openrouter = bool(settings.OPENROUTER_API_KEY)

        if role == "intent_classification":
            # Primary: Groq 20B (1,000 RPD) -> FB1: Gemini 3.1 Flash Lite (500 RPD) -> FB2: Gemini 3.5 Flash Lite (500 RPD) / OpenRouter
            candidates: list[BaseChatModel] = [
                cls._groq(settings.FAST_ROUTER_GROQ_MODEL, temperature=0.0),
                cls._gemini(settings.GEMINI_FLASH_LITE_3_1_MODEL, temperature=0.0),
                cls._gemini(settings.GEMINI_FLASH_LITE_3_5_MODEL, temperature=0.0),
            ]
            if has_openrouter:
                candidates.append(cls._openrouter(settings.OPENROUTER_LIGHTNING_MODEL, temperature=0.0))
            return candidates

        if role == "collect_appointment_details":
            # Primary: Gemini 3.5 Flash Lite (500 RPD) -> FB1: Gemini 3.1 Flash Lite (500 RPD) -> FB2: Groq Qwen 27B
            return [
                cls._gemini(settings.GEMINI_FLASH_LITE_3_5_MODEL, temperature=0.0),
                cls._gemini(settings.GEMINI_FLASH_LITE_3_1_MODEL, temperature=0.0),
                cls._groq(settings.ALT_GROQ_MODEL, temperature=0.0),
            ]

        if role == "collect_reschedule_details":
            # Primary: Gemini 3.1 Flash Lite (500 RPD) -> FB1: Gemini 3.5 Flash Lite (500 RPD) -> FB2: Groq Qwen 27B
            return [
                cls._gemini(settings.GEMINI_FLASH_LITE_3_1_MODEL, temperature=0.0),
                cls._gemini(settings.GEMINI_FLASH_LITE_3_5_MODEL, temperature=0.0),
                cls._groq(settings.ALT_GROQ_MODEL, temperature=0.0),
            ]

        if role == "handle_retreive_faqs":
            # Primary: Groq 20B (1,000 RPD) -> FB1: Gemini 3.1 Flash Lite (500 RPD, 250K TPM) -> FB2: Gemini 3.5 Flash Lite (500 RPD) / OpenRouter Free
            candidates = [
                cls._groq(settings.FAST_ROUTER_GROQ_MODEL, temperature=0.2),
                cls._gemini(settings.GEMINI_FLASH_LITE_3_1_MODEL, temperature=0.2),
                cls._gemini(settings.GEMINI_FLASH_LITE_3_5_MODEL, temperature=0.2),
            ]
            if has_openrouter:
                candidates.append(cls._openrouter(settings.OPENROUTER_FAST_MODEL, temperature=0.2))
            return candidates

        raise ValueError(f"Unknown agent model role: {role}")

    @classmethod
    def get_text_llm(cls, role: AgentModelRole) -> Runnable:
        """
        Return a plain-text chat model runnable with automatic multi-bucket fallbacks.
        """
        candidates = cls.get_candidates(role)
        primary, fallbacks = candidates[0], candidates[1:]
        return primary.with_fallbacks(fallbacks) if fallbacks else primary

    @classmethod
    def get_structured_llm(cls, role: AgentModelRole, schema: Type[BaseModel]) -> Runnable:
        """
        Bind `.with_structured_output(schema)` to EACH candidate model individually
        BEFORE chaining them with `.with_fallbacks(...)`.
        """
        candidates = cls.get_candidates(role)
        bound_candidates = [model.with_structured_output(schema) for model in candidates]
        primary, fallbacks = bound_candidates[0], bound_candidates[1:]
        return primary.with_fallbacks(fallbacks) if fallbacks else primary


def invoke_structured(
    runnable: Runnable,
    payload: Any,
    schema_cls: Type[BaseModel],
    fallback_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Invoke a structured-output runnable and safely convert its output to a dict.
    Guards against `.with_structured_output()` returning `None` on malformed tool JSON.
    """
    result = runnable.invoke(payload)
    if result is None:
        logger.warning(
            "Structured output returned None for schema %s; using safe fallback.",
            schema_cls.__name__,
        )
        return fallback_dict if fallback_dict is not None else schema_cls().model_dump()
    if isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    return fallback_dict if fallback_dict is not None else schema_cls().model_dump()


def clean_str(value: Any) -> str | None:
    """Strip LLM string output; treat empty / 'null' strings as missing."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned and cleaned.lower() not in ("null", "none") else None
