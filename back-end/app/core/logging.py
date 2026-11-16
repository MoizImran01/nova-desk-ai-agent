from contextvars import ContextVar
import logging
import sys
import uuid
from app.core.config import settings

# Context variable to store a unique correlation ID for every incoming HTTP request
request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Return the current request's correlation ID, or '-' if outside a request."""
    return request_id_ctx_var.get()


def set_request_id(request_id: str | None = None) -> str:
    """Set and return a correlation ID for the current execution context."""
    rid = request_id or str(uuid.uuid4())
    request_id_ctx_var.set(rid)
    return rid


class RequestIdFilter(logging.Filter):
    """Injects `request_id` into every LogRecord so formatters can include it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def setup_logging() -> None:
    """Configure application-wide structured console logging with correlation IDs."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [req:%(request_id)s] | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers on hot-reload
    if not any(isinstance(h, logging.StreamHandler) and any(isinstance(f, RequestIdFilter) for f in h.filters) for h in root_logger.handlers):
        root_logger.handlers = [handler]

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
