import hmac
from typing import Optional

from fastapi import Header

from app.core.config import settings
from app.core.exceptions import AuthenticationError, ServiceUnavailableError


def verify_admin_api_key(
    x_admin_api_key: Optional[str] = Header(default=None, alias="X-Admin-API-Key"),
) -> None:
    """
    Admin guard for protected endpoints (e.g. /upload-faqs).
    Fails closed with 503 when ADMIN_API_KEY is unset or blank, and raises 401
    when configured and the X-Admin-API-Key header is missing or does not match.
    """
    if not settings.ADMIN_API_KEY or not settings.ADMIN_API_KEY.strip():
        raise ServiceUnavailableError(
            "Admin endpoints are disabled until ADMIN_API_KEY is configured."
        )
    if not x_admin_api_key or not hmac.compare_digest(x_admin_api_key, settings.ADMIN_API_KEY):
        raise AuthenticationError("Invalid or missing X-Admin-API-Key header.")
