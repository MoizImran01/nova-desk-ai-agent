import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.core.exceptions import NovaDeskError
from app.core.logging import get_request_id

logger = logging.getLogger(__name__)


def _build_error_payload(
    *,
    code: str,
    message: str,
    details: dict | list | None = None,
) -> dict:
    """
    Construct a standardized error response body.
    Includes `detail` at the top level for backwards compatibility with clients
    expecting FastAPI's default `{"detail": "..."}` format, alongside a rich
    `error` object containing `code`, `message`, `details`, and `request_id`.
    """
    request_id = get_request_id()
    return {
        "detail": message,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        },
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI application instance."""

    @app.exception_handler(NovaDeskError)
    async def handle_nova_desk_error(request: Request, exc: NovaDeskError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error(
                "NovaDeskError [%s] on %s %s: %s",
                exc.error_code,
                request.method,
                request.url.path,
                exc.message,
                exc_info=True,
            )
        else:
            logger.warning(
                "Domain Error [%s] on %s %s: %s (details=%s)",
                exc.error_code,
                request.method,
                request.url.path,
                exc.message,
                exc.details,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error_payload(
                code=exc.error_code,
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        sanitized_errors = [
            {
                "loc": list(err.get("loc", [])),
                "msg": err.get("msg", "Invalid value"),
                "type": err.get("type", "value_error"),
            }
            for err in exc.errors()
        ]
        logger.warning(
            "Request validation failed on %s %s: %s",
            request.method,
            request.url.path,
            sanitized_errors,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_build_error_payload(
                code="REQUEST_VALIDATION_ERROR",
                message="Invalid request payload.",
                details={"validation_errors": sanitized_errors},
            ),
        )

    @app.exception_handler(SQLAlchemyError)
    async def handle_sqlalchemy_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.error(
            "Unhandled SQLAlchemyError on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_build_error_payload(
                code="DATABASE_ERROR",
                message="An unexpected database error occurred. Please try again shortly.",
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.critical(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_build_error_payload(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal server error occurred.",
            ),
        )
