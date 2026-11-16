from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import close_workflow, init_workflow
from app.api.exception_handlers import register_exception_handlers
from app.api.routes import chat, knowledge_base
from app.core.config import settings
from app.core.database import create_db_and_tables, engine
from app.core.logging import set_request_id, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 1. Lifespan Context Manager (Modern FastAPI Startup/Shutdown)
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up %s (v%s) in '%s' mode...", settings.PROJECT_NAME, settings.VERSION, settings.ENVIRONMENT)
    logger.info("Ensuring database tables and LangGraph checkpointer exist...")
    create_db_and_tables()
    init_workflow()
    yield
    logger.info("Shutting down %s and closing database pools...", settings.PROJECT_NAME)
    close_workflow()
    engine.dispose()


# ---------------------------------------------------------
# 2. Application Instance
# ---------------------------------------------------------
app = FastAPI(
    title="Nova Desk AI Receptionist API",
    description="Backend for the Med Spa AI scheduling and RAG system.",
    version=settings.VERSION,
    lifespan=lifespan,
)

# Register centralized exception handlers
register_exception_handlers(app)


# ---------------------------------------------------------
# 3. Correlation ID & Request Logging Middleware
# ---------------------------------------------------------
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    incoming_req_id = request.headers.get("X-Request-ID")
    req_id = set_request_id(incoming_req_id)
    start_time = time.perf_counter()

    response = await call_next(request)

    duration_ms = int((time.perf_counter() - start_time) * 1000)
    response.headers["X-Request-ID"] = req_id
    logger.info(
        "%s %s -> %d (%dms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ---------------------------------------------------------
# 4. CORS Middleware Configuration
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# ---------------------------------------------------------
# 5. Register Routers
# ---------------------------------------------------------
app.include_router(chat.router)
app.include_router(knowledge_base.router)


# ---------------------------------------------------------
# 6. Health Check Endpoint
# ---------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "message": "Nova Desk API is running.",
    }
