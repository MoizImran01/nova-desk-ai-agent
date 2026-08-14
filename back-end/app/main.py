from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import your database initializer
from app.core.database import create_db_and_tables

# Import your routers
from app.api.routes import chat
from app.api.routes import knowledge_base
# from app.api.routes import admin  # Uncomment this when you build the admin/upload routes

# ---------------------------------------------------------
# 1. Lifespan Context Manager (Modern FastAPI Startup)
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # What happens when the server starts
    print("🚀 Starting up Nova Desk API...")
    print("📦 Initializing Database Tables...")
    create_db_and_tables()
    yield
    # What happens when the server shuts down
    print("🛑 Shutting down Nova Desk API...")

# ---------------------------------------------------------
# 2. Application Instance
# ---------------------------------------------------------
app = FastAPI(
    title="Nova Desk AI Receptionist API",
    description="Backend for the Med Spa AI scheduling and RAG system.",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------
# 3. CORS Middleware Configuration
# ---------------------------------------------------------
# During development, allowing "*" is fine. 
# In production, change this to the exact domain of the Med Spa website.
origins = [
    "http://localhost:5173",  # React/Vite default port
    "http://127.0.0.1:5173",
    "*"                       # Allow all for widget embedding
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, PUT, etc.
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 4. Register Routers
# ---------------------------------------------------------
app.include_router(chat.router)
# app.include_router(admin.router) 
app.include_router(knowledge_base.router)

# ---------------------------------------------------------
# 5. Health Check Endpoint
# ---------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    return {"status": "online", "message": "Nova Desk API is running."}