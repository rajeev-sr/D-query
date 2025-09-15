from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(emails.router, prefix="/api/v1/emails", tags=["emails"])
app.include_router(config.router, prefix="/api/v1", tags=["config"])
app.include_router(processing.router, prefix="/api/v1/processing", tags=["processing"])import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
import sys
import os

# Add the project root to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routers import dashboard, emails, health, config, processing
from api.dependencies import get_email_processor, get_gmail_client, get_decision_engine
import uvicorn

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'api.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize components
    logger.info("🚀 Starting D-Query Email Automation API")
    
    # Warm up the models and connections
    try:
        # This will initialize the Gmail client
        await get_gmail_client()
        logger.info("✅ Gmail client initialized")
        
        # This will initialize the decision engine
        await get_decision_engine()
        logger.info("✅ Decision engine initialized")
        
        # This will initialize the email processor
        await get_email_processor()
        logger.info("✅ Email processor initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down D-Query Email Automation API")

# Create FastAPI app with lifespan
app = FastAPI(
    title="D-Query Email Automation API",
    description="AI-powered email automation system with RAG, fine-tuned models, and intelligent routing",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(emails.router, prefix="/api/v1", tags=["emails"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["dashboard"])
app.include_router(config.router, prefix="/api/v1", tags=["config"])

# Serve React build files in production
@app.get("/")
async def serve_frontend():
    """Serve the React frontend"""
    try:
        return FileResponse("frontend/build/index.html")
    except FileNotFoundError:
        return {"message": "D-Query Email Automation API", "docs": "/api/docs"}

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "d-query-email-automation",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )