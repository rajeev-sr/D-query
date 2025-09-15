from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
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
    """Application lifespan manager"""
    logger.info("🚀 Starting D-Query Email Automation API")
    
    # Initialize components
    try:
        # Initialize Gmail client
        gmail_client = get_gmail_client()
        logger.info("✅ Gmail client initialized")
        
        # Initialize decision engine
        decision_engine = get_decision_engine()
        logger.info("✅ Decision engine initialized")
        
        # Initialize email processor
        email_processor = get_email_processor()
        logger.info("✅ Email processor initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {str(e)}")
    
    yield
    
    logger.info("🛑 Shutting down D-Query Email Automation API")

# Create FastAPI application
app = FastAPI(
    title="D-Query Email Automation API",
    description="Automated email processing and response system with AI-powered decision making",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(emails.router, prefix="/api/v1/emails", tags=["emails"])
app.include_router(config.router, prefix="/api/v1", tags=["config"])
app.include_router(processing.router, prefix="/api/v1/processing", tags=["processing"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "D-Query Email Automation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon"""
    return FileResponse("static/favicon.ico")

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["api", "src"]
    )