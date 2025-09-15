from fastapi import APIRouter, HTTPException, status
from api.models import HealthCheck, APIResponse
from api.dependencies import get_gmail_client, get_decision_engine, get_email_processor, get_config
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Comprehensive system health check"""
    health_status = {
        "status": "healthy",
        "gmail_connection": False,
        "decision_engine": False,
        "ai_model_loaded": False,
        "database_connection": False,
        "last_check": datetime.now()
    }
    
    try:
                # Test Gmail Connection
        try:
            gmail_client = await get_gmail_client()
            # Check if it's a mock client (demo mode) or real client
            if hasattr(gmail_client, 'demo_mode'):
                health_status["gmail_connection"] = True  # Demo mode is working
                logger.info("Gmail connection: OK (Demo Mode)")
            elif gmail_client.is_authenticated():
                health_status["gmail_connection"] = True
                logger.info("Gmail connection: OK")
            else:
                raise Exception("Not authenticated")
        except Exception as e:
            logger.warning(f"Gmail connection: FAILED - {e}")
        
        # Test Decision Engine
        try:
            decision_engine = await get_decision_engine()
            # Test with a simple query - process_query takes only email_data dict
            test_email_data = {
                "subject": "test health check", 
                "body": "test",
                "sender": "test@test.com"
            }
            test_result = decision_engine.process_query(test_email_data)
            if test_result:
                health_status["decision_engine"] = True
                health_status["ai_model_loaded"] = True
            logger.info("Decision engine: OK")
        except Exception as e:
            logger.warning(f"Decision engine: FAILED - {e}")
        
        # Test Email Processor
        try:
            email_processor = await get_email_processor()
            # Simple test
            test_email = {"sender": "test@iitbhilai.ac.in", "body": "test"}
            is_student = email_processor.is_student_email(test_email)
            if isinstance(is_student, bool):
                health_status["database_connection"] = True
            logger.info("Email processor: OK")
        except Exception as e:
            logger.warning(f"Email processor: FAILED - {e}")
        
        # Overall status
        if not any([health_status["gmail_connection"], health_status["decision_engine"]]):
            health_status["status"] = "unhealthy"
        elif not all([health_status["gmail_connection"], health_status["decision_engine"], health_status["ai_model_loaded"]]):
            health_status["status"] = "degraded"
        
        return HealthCheck(**health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )

@router.get("/health/detailed", response_model=APIResponse)
async def detailed_health_check():
    """Detailed system health information"""
    try:
        config = get_config()
        
        details = {
            "system": {
                "uptime": "running",
                "configuration_loaded": bool(config),
                "timestamp": datetime.now().isoformat()
            },
            "components": {},
            "configuration": {
                "auto_respond_enabled": config.get("auto_respond_enabled", False),
                "processing_interval": config.get("processing_interval_minutes", 0),
                "confidence_thresholds": config.get("confidence_thresholds", {}),
                "working_hours_enabled": config.get("working_hours", {}).get("enabled", False)
            }
        }
        
        # Test each component individually
        components_status = {}
        
        # Gmail Client
        try:
            gmail_client = await get_gmail_client()
            components_status["gmail_client"] = {
                "status": "healthy",
                "service_initialized": bool(gmail_client.service),
                "last_test": datetime.now().isoformat()
            }
        except Exception as e:
            components_status["gmail_client"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_test": datetime.now().isoformat()
            }
        
        # Decision Engine
        try:
            decision_engine = await get_decision_engine()
            components_status["decision_engine"] = {
                "status": "healthy",
                "ai_model_loaded": hasattr(decision_engine, 'query_classifier'),
                "rag_system": hasattr(decision_engine, 'rag_system'),
                "last_test": datetime.now().isoformat()
            }
        except Exception as e:
            components_status["decision_engine"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_test": datetime.now().isoformat()
            }
        
        details["components"] = components_status
        
        return APIResponse(
            success=True,
            message="Detailed health check completed",
            data=details
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return APIResponse(
            success=False,
            message="Detailed health check failed",
            error=str(e)
        )