from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class EmailStatus(str, Enum):
    UNREAD = "unread"
    READ = "read"
    PROCESSED = "processed"
    AUTO_RESPONDED = "auto_responded"
    NEEDS_REVIEW = "needs_review"
    ESCALATED = "escalated"

class EmailAction(str, Enum):
    AUTO_RESPOND = "auto_respond"
    NEEDS_REVIEW = "needs_review"
    ESCALATE = "escalate"
    IGNORE = "ignore"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Request Models
class EmailProcessRequest(BaseModel):
    email_ids: Optional[List[str]] = None
    force_reprocess: bool = False
    max_emails: Optional[int] = 10

class EmailResponseRequest(BaseModel):
    email_id: str
    response_text: str
    subject: Optional[str] = None

class EmailReviewRequest(BaseModel):
    email_id: str
    action: EmailAction
    review_notes: Optional[str] = None
    custom_response: Optional[str] = None

class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any]

# Response Models
class EmailSender(BaseModel):
    email: str
    name: Optional[str] = None

class EmailData(BaseModel):
    id: str
    sender: EmailSender
    subject: str
    body: str
    date: datetime
    status: EmailStatus
    thread_id: Optional[str] = None
    labels: List[str] = []
    attachments: List[str] = []
    confidence_score: Optional[float] = None
    suggested_action: Optional[EmailAction] = None
    ai_response: Optional[str] = None
    is_student_email: bool = False

class ProcessingResult(BaseModel):
    email_id: str
    action_taken: EmailAction
    confidence_score: float
    processing_time: float
    ai_response: Optional[str] = None
    error: Optional[str] = None

class BatchProcessingResult(BaseModel):
    total_processed: int
    successful: int
    failed: int
    auto_responded: int
    needs_review: int
    escalated: int
    results: List[ProcessingResult]
    processing_time: float

class SystemStats(BaseModel):
    total_emails_processed: int
    auto_response_rate: float
    average_confidence: float
    emails_pending_review: int
    system_uptime: float
    last_processing_run: Optional[datetime] = None
    processing_status: ProcessingStatus

class ProcessingMetrics(BaseModel):
    daily_processed: Dict[str, int]
    weekly_processed: Dict[str, int]
    confidence_distribution: Dict[str, int]
    action_distribution: Dict[str, int]
    response_times: List[float]

class ConfigResponse(BaseModel):
    processing_interval_minutes: int
    max_emails_per_batch: int
    auto_respond_enabled: bool
    confidence_thresholds: Dict[str, float]
    human_review_required_categories: List[str]
    escalation_keywords: List[str]
    working_hours: Dict[str, Any]
    email_settings: Dict[str, Any]
    safety_settings: Dict[str, Any]

class EmailListResponse(BaseModel):
    emails: List[EmailData]
    total_count: int
    page: int
    page_size: int
    has_next: bool

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    gmail_connection: bool
    decision_engine: bool
    ai_model_loaded: bool
    database_connection: bool
    last_check: datetime

# Gmail-specific models
class GmailLabel(BaseModel):
    id: str
    name: str
    type: str

class EmailThread(BaseModel):
    id: str
    emails: List[EmailData]
    subject: str
    participants: List[EmailSender]
    last_activity: datetime