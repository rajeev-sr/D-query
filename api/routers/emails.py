from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional
from datetime import datetime
import logging
import asyncio

from api.models import (
    EmailData, EmailListResponse, EmailProcessRequest, EmailResponseRequest,
    EmailReviewRequest, BatchProcessingResult, ProcessingResult, APIResponse,
    EmailStatus, EmailAction, EmailSender
)
from api.dependencies import (
    get_gmail_client, get_decision_engine, get_email_processor, 
    get_automated_processor, get_email_sender, get_config
)

router = APIRouter()
logger = logging.getLogger(__name__)

def convert_gmail_email_to_model(email_data: dict) -> EmailData:
    """Convert Gmail email data to our EmailData model"""
    try:
        sender_info = EmailSender(
            email=email_data.get('sender', ''),
            name=email_data.get('sender_name')
        )
        
        return EmailData(
            id=email_data['id'],
            sender=sender_info,
            subject=email_data.get('subject', ''),
            body=email_data.get('body', ''),
            date=datetime.fromisoformat(email_data['date'].replace('Z', '+00:00')),
            status=EmailStatus.UNREAD if email_data.get('unread', True) else EmailStatus.READ,
            thread_id=email_data.get('thread_id'),
            labels=email_data.get('labels', []),
            attachments=email_data.get('attachments', []),
            is_student_email=email_data.get('is_student_email', False)
        )
    except Exception as e:
        logger.error(f"Failed to convert email data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process email data: {str(e)}"
        )

@router.get("/emails", response_model=EmailListResponse)
async def get_emails(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of emails per page"),
    status_filter: Optional[EmailStatus] = Query(None, description="Filter by email status"),
    unread_only: bool = Query(False, description="Show only unread emails"),
    student_only: bool = Query(False, description="Show only student emails"),
    gmail_client = Depends(get_gmail_client),
    email_processor = Depends(get_email_processor)
):
    """Get list of emails with pagination and filtering"""
    try:
        # Build Gmail query
        query_parts = []
        if unread_only or status_filter == EmailStatus.UNREAD:
            query_parts.append("is:unread")
        
        gmail_query = " ".join(query_parts) if query_parts else None
        
        # Fetch emails from Gmail
        max_results = page * page_size  # Get enough emails for pagination
        emails_data = gmail_client.fetch_emails(
            max_results=max_results,
            query=gmail_query
        )
        
        # Process emails
        processed_emails = []
        for email_data in emails_data:
            # Check if it's a student email
            is_student = email_processor.is_student_email(email_data)
            email_data['is_student_email'] = is_student
            
            # Apply student filter
            if student_only and not is_student:
                continue
            
            email_model = convert_gmail_email_to_model(email_data)
            processed_emails.append(email_model)
        
        # Apply pagination
        total_count = len(processed_emails)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_emails = processed_emails[start_idx:end_idx]
        
        return EmailListResponse(
            emails=paginated_emails,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=end_idx < total_count
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch emails: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch emails: {str(e)}"
        )

@router.get("/emails/{email_id}", response_model=EmailData)
async def get_email_detail(
    email_id: str,
    gmail_client = Depends(get_gmail_client),
    email_processor = Depends(get_email_processor)
):
    """Get detailed information about a specific email"""
    try:
        # Fetch single email
        email_data = gmail_client.get_email_by_id(email_id)
        if not email_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Email not found"
            )
        
        # Process email
        is_student = email_processor.is_student_email(email_data)
        email_data['is_student_email'] = is_student
        
        return convert_gmail_email_to_model(email_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch email {email_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch email: {str(e)}"
        )

@router.post("/emails/process", response_model=BatchProcessingResult)
async def process_emails(
    request: EmailProcessRequest,
    automated_processor = Depends(get_automated_processor)
):
    """Process a batch of emails with AI decision engine"""
    try:
        start_time = datetime.now()
        
        # Use automated processor for batch processing
        if request.email_ids:
            # Process specific emails
            results = []
            for email_id in request.email_ids:
                try:
                    # This would need to be implemented in automated_processor
                    # For now, we'll simulate the processing
                    result = ProcessingResult(
                        email_id=email_id,
                        action_taken=EmailAction.NEEDS_REVIEW,
                        confidence_score=0.5,
                        processing_time=1.0,
                        ai_response=None
                    )
                    results.append(result)
                except Exception as e:
                    result = ProcessingResult(
                        email_id=email_id,
                        action_taken=EmailAction.ESCALATE,
                        confidence_score=0.0,
                        processing_time=0.0,
                        error=str(e)
                    )
                    results.append(result)
        else:
            # Process all unread emails
            processed_results = automated_processor.process_emails_batch(
                max_emails=request.max_emails or 10
            )
            
            results = []
            for result in processed_results:
                processing_result = ProcessingResult(
                    email_id=result.get('email_id', ''),
                    action_taken=EmailAction(result.get('action', 'needs_review')),
                    confidence_score=result.get('confidence_score', 0.0),
                    processing_time=result.get('processing_time', 0.0),
                    ai_response=result.get('ai_response'),
                    error=result.get('error')
                )
                results.append(processing_result)
        
        # Calculate summary statistics
        total_processed = len(results)
        successful = len([r for r in results if not r.error])
        failed = len([r for r in results if r.error])
        auto_responded = len([r for r in results if r.action_taken == EmailAction.AUTO_RESPOND])
        needs_review = len([r for r in results if r.action_taken == EmailAction.NEEDS_REVIEW])
        escalated = len([r for r in results if r.action_taken == EmailAction.ESCALATE])
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchProcessingResult(
            total_processed=total_processed,
            successful=successful,
            failed=failed,
            auto_responded=auto_responded,
            needs_review=needs_review,
            escalated=escalated,
            results=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Failed to process emails: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process emails: {str(e)}"
        )

@router.post("/emails/{email_id}/respond", response_model=APIResponse)
async def send_email_response(
    email_id: str,
    request: EmailResponseRequest,
    gmail_client = Depends(get_gmail_client),
    email_sender = Depends(get_email_sender)
):
    """Send a response to a specific email"""
    try:
        # Get original email details
        original_email = gmail_client.get_email_by_id(email_id)
        if not original_email:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Original email not found"
            )
        
        # Extract sender information
        original_sender = original_email.get('sender', '')
        original_subject = original_email.get('subject', '')
        
        # Prepare response
        response_subject = request.subject or f"Re: {original_subject}"
        
        # Send the response
        success = email_sender.send_email(
            to_email=original_sender,
            subject=response_subject,
            body=request.response_text,
            reply_to_id=email_id
        )
        
        if success:
            # Mark original email as processed
            gmail_client.mark_as_read(email_id)
            
            return APIResponse(
                success=True,
                message="Email response sent successfully",
                data={"email_id": email_id, "response_sent": True}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send email response"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send response for email {email_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send email response: {str(e)}"
        )

@router.post("/emails/{email_id}/review", response_model=APIResponse)
async def review_email(
    email_id: str,
    request: EmailReviewRequest,
    gmail_client = Depends(get_gmail_client),
    email_sender = Depends(get_email_sender)
):
    """Review and take action on an email"""
    try:
        # Get email details
        email_data = gmail_client.get_email_by_id(email_id)
        if not email_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Email not found"
            )
        
        result_data = {"email_id": email_id, "action": request.action.value}
        
        # Take action based on review decision
        if request.action == EmailAction.AUTO_RESPOND and request.custom_response:
            # Send custom response
            original_sender = email_data.get('sender', '')
            original_subject = email_data.get('subject', '')
            response_subject = f"Re: {original_subject}"
            
            success = email_sender.send_email(
                to_email=original_sender,
                subject=response_subject,
                body=request.custom_response,
                reply_to_id=email_id
            )
            
            if success:
                gmail_client.mark_as_read(email_id)
                result_data["response_sent"] = True
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to send custom response"
                )
                
        elif request.action == EmailAction.ESCALATE:
            # Add escalation label and keep unread
            gmail_client.add_label(email_id, "ESCALATED")
            result_data["escalated"] = True
            
        elif request.action == EmailAction.IGNORE:
            # Mark as read and archive
            gmail_client.mark_as_read(email_id)
            gmail_client.archive_email(email_id)
            result_data["archived"] = True
        
        # Log the review action
        logger.info(f"Email {email_id} reviewed: {request.action.value} - {request.review_notes}")
        
        return APIResponse(
            success=True,
            message=f"Email review completed: {request.action.value}",
            data=result_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to review email {email_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to review email: {str(e)}"
        )

@router.get("/emails/pending-review", response_model=EmailListResponse)
async def get_pending_review_emails(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    gmail_client = Depends(get_gmail_client),
    email_processor = Depends(get_email_processor)
):
    """Get emails that need human review"""
    try:
        # Fetch emails with NEEDS_REVIEW label or unprocessed unread emails
        emails_data = gmail_client.fetch_emails(
            max_results=page * page_size,
            query="is:unread -label:AI_PROCESSED"
        )
        
        # Process and filter emails
        processed_emails = []
        for email_data in emails_data:
            is_student = email_processor.is_student_email(email_data)
            email_data['is_student_email'] = is_student
            
            email_model = convert_gmail_email_to_model(email_data)
            email_model.status = EmailStatus.NEEDS_REVIEW
            processed_emails.append(email_model)
        
        # Apply pagination
        total_count = len(processed_emails)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_emails = processed_emails[start_idx:end_idx]
        
        return EmailListResponse(
            emails=paginated_emails,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=end_idx < total_count
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch pending review emails: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch pending review emails: {str(e)}"
        )