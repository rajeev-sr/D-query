# src/approval_system.py
"""
Human-in-the-Loop Approval System
Manages response approval workflow before sending emails
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import os

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    EXPIRED = "expired"

@dataclass
class ApprovalRequest:
    """Structure for approval requests"""
    request_id: str
    query_id: str
    original_email: Dict[str, Any]
    generated_response: str
    rag_sources: List[str]
    ai_confidence: float
    quality_scores: Dict[str, float]
    classification: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    status: ApprovalStatus
    priority: str
    reviewer_notes: str = ""
    approved_by: str = ""
    approved_at: Optional[datetime] = None
    revision_count: int = 0

class ApprovalSystem:
    """Manages human approval workflow for AI-generated responses"""
    
    def __init__(self, approval_timeout_hours: int = 24):
        self.approval_timeout_hours = approval_timeout_hours
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.approval_history: List[ApprovalRequest] = []
        self.approval_criteria = self._load_approval_criteria()
        self.data_file = "data/approval_requests.json"
        self.stats_file = "data/approval_stats.json"
        
        # Load existing requests
        self._load_approval_data()
        
    def _load_approval_criteria(self) -> Dict[str, Any]:
        """Load approval criteria configuration"""
        return {
            "auto_approval_threshold": {
                "ai_confidence": 0.9,
                "quality_score": 0.85,
                "classification_confidence": 0.9
            },
            "requires_approval": {
                "ai_confidence_below": 0.8,
                "quality_score_below": 0.7,
                "sensitive_categories": ["complaint", "academic_appeal", "emergency"],
                "high_priority_senders": ["dean", "principal", "director"]
            },
            "auto_reject": {
                "ai_confidence_below": 0.5,
                "quality_score_below": 0.4,
                "spam_indicators": True
            }
        }
    
    def evaluate_for_approval(self, query_data: Dict, response_data: Dict, 
                            quality_scores: Dict) -> Dict[str, Any]:
        """Evaluate if response needs human approval"""
        
        ai_confidence = response_data.get("confidence", 0.0)
        quality_score = quality_scores.get("overall", 0.0)
        classification = query_data.get("classification", {})
        classification_confidence = classification.get("confidence", 0.0)
        
        # Check auto-approval criteria
        auto_approval = self.approval_criteria["auto_approval_threshold"]
        if (ai_confidence >= auto_approval["ai_confidence"] and 
            quality_score >= auto_approval["quality_score"] and
            classification_confidence >= auto_approval["classification_confidence"]):
            
            return {
                "needs_approval": False,
                "auto_approved": True,
                "reason": "meets_auto_approval_criteria",
                "confidence_scores": {
                    "ai_confidence": ai_confidence,
                    "quality_score": quality_score,
                    "classification_confidence": classification_confidence
                }
            }
        
        # Check auto-reject criteria
        auto_reject = self.approval_criteria["auto_reject"]
        if (ai_confidence < auto_reject["ai_confidence_below"] or
            quality_score < auto_reject["quality_score_below"]):
            
            return {
                "needs_approval": True,
                "auto_rejected": True,
                "reason": "below_quality_threshold",
                "requires_revision": True,
                "confidence_scores": {
                    "ai_confidence": ai_confidence,
                    "quality_score": quality_score,
                    "classification_confidence": classification_confidence
                }
            }
        
        # Check manual approval criteria
        requires_approval = self.approval_criteria["requires_approval"]
        needs_approval_reasons = []
        
        if ai_confidence < requires_approval["ai_confidence_below"]:
            needs_approval_reasons.append("low_ai_confidence")
        
        if quality_score < requires_approval["quality_score_below"]:
            needs_approval_reasons.append("low_quality_score")
        
        category = classification.get("category", "").lower()
        if category in requires_approval["sensitive_categories"]:
            needs_approval_reasons.append("sensitive_category")
        
        sender = query_data.get("email_content", {}).get("sender", "").lower()
        if any(vip in sender for vip in requires_approval["high_priority_senders"]):
            needs_approval_reasons.append("high_priority_sender")
        
        return {
            "needs_approval": len(needs_approval_reasons) > 0,
            "auto_approved": False,
            "reasons": needs_approval_reasons,
            "priority": "high" if "high_priority_sender" in needs_approval_reasons else "normal",
            "confidence_scores": {
                "ai_confidence": ai_confidence,
                "quality_score": quality_score,
                "classification_confidence": classification_confidence
            }
        }
    
    def create_approval_request(self, query_data: Dict, response_data: Dict, 
                              quality_scores: Dict, evaluation_result: Dict) -> ApprovalRequest:
        """Create new approval request"""
        
        request_id = f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        query_id = query_data.get("query_id", "unknown")
        
        approval_request = ApprovalRequest(
            request_id=request_id,
            query_id=query_id,
            original_email=query_data.get("email_content", {}),
            generated_response=response_data.get("generated_response", ""),
            rag_sources=response_data.get("rag_sources", []),
            ai_confidence=response_data.get("confidence", 0.0),
            quality_scores=quality_scores,
            classification=query_data.get("classification", {}),
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=self.approval_timeout_hours),
            status=ApprovalStatus.PENDING,
            priority=evaluation_result.get("priority", "normal")
        )
        
        # Store pending request
        self.pending_requests[request_id] = approval_request
        
        # Save to file
        self._save_approval_data()
        
        logging.info(f"Created approval request {request_id} for query {query_id}")
        
        return approval_request
    
    def get_pending_approvals(self, priority: Optional[str] = None) -> List[ApprovalRequest]:
        """Get pending approval requests"""
        
        # Clean up expired requests first
        self._cleanup_expired_requests()
        
        pending = list(self.pending_requests.values())
        
        if priority:
            pending = [req for req in pending if req.priority == priority]
        
        # Sort by priority and creation time
        priority_order = {"high": 0, "medium": 1, "normal": 2}
        pending.sort(key=lambda x: (priority_order.get(x.priority, 2), x.created_at))
        
        return pending
    
    def approve_request(self, request_id: str, reviewer: str, notes: str = "") -> Dict[str, Any]:
        """Approve an approval request"""
        
        if request_id not in self.pending_requests:
            return {
                "success": False,
                "error": "Request not found or already processed"
            }
        
        request = self.pending_requests[request_id]
        
        # Check if expired
        if datetime.now() > request.expires_at:
            request.status = ApprovalStatus.EXPIRED
            return {
                "success": False,
                "error": "Request has expired"
            }
        
        # Approve request
        request.status = ApprovalStatus.APPROVED
        request.approved_by = reviewer
        request.approved_at = datetime.now()
        request.reviewer_notes = notes
        
        # Move to history
        self.approval_history.append(request)
        del self.pending_requests[request_id]
        
        # Save changes
        self._save_approval_data()
        
        logging.info(f"Request {request_id} approved by {reviewer}")
        
        return {
            "success": True,
            "approved_at": request.approved_at.isoformat(),
            "approved_by": reviewer,
            "ready_to_send": True
        }
    
    def reject_request(self, request_id: str, reviewer: str, reason: str, 
                      suggest_revision: bool = False) -> Dict[str, Any]:
        """Reject an approval request"""
        
        if request_id not in self.pending_requests:
            return {
                "success": False,
                "error": "Request not found or already processed"
            }
        
        request = self.pending_requests[request_id]
        
        if suggest_revision:
            request.status = ApprovalStatus.NEEDS_REVISION
            request.revision_count += 1
        else:
            request.status = ApprovalStatus.REJECTED
        
        request.reviewer_notes = reason
        request.approved_by = reviewer
        request.approved_at = datetime.now()
        
        # Move to history if fully rejected
        if not suggest_revision:
            self.approval_history.append(request)
            del self.pending_requests[request_id]
        
        # Save changes
        self._save_approval_data()
        
        action = "marked for revision" if suggest_revision else "rejected"
        logging.info(f"Request {request_id} {action} by {reviewer}")
        
        return {
            "success": True,
            "action": "revision_required" if suggest_revision else "rejected",
            "needs_revision": suggest_revision,
            "reason": reason
        }
    
    def update_response(self, request_id: str, updated_response: str, 
                       updated_by: str) -> Dict[str, Any]:
        """Update response for requests needing revision"""
        
        if request_id not in self.pending_requests:
            return {
                "success": False,
                "error": "Request not found or not pending"
            }
        
        request = self.pending_requests[request_id]
        
        if request.status != ApprovalStatus.NEEDS_REVISION:
            return {
                "success": False,
                "error": "Request is not marked for revision"
            }
        
        # Update response
        request.generated_response = updated_response
        request.status = ApprovalStatus.PENDING
        request.expires_at = datetime.now() + timedelta(hours=self.approval_timeout_hours)
        
        # Save changes
        self._save_approval_data()
        
        logging.info(f"Response updated for request {request_id} by {updated_by}")
        
        return {
            "success": True,
            "status": "pending_approval",
            "updated_at": datetime.now().isoformat()
        }
    
    def get_approval_stats(self) -> Dict[str, Any]:
        """Get approval system statistics"""
        
        # Calculate stats from history
        total_requests = len(self.approval_history)
        if total_requests == 0:
            return {"message": "No approval history available"}
        
        approved = len([r for r in self.approval_history if r.status == ApprovalStatus.APPROVED])
        rejected = len([r for r in self.approval_history if r.status == ApprovalStatus.REJECTED])
        revised = len([r for r in self.approval_history if r.revision_count > 0])
        
        # Calculate average processing times
        processing_times = []
        for request in self.approval_history:
            if request.approved_at:
                processing_time = (request.approved_at - request.created_at).total_seconds() / 3600
                processing_times.append(processing_time)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Quality score analysis
        quality_scores = [r.quality_scores.get("overall", 0.0) for r in self.approval_history]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        stats = {
            "total_requests": total_requests,
            "approved": approved,
            "rejected": rejected,
            "revision_rate": revised / total_requests * 100,
            "approval_rate": approved / total_requests * 100,
            "average_processing_time_hours": round(avg_processing_time, 2),
            "average_quality_score": round(avg_quality_score, 3),
            "pending_requests": len(self.pending_requests),
            "current_backlog": {
                "high_priority": len([r for r in self.pending_requests.values() if r.priority == "high"]),
                "normal_priority": len([r for r in self.pending_requests.values() if r.priority == "normal"])
            }
        }
        
        return stats
    
    def _cleanup_expired_requests(self):
        """Clean up expired approval requests"""
        now = datetime.now()
        expired_requests = []
        
        for request_id, request in list(self.pending_requests.items()):
            if now > request.expires_at:
                request.status = ApprovalStatus.EXPIRED
                expired_requests.append(request_id)
                
                # Move to history
                self.approval_history.append(request)
                del self.pending_requests[request_id]
        
        if expired_requests:
            self._save_approval_data()
            logging.info(f"Cleaned up {len(expired_requests)} expired approval requests")
    
    def _load_approval_data(self):
        """Load approval data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct pending requests
                for req_data in data.get("pending_requests", []):
                    request = self._dict_to_approval_request(req_data)
                    self.pending_requests[request.request_id] = request
                
                # Reconstruct history
                for req_data in data.get("approval_history", []):
                    request = self._dict_to_approval_request(req_data)
                    self.approval_history.append(request)
                    
                logging.info(f"Loaded {len(self.pending_requests)} pending requests and {len(self.approval_history)} history records")
                
        except Exception as e:
            logging.error(f"Failed to load approval data: {e}")
    
    def _save_approval_data(self):
        """Save approval data to file"""
        try:
            os.makedirs("data", exist_ok=True)
            
            data = {
                "pending_requests": [self._approval_request_to_dict(req) for req in self.pending_requests.values()],
                "approval_history": [self._approval_request_to_dict(req) for req in self.approval_history],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Failed to save approval data: {e}")
    
    def _approval_request_to_dict(self, request: ApprovalRequest) -> Dict:
        """Convert ApprovalRequest to dictionary"""
        data = asdict(request)
        data["status"] = request.status.value
        data["created_at"] = request.created_at.isoformat()
        data["expires_at"] = request.expires_at.isoformat()
        if request.approved_at:
            data["approved_at"] = request.approved_at.isoformat()
        return data
    
    def _dict_to_approval_request(self, data: Dict) -> ApprovalRequest:
        """Convert dictionary to ApprovalRequest"""
        data["status"] = ApprovalStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        if data.get("approved_at"):
            data["approved_at"] = datetime.fromisoformat(data["approved_at"])
        else:
            data["approved_at"] = None
        
        return ApprovalRequest(**data)
    
    def get_request_details(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get detailed information for specific request"""
        
        # Check pending requests
        if request_id in self.pending_requests:
            return self.pending_requests[request_id]
        
        # Check history
        for request in self.approval_history:
            if request.request_id == request_id:
                return request
        
        return None
    
    def bulk_approve(self, request_ids: List[str], reviewer: str, notes: str = "") -> Dict[str, Any]:
        """Bulk approve multiple requests"""
        results = {
            "approved": [],
            "failed": [],
            "total": len(request_ids)
        }
        
        for request_id in request_ids:
            result = self.approve_request(request_id, reviewer, notes)
            if result["success"]:
                results["approved"].append(request_id)
            else:
                results["failed"].append({
                    "request_id": request_id,
                    "error": result["error"]
                })
        
        return results