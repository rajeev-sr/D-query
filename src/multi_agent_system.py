# src/multi_agent_system.py
"""
Multi-Agent System for AI Query Handler
Implements specialized agents for different tasks
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from abc import ABC, abstractmethod

@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    message_id: str

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.message_history = []
        self.status = "idle"
        
    @abstractmethod
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and return response if needed"""
        pass
    
    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle specific task type"""
        return task_type in self.capabilities
    
    def log_message(self, message: AgentMessage):
        """Log message for debugging and audit"""
        self.message_history.append(message)
        logging.info(f"Agent {self.agent_id} received: {message.message_type} from {message.sender}")

class PlannerAgent(BaseAgent):
    """Plans and coordinates the query handling process"""
    
    def __init__(self):
        super().__init__(
            agent_id="planner",
            capabilities=["task_planning", "workflow_coordination", "resource_allocation"]
        )
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process planning requests"""
        self.log_message(message)
        
        if message.message_type == "new_query":
            return self._create_processing_plan(message.content)
        elif message.message_type == "task_completed":
            return self._handle_task_completion(message.content)
        elif message.message_type == "escalation_request":
            return self._handle_escalation(message.content)
        
        return None
    
    def _create_processing_plan(self, query_data: Dict) -> AgentMessage:
        """Create processing plan for new query"""
        plan = {
            "query_id": query_data.get("query_id"),
            "steps": [
                {"agent": "classifier", "task": "classify_query", "priority": 1},
                {"agent": "filter", "task": "gemini_filter", "priority": 1},
                {"agent": "responder", "task": "generate_response", "priority": 2},
                {"agent": "validator", "task": "validate_response", "priority": 3}
            ],
            "success_criteria": {
                "classification_confidence": 0.8,
                "response_quality_score": 0.7,
                "rag_relevance": 0.6
            },
            "fallback_plan": "escalate_to_human"
        }
        
        return AgentMessage(
            sender=self.agent_id,
            receiver="coordinator",
            message_type="processing_plan",
            content=plan,
            timestamp=datetime.now(),
            message_id=f"plan_{query_data.get('query_id', 'unknown')}"
        )
    
    def _handle_task_completion(self, completion_data: Dict) -> Optional[AgentMessage]:
        """Handle task completion and determine next steps"""
        task_result = completion_data.get("result", {})
        success = completion_data.get("success", False)
        
        if success and task_result.get("confidence", 0) > 0.8:
            # Continue with next step
            return AgentMessage(
                sender=self.agent_id,
                receiver="coordinator",
                message_type="continue_processing",
                content={"next_step": completion_data.get("next_agent")},
                timestamp=datetime.now(),
                message_id=f"continue_{completion_data.get('task_id')}"
            )
        else:
            # Request human intervention
            return self._handle_escalation(completion_data)
    
    def _handle_escalation(self, escalation_data: Dict) -> AgentMessage:
        """Handle escalation requests"""
        escalation_plan = {
            "escalation_type": escalation_data.get("reason", "low_confidence"),
            "human_review_required": True,
            "priority": "high" if escalation_data.get("urgent", False) else "normal",
            "context": escalation_data,
            "recommended_actions": [
                "manual_review",
                "context_enhancement",
                "response_revision"
            ]
        }
        
        return AgentMessage(
            sender=self.agent_id,
            receiver="escalation_handler",
            message_type="escalation_plan",
            content=escalation_plan,
            timestamp=datetime.now(),
            message_id=f"escalation_{escalation_data.get('query_id')}"
        )

class ClassifierAgent(BaseAgent):
    """Specialized agent for query classification"""
    
    def __init__(self, decision_engine):
        super().__init__(
            agent_id="classifier", 
            capabilities=["query_classification", "intent_detection", "category_analysis"]
        )
        self.decision_engine = decision_engine
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process classification requests"""
        self.log_message(message)
        
        if message.message_type == "classify_query":
            return self._classify_query(message.content)
        
        return None
    
    def _classify_query(self, query_data: Dict) -> AgentMessage:
        """Classify the query using the decision engine"""
        try:
            email_content = query_data.get("email_content", {})
            
            # Use decision engine for classification
            decision = self.decision_engine.make_decision(
                email_content.get("subject", ""),
                email_content.get("body", ""),
                email_content.get("sender", "")
            )
            
            result = {
                "query_id": query_data.get("query_id"),
                "classification": {
                    "category": decision.get("category", "unknown"),
                    "confidence": decision.get("confidence", 0.0),
                    "requires_human": decision.get("requires_human", True),
                    "priority": decision.get("priority", "normal")
                },
                "success": decision.get("confidence", 0.0) > 0.7,
                "next_agent": "filter" if decision.get("confidence", 0.0) > 0.7 else "escalation_handler"
            }
            
            return AgentMessage(
                sender=self.agent_id,
                receiver="planner",
                message_type="classification_result",
                content=result,
                timestamp=datetime.now(),
                message_id=f"classification_{query_data.get('query_id')}"
            )
            
        except Exception as e:
            logging.error(f"Classification error: {e}")
            return AgentMessage(
                sender=self.agent_id,
                receiver="planner",
                message_type="task_failed",
                content={"error": str(e), "query_id": query_data.get("query_id")},
                timestamp=datetime.now(),
                message_id=f"error_{query_data.get('query_id')}"
            )

class FilterAgent(BaseAgent):
    """Specialized agent for Gemini-based query filtering"""
    
    def __init__(self, gemini_filter):
        super().__init__(
            agent_id="filter",
            capabilities=["query_filtering", "relevance_check", "student_verification"]
        )
        self.gemini_filter = gemini_filter
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process filtering requests"""
        self.log_message(message)
        
        if message.message_type == "filter_query":
            return self._filter_query(message.content)
        
        return None
    
    def _filter_query(self, query_data: Dict) -> AgentMessage:
        """Filter query using Gemini"""
        try:
            email_content = query_data.get("email_content", {})
            
            if self.gemini_filter:
                is_query_related = self.gemini_filter.is_query_related(
                    email_content.get("subject", ""),
                    email_content.get("body", ""),
                    email_content.get("sender", "")
                )
                
                result = {
                    "query_id": query_data.get("query_id"),
                    "filter_result": {
                        "is_valid_query": is_query_related["is_query"],
                        "confidence": is_query_related["confidence"],
                        "reasoning": is_query_related["reasoning"],
                        "category": is_query_related.get("category", "unknown")
                    },
                    "success": True,
                    "next_agent": "responder" if is_query_related["is_query"] else "escalation_handler"
                }
            else:
                # Fallback when Gemini is not available
                result = {
                    "query_id": query_data.get("query_id"),
                    "filter_result": {
                        "is_valid_query": True,  # Allow all queries when filter unavailable
                        "confidence": 0.5,
                        "reasoning": "Gemini filter unavailable, allowing query",
                        "category": "unknown"
                    },
                    "success": True,
                    "next_agent": "responder"
                }
            
            return AgentMessage(
                sender=self.agent_id,
                receiver="planner",
                message_type="filter_result",
                content=result,
                timestamp=datetime.now(),
                message_id=f"filter_{query_data.get('query_id')}"
            )
            
        except Exception as e:
            logging.error(f"Filtering error: {e}")
            return AgentMessage(
                sender=self.agent_id,
                receiver="planner",
                message_type="task_failed",
                content={"error": str(e), "query_id": query_data.get("query_id")},
                timestamp=datetime.now(),
                message_id=f"error_{query_data.get('query_id')}"
            )

class ResponderAgent(BaseAgent):
    """Specialized agent for response generation"""
    
    def __init__(self, email_processor):
        super().__init__(
            agent_id="responder",
            capabilities=["response_generation", "rag_retrieval", "email_composition"]
        )
        self.email_processor = email_processor
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process response generation requests"""
        self.log_message(message)
        
        if message.message_type == "generate_response":
            return self._generate_response(message.content)
        
        return None
    
    def _generate_response(self, query_data: Dict) -> AgentMessage:
        """Generate response using email processor and RAG"""
        try:
            email_content = query_data.get("email_content", {})
            classification = query_data.get("classification", {})
            
            # Generate response using email processor
            processed_result = self.email_processor.process_email(email_content)
            
            result = {
                "query_id": query_data.get("query_id"),
                "response": {
                    "generated_response": processed_result.get("response", ""),
                    "rag_sources": processed_result.get("rag_sources", []),
                    "confidence": processed_result.get("confidence", 0.0),
                    "requires_review": processed_result.get("confidence", 0.0) < 0.8
                },
                "success": processed_result.get("confidence", 0.0) > 0.6,
                "next_agent": "validator" if processed_result.get("confidence", 0.0) > 0.6 else "escalation_handler"
            }
            
            return AgentMessage(
                sender=self.agent_id,
                receiver="planner",
                message_type="response_generated",
                content=result,
                timestamp=datetime.now(),
                message_id=f"response_{query_data.get('query_id')}"
            )
            
        except Exception as e:
            logging.error(f"Response generation error: {e}")
            return AgentMessage(
                sender=self.agent_id,
                receiver="planner",
                message_type="task_failed",
                content={"error": str(e), "query_id": query_data.get("query_id")},
                timestamp=datetime.now(),
                message_id=f"error_{query_data.get('query_id')}"
            )

class ValidatorAgent(BaseAgent):
    """Specialized agent for response validation"""
    
    def __init__(self, evaluation_system):
        super().__init__(
            agent_id="validator",
            capabilities=["response_validation", "quality_assessment", "approval_recommendation"]
        )
        self.evaluation_system = evaluation_system
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process validation requests"""
        self.log_message(message)
        
        if message.message_type == "validate_response":
            return self._validate_response(message.content)
        
        return None
    
    def _validate_response(self, validation_data: Dict) -> AgentMessage:
        """Validate response quality"""
        try:
            response = validation_data.get("response", {})
            query_data = validation_data.get("query_data", {})
            
            # Evaluate response quality
            quality_metrics = self.evaluation_system.evaluate_response_quality(
                response.get("generated_response", ""),
                query_data.get("email_content", {}).get("body", ""),
                response.get("rag_sources", [])
            )
            
            # Determine if response meets quality thresholds
            quality_threshold = 0.7
            approval_recommended = quality_metrics.get("overall", 0.0) >= quality_threshold
            
            result = {
                "query_id": validation_data.get("query_id"),
                "validation": {
                    "quality_score": quality_metrics.get("overall", 0.0),
                    "detailed_scores": quality_metrics,
                    "approval_recommended": approval_recommended,
                    "requires_human_review": not approval_recommended,
                    "validation_timestamp": datetime.now().isoformat()
                },
                "success": True,
                "next_step": "approve_and_send" if approval_recommended else "human_review"
            }
            
            return AgentMessage(
                sender=self.agent_id,
                receiver="planner",
                message_type="validation_complete",
                content=result,
                timestamp=datetime.now(),
                message_id=f"validation_{validation_data.get('query_id')}"
            )
            
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return AgentMessage(
                sender=self.agent_id,
                receiver="planner",
                message_type="task_failed",
                content={"error": str(e), "query_id": validation_data.get("query_id")},
                timestamp=datetime.now(),
                message_id=f"error_{validation_data.get('query_id')}"
            )

class EscalationAgent(BaseAgent):
    """Specialized agent for handling escalations and human handoff"""
    
    def __init__(self, notification_system=None):
        super().__init__(
            agent_id="escalation_handler",
            capabilities=["escalation_management", "human_notification", "priority_assessment"]
        )
        self.notification_system = notification_system
        self.pending_escalations = []
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process escalation requests"""
        self.log_message(message)
        
        if message.message_type == "escalate_query":
            return self._handle_escalation(message.content)
        elif message.message_type == "escalation_resolved":
            return self._handle_resolution(message.content)
        
        return None
    
    def _handle_escalation(self, escalation_data: Dict) -> AgentMessage:
        """Handle query escalation to human"""
        try:
            # Assess priority
            priority = self._assess_priority(escalation_data)
            
            escalation_record = {
                "escalation_id": f"esc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "query_id": escalation_data.get("query_id"),
                "reason": escalation_data.get("reason", "low_confidence"),
                "priority": priority,
                "escalated_at": datetime.now().isoformat(),
                "context": escalation_data,
                "status": "pending_human_review"
            }
            
            self.pending_escalations.append(escalation_record)
            
            # Send notification if system available
            if self.notification_system:
                self.notification_system.send_escalation_notification(escalation_record)
            
            result = {
                "escalation_id": escalation_record["escalation_id"],
                "status": "escalated_to_human",
                "priority": priority,
                "estimated_review_time": self._estimate_review_time(priority),
                "human_action_required": True
            }
            
            return AgentMessage(
                sender=self.agent_id,
                receiver="coordinator",
                message_type="escalation_processed",
                content=result,
                timestamp=datetime.now(),
                message_id=f"escalation_{escalation_record['escalation_id']}"
            )
            
        except Exception as e:
            logging.error(f"Escalation handling error: {e}")
            return None
    
    def _assess_priority(self, escalation_data: Dict) -> str:
        """Assess escalation priority"""
        # Simple priority assessment logic
        reason = escalation_data.get("reason", "")
        confidence = escalation_data.get("confidence", 0.0)
        
        if "urgent" in reason.lower() or confidence < 0.3:
            return "high"
        elif confidence < 0.5:
            return "medium"
        else:
            return "normal"
    
    def _estimate_review_time(self, priority: str) -> str:
        """Estimate human review time based on priority"""
        time_estimates = {
            "high": "within 1 hour",
            "medium": "within 4 hours", 
            "normal": "within 24 hours"
        }
        return time_estimates.get(priority, "within 24 hours")
    
    def get_pending_escalations(self) -> List[Dict]:
        """Get list of pending escalations"""
        return [esc for esc in self.pending_escalations if esc["status"] == "pending_human_review"]

class AgentCoordinator:
    """Coordinates communication and task flow between agents"""
    
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self.message_queue = []
        self.active_workflows = {}
        
    def add_agent(self, agent: BaseAgent):
        """Add agent to the system"""
        self.agents[agent.agent_id] = agent
        
    def process_query(self, query_data: Dict) -> Dict:
        """Process query through the multi-agent system"""
        query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        query_data["query_id"] = query_id
        
        # Start with planner
        initial_message = AgentMessage(
            sender="coordinator",
            receiver="planner",
            message_type="new_query",
            content=query_data,
            timestamp=datetime.now(),
            message_id=f"init_{query_id}"
        )
        
        # Track workflow
        self.active_workflows[query_id] = {
            "status": "processing",
            "current_agent": "planner",
            "start_time": datetime.now(),
            "messages": [initial_message]
        }
        
        # Process through agents
        result = self._execute_workflow(query_id, initial_message)
        
        return result
    
    def _execute_workflow(self, query_id: str, initial_message: AgentMessage) -> Dict:
        """Execute the multi-agent workflow"""
        current_message = initial_message
        max_iterations = 10  # Prevent infinite loops
        iterations = 0
        
        while current_message and iterations < max_iterations:
            iterations += 1
            
            # Find target agent
            receiver_agent = self.agents.get(current_message.receiver)
            if not receiver_agent:
                logging.error(f"Agent not found: {current_message.receiver}")
                break
            
            # Process message
            response = receiver_agent.process_message(current_message)
            
            # Update workflow tracking
            if query_id in self.active_workflows:
                self.active_workflows[query_id]["messages"].append(current_message)
                if response:
                    self.active_workflows[query_id]["current_agent"] = response.receiver
            
            # Handle coordinator messages
            if response and response.receiver == "coordinator":
                result = self._handle_coordinator_message(query_id, response)
                if result.get("completed", False):
                    return result
            
            current_message = response
        
        # Workflow completed or terminated
        if query_id in self.active_workflows:
            self.active_workflows[query_id]["status"] = "completed"
            self.active_workflows[query_id]["end_time"] = datetime.now()
        
        return {
            "query_id": query_id,
            "status": "completed",
            "iterations": iterations,
            "workflow": self.active_workflows.get(query_id, {})
        }
    
    def _handle_coordinator_message(self, query_id: str, message: AgentMessage) -> Dict:
        """Handle messages directed to coordinator"""
        
        if message.message_type == "processing_plan":
            # Execute the processing plan
            return self._execute_plan(query_id, message.content)
        elif message.message_type == "escalation_processed":
            # Query has been escalated
            return {
                "query_id": query_id,
                "status": "escalated",
                "escalation_details": message.content,
                "completed": True
            }
        elif message.message_type == "response_approved":
            # Response approved for sending
            return {
                "query_id": query_id,
                "status": "response_ready",
                "response": message.content,
                "completed": True
            }
        
        return {"completed": False}
    
    def _execute_plan(self, query_id: str, plan: Dict) -> Dict:
        """Execute processing plan"""
        # This is a simplified execution - in practice you'd implement
        # proper sequential execution of plan steps
        return {"completed": False}
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        agent_status = {
            agent_id: {
                "status": agent.status,
                "capabilities": agent.capabilities,
                "message_count": len(agent.message_history)
            }
            for agent_id, agent in self.agents.items()
        }
        
        return {
            "active_workflows": len([w for w in self.active_workflows.values() if w["status"] == "processing"]),
            "completed_workflows": len([w for w in self.active_workflows.values() if w["status"] == "completed"]),
            "agents": agent_status,
            "system_health": "healthy" if all(agent.status != "error" for agent in self.agents.values()) else "degraded"
        }