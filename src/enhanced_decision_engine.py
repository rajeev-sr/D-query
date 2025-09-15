# src/enhanced_decision_engine.py
from src.enhanced_classifier import EnhancedQueryClassifier
from src.decision_engine import DecisionEngine
from typing import Dict, List
import logging

class EnhancedDecisionEngine(DecisionEngine):
    def __init__(self, model_path="models/fine_tuned", docs_dir="data/institute_docs"):
        # Initialize with enhanced classifier
        self.classifier = EnhancedQueryClassifier(model_path, docs_dir)
        
        # Decision thresholds (more lenient for testing)
        self.confidence_thresholds = {
            'auto_respond': 0.60,  # Lowered threshold for auto-response
            'review_needed': 0.3,   # Lowered threshold for review
            'escalate': 0.0
        }
        
        # Keywords for filtering - only truly sensitive items that need human attention
        self.sensitive_keywords = [
            'complaint', 'urgent', 'emergency', 'serious problem',
            'very wrong', 'major error', 'big mistake', 'very confused',
            'angry', 'disappointed', 'frustrated', 'nobody is helping', 
            'very urgent', 'critical issue', 'immediate attention'
        ]
        
        # High-confidence categories for auto-response
        self.auto_response_categories = ['academic', 'administrative', 'fee', 'technical', 'registration']
    
    def process_query(self, email_data: Dict) -> Dict:
        """Enhanced query processing with RAG context"""
        
        query_text = f"{email_data.get('subject', '')} {email_data.get('body', '')}"
        
        # Get enhanced AI classification with RAG
        try:
            ai_result = self.classifier.classify_and_respond_with_context(query_text)
        except Exception as e:
            logging.error(f"Enhanced classification failed: {e}")
            return {
                "action": "escalate",
                "reason": f"AI processing failed: {str(e)}",
                "confidence": 0.0,
                "response": None
            }
        
        if "error" in ai_result:
            return {
                "action": "escalate",
                "reason": f"AI classification failed: {ai_result['error']}",
                "confidence": 0.0,
                "response": None
            }
        
        # Apply enhanced business rules
        final_decision = self._apply_enhanced_business_rules(ai_result, email_data)
        
        return final_decision
    
    def _apply_enhanced_business_rules(self, ai_result: Dict, email_data: Dict) -> Dict:
        """Enhanced business rules considering RAG context"""
        
        query_text = f"{email_data.get('subject', '')} {email_data.get('body', '')}".lower()
        rag_context = ai_result.get('rag_context', {})
        
        # Rule 1: Always escalate sensitive keywords
        if any(keyword in query_text for keyword in self.sensitive_keywords):
            return {
                "action": "escalate",
                "reason": "Contains sensitive keywords requiring human attention",
                "confidence": 1.0,
                "ai_category": ai_result.get('category'),
                "ai_response": ai_result.get('response'),
                "rag_sources": rag_context.get('sources', []),
                "response": None
            }
        
        # Rule 2: Check for auto-response eligibility
        ai_confidence = ai_result.get('confidence', 0)
        ai_category = ai_result.get('category')
        
        # High confidence AI response can auto-respond
        if (ai_confidence >= self.confidence_thresholds['auto_respond'] and
            ai_category in self.auto_response_categories):
            
            return {
                "action": "auto_respond",
                "reason": f"High AI confidence with good category (confidence: {ai_confidence:.2f})",
                "confidence": ai_confidence,
                "category": ai_category,
                "response": ai_result.get('response'),
                "rag_sources": rag_context.get('sources', []),
                "enhanced": ai_result.get('enhanced', False)
            }
        
        # Rule 3: RAG context quality boost
        context_confidence = rag_context.get('context_confidence', 0)
        context_used = rag_context.get('context_used', False)
        
        if context_used and context_confidence > 0.7:
            # High quality RAG context available - boost confidence
            boosted_confidence = min(1.0, ai_confidence + 0.2)
            
            if (boosted_confidence >= self.confidence_thresholds['auto_respond'] and
                ai_category in self.auto_response_categories):
                
                return {
                    "action": "auto_respond",
                    "reason": f"High confidence with verified context (confidence: {boosted_confidence:.2f})",
                    "confidence": boosted_confidence,
                    "category": ai_category,
                    "response": ai_result.get('response'),
                    "rag_sources": rag_context.get('sources', []),
                    "enhanced": ai_result.get('enhanced', False)
                }
        
        # Rule 4: Low confidence or no RAG context
        if (context_confidence < 0.3 or not context_used or 
            ai_confidence < self.confidence_thresholds['review_needed']):
            
            return {
                "action": "escalate",
                "reason": f"Low confidence or insufficient context (AI: {ai_confidence:.2f}, RAG: {context_confidence:.2f})",
                "confidence": ai_confidence,
                "ai_category": ai_category,
                "ai_response": ai_result.get('response'),
                "rag_sources": rag_context.get('sources', []),
                "response": None
            }
        
        # Rule 5: Medium confidence - needs review
        return {
            "action": "review_needed",
            "reason": f"Medium confidence requiring review (AI: {ai_confidence:.2f}, RAG: {context_confidence:.2f})",
            "confidence": ai_confidence,
            "category": ai_category,
            "response": ai_result.get('response'),
            "rag_sources": rag_context.get('sources', []),
            "enhanced": ai_result.get('enhanced', False)
        }
    
    def get_system_stats(self) -> Dict:
        """Get enhanced system statistics"""
        base_stats = {
            'model_loaded': self.classifier.model is not None,
            'rag_enabled': self.classifier.rag_enabled
        }
        
        if self.classifier.rag_enabled:
            rag_stats = self.classifier.rag_system.vector_db.get_stats()
            base_stats.update({
                'knowledge_base': rag_stats,
                'total_documents': rag_stats.get('total_chunks', 0)
            })
        
        return base_stats