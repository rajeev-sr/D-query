from src.model_inference import QueryClassifier
from typing import Dict, List
import logging

class DecisionEngine:
    def __init__(self, model_path="models/fine_tuned"):
        self.classifier = QueryClassifier(model_path)
        self.confidence_thresholds = {
            'auto_respond': 0.8,
            'review_needed': 0.5,
            'escalate': 0.0
        }
        
        # Keywords for additional filtering
        self.sensitive_keywords = [
            'complaint', 'urgent', 'emergency', 'problem',
            'issue', 'wrong', 'error', 'mistake', 'confused'
        ]
        
        self.simple_patterns = {
            'academic': ['exam', 'test', 'assignment', 'homework', 'grade', 'marks'],
            'administrative': [
                'registration', 'enrollment', 'fee', 'certificate',
                'admission', 'scholarship', 'hostel', 'leave', 'attendance',
                'payment', 'refund', 'disciplinary', 'convocation', 'identity card',
                'bonafide', 'withdrawal', 'rules', 'notice', 'office',
                'document', 'approval', 'clearance', 'committee', 'council', 'quota',
                'mess', 'canteen', 'library', 'holiday', 'schedule', 'deadline',
                'reminder', 'update', 'announcement', 'policy', 'guidelines', 'form',
                'submission', 'verification', 'process', 'status', 'result', 'marksheet',
                'hostel allotment', 'room change', 'discipline', 'security', 'medical',
                'insurance', 'transport', 'bus', 'parking', 'gatepass', 'visitor', 'parent'
            ],
            'technical': ['login', 'password', 'access', 'website', 'portal', 'system']
        }
    
    def process_query(self, email_data: Dict) -> Dict:
        """Process email query and make decision"""
        
        query_text = f"{email_data.get('subject', '')} {email_data.get('body', '')}"
        
        # Get AI classification
        ai_result = self.classifier.classify_and_respond(query_text)
        
        if "error" in ai_result:
            return {
                "action": "escalate",
                "reason": f"AI classification failed: {ai_result['error']}",
                "confidence": 0.0,
                "response": None
            }
        
        # Apply business rules
        final_decision = self._apply_business_rules(ai_result, email_data)
        
        return final_decision
    
    def _apply_business_rules(self, ai_result: Dict, email_data: Dict) -> Dict:
        """Apply business rules to refine AI decision"""
        
        query_text = f"{email_data.get('subject', '')} {email_data.get('body', '')}".lower()
        
        # Rule 1: Check for sensitive keywords
        if any(keyword in query_text for keyword in self.sensitive_keywords):
            return {
                "action": "escalate",
                "reason": "Contains sensitive keywords",
                "confidence": 1.0,
                "ai_category": ai_result['category'],
                "ai_response": ai_result['response'],
                "response": None
            }
        
        # Rule 2: Low confidence threshold
        if ai_result['confidence'] < self.confidence_thresholds['review_needed']:
            return {
                "action": "escalate",
                "reason": f"Low confidence: {ai_result['confidence']:.2f}",
                "confidence": ai_result['confidence'],
                "ai_category": ai_result['category'],
                "ai_response": ai_result['response'],
                "response": None
            }
        
        # Rule 3: High confidence for simple categories
        if (ai_result['confidence'] >= self.confidence_thresholds['auto_respond'] and
            ai_result['category'] in ['academic', 'administrative']):
            
            return {
                "action": "auto_respond",
                "reason": f"High confidence classification: {ai_result['category']}",
                "confidence": ai_result['confidence'],
                "category": ai_result['category'],
                "response": ai_result['response']
            }
        
        # Rule 4: Medium confidence - needs review
        return {
            "action": "review_needed",
            "reason": f"Medium confidence: {ai_result['confidence']:.2f}",
            "confidence": ai_result['confidence'],
            "category": ai_result['category'],
            "response": ai_result['response']
        }
    
    def batch_process(self, emails: List[Dict]) -> List[Dict]:
        """Process multiple emails"""
        results = []
        
        for email in emails:
            try:
                result = self.process_query(email)
                result['email_id'] = email.get('id', 'unknown')
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing email {email.get('id')}: {e}")
                results.append({
                    "email_id": email.get('id', 'unknown'),
                    "action": "escalate",
                    "reason": f"Processing error: {str(e)}",
                    "confidence": 0.0
                })
        
        return results