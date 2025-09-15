# src/evaluation_system.py
"""
Comprehensive evaluation system for AI Query Handler
Measures quality and reliability of agent outputs
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict
import re

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Classification Metrics
    classification_accuracy: float = 0.0
    classification_precision: float = 0.0
    classification_recall: float = 0.0
    classification_f1: float = 0.0
    
    # Response Quality Metrics
    response_relevance_score: float = 0.0
    response_helpfulness_score: float = 0.0
    response_completeness_score: float = 0.0
    response_clarity_score: float = 0.0
    
    # System Performance Metrics
    avg_response_time_seconds: float = 0.0
    successful_resolutions: int = 0
    escalation_rate: float = 0.0
    user_satisfaction_score: float = 0.0
    
    # RAG System Metrics
    rag_retrieval_accuracy: float = 0.0
    rag_source_relevance: float = 0.0
    knowledge_coverage: float = 0.0
    
    # Automation Metrics
    automation_rate: float = 0.0
    human_intervention_rate: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

class EvaluationSystem:
    """Comprehensive evaluation system for the AI Query Handler"""
    
    def __init__(self):
        self.evaluation_history = []
        self.metrics_file = "data/evaluation_metrics.json"
        self.benchmark_responses = self._load_benchmark_data()
        
    def _load_benchmark_data(self) -> Dict:
        """Load benchmark responses for evaluation"""
        benchmark_path = "data/benchmark_responses.json"
        try:
            if os.path.exists(benchmark_path):
                with open(benchmark_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load benchmark data: {e}")
        
        # Create sample benchmark data if not exists
        return {
            "sample_queries": [
                {
                    "query": "What are the admission requirements for Computer Science?",
                    "expected_category": "academic_query",
                    "expected_keywords": ["admission", "requirements", "computer science"],
                    "quality_score": 0.9
                },
                {
                    "query": "How to submit assignment for Data Structures course?",
                    "expected_category": "academic_query", 
                    "expected_keywords": ["assignment", "submission", "data structures"],
                    "quality_score": 0.85
                }
            ]
        }
    
    def evaluate_classification(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate classification performance"""
        if not predictions or not ground_truth:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        correct = 0
        total = min(len(predictions), len(ground_truth))
        
        for pred, truth in zip(predictions, ground_truth):
            if pred.get('category') == truth.get('category'):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        # For simplicity, using accuracy as proxy for precision/recall
        # In production, you'd calculate per-class metrics
        return {
            "accuracy": accuracy,
            "precision": accuracy,  # Simplified
            "recall": accuracy,     # Simplified
            "f1": accuracy         # Simplified
        }
    
    def evaluate_response_quality(self, response: str, query: str, rag_sources: List[str]) -> Dict[str, float]:
        """Evaluate response quality using multiple metrics"""
        
        # 1. Relevance Score (keyword overlap)
        relevance_score = self._calculate_relevance_score(response, query)
        
        # 2. Helpfulness Score (presence of actionable information)
        helpfulness_score = self._calculate_helpfulness_score(response)
        
        # 3. Completeness Score (response length and structure)
        completeness_score = self._calculate_completeness_score(response, query)
        
        # 4. Clarity Score (readability and structure)
        clarity_score = self._calculate_clarity_score(response)
        
        return {
            "relevance": relevance_score,
            "helpfulness": helpfulness_score,
            "completeness": completeness_score,
            "clarity": clarity_score,
            "overall": (relevance_score + helpfulness_score + completeness_score + clarity_score) / 4
        }
    
    def _calculate_relevance_score(self, response: str, query: str) -> float:
        """Calculate relevance based on keyword overlap"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        query_words = query_words - stop_words
        response_words = response_words - stop_words
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(response_words))
        return overlap / len(query_words)
    
    def _calculate_helpfulness_score(self, response: str) -> float:
        """Calculate helpfulness based on actionable content"""
        helpful_indicators = [
            "you can", "please", "contact", "visit", "follow", "submit", "apply",
            "steps", "process", "procedure", "requirements", "deadline", "office hours"
        ]
        
        response_lower = response.lower()
        helpful_count = sum(1 for indicator in helpful_indicators if indicator in response_lower)
        
        # Normalize by response length and cap at 1.0
        base_score = helpful_count / max(len(helpful_indicators), 1)
        length_bonus = min(0.2, len(response.split()) / 100)  # Bonus for detailed responses
        
        return min(1.0, base_score + length_bonus)
    
    def _calculate_completeness_score(self, response: str, query: str) -> float:
        """Calculate completeness based on response comprehensiveness"""
        # Check for question answering completeness
        has_direct_answer = any(word in response.lower() for word in ["yes", "no", "required", "available", "procedure"])
        has_details = len(response.split()) > 20
        has_contact_info = any(word in response.lower() for word in ["contact", "email", "phone", "office"])
        
        completeness_factors = [has_direct_answer, has_details, has_contact_info]
        return sum(completeness_factors) / len(completeness_factors)
    
    def _calculate_clarity_score(self, response: str) -> float:
        """Calculate clarity based on readability"""
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()])
        
        # Optimal sentence length is 15-25 words
        if 15 <= avg_sentence_length <= 25:
            length_score = 1.0
        elif 10 <= avg_sentence_length <= 30:
            length_score = 0.8
        else:
            length_score = 0.6
        
        # Check for good structure (paragraphs, bullet points)
        has_structure = bool(re.search(r'[\n\r]|\d+\.|\*|\-', response))
        structure_score = 1.0 if has_structure else 0.7
        
        return (length_score + structure_score) / 2
    
    def evaluate_rag_system(self, query: str, retrieved_sources: List[str], response: str) -> Dict[str, float]:
        """Evaluate RAG system performance"""
        
        # 1. Retrieval Accuracy (relevant sources retrieved)
        retrieval_accuracy = self._calculate_retrieval_accuracy(query, retrieved_sources)
        
        # 2. Source Relevance (quality of retrieved sources)
        source_relevance = self._calculate_source_relevance(retrieved_sources, response)
        
        # 3. Knowledge Coverage (how well sources cover the query)
        knowledge_coverage = self._calculate_knowledge_coverage(query, retrieved_sources)
        
        return {
            "retrieval_accuracy": retrieval_accuracy,
            "source_relevance": source_relevance,
            "knowledge_coverage": knowledge_coverage
        }
    
    def _calculate_retrieval_accuracy(self, query: str, sources: List[str]) -> float:
        """Calculate retrieval accuracy"""
        if not sources:
            return 0.0
        
        query_words = set(query.lower().split())
        relevant_sources = 0
        
        for source in sources:
            source_words = set(source.lower().split())
            overlap = len(query_words.intersection(source_words))
            if overlap >= 2:  # At least 2 word overlap
                relevant_sources += 1
        
        return relevant_sources / len(sources)
    
    def _calculate_source_relevance(self, sources: List[str], response: str) -> float:
        """Calculate how relevant sources are to the response"""
        if not sources:
            return 0.0
        
        response_words = set(response.lower().split())
        relevant_sources = 0
        
        for source in sources:
            source_words = set(source.lower().split())
            overlap = len(response_words.intersection(source_words))
            if overlap >= 3:  # At least 3 word overlap
                relevant_sources += 1
        
        return relevant_sources / len(sources)
    
    def _calculate_knowledge_coverage(self, query: str, sources: List[str]) -> float:
        """Calculate knowledge coverage"""
        # Simple heuristic: more sources = better coverage
        if len(sources) >= 3:
            return 1.0
        elif len(sources) >= 2:
            return 0.8
        elif len(sources) >= 1:
            return 0.6
        else:
            return 0.0
    
    def evaluate_system_performance(self, processing_log: List[Dict]) -> Dict[str, Any]:
        """Evaluate overall system performance"""
        if not processing_log:
            return {"avg_response_time": 0.0, "automation_rate": 0.0, "escalation_rate": 0.0}
        
        # Calculate response times
        response_times = []
        automated_responses = 0
        escalations = 0
        successful_resolutions = 0
        
        for log_entry in processing_log:
            if 'response_time' in log_entry:
                response_times.append(log_entry['response_time'])
            
            action = log_entry.get('action', '')
            if action == 'auto_respond':
                automated_responses += 1
                successful_resolutions += 1
            elif action == 'escalate':
                escalations += 1
        
        total_processed = len(processing_log)
        
        return {
            "avg_response_time": np.mean(response_times) if response_times else 0.0,
            "automation_rate": automated_responses / total_processed if total_processed > 0 else 0.0,
            "escalation_rate": escalations / total_processed if total_processed > 0 else 0.0,
            "successful_resolutions": successful_resolutions,
            "total_processed": total_processed
        }
    
    def comprehensive_evaluation(self, system_data: Dict) -> EvaluationMetrics:
        """Perform comprehensive evaluation of the entire system"""
        
        # Extract data
        processing_log = system_data.get('processing_log', [])
        classifications = system_data.get('classifications', [])
        responses = system_data.get('responses', [])
        rag_data = system_data.get('rag_data', [])
        
        # Evaluate classification
        classification_metrics = self.evaluate_classification(
            classifications, 
            self.benchmark_responses.get('sample_queries', [])
        )
        
        # Evaluate response quality
        response_quality_scores = []
        for response_data in responses:
            quality = self.evaluate_response_quality(
                response_data.get('response', ''),
                response_data.get('query', ''),
                response_data.get('rag_sources', [])
            )
            response_quality_scores.append(quality['overall'])
        
        avg_response_quality = np.mean(response_quality_scores) if response_quality_scores else 0.0
        
        # Evaluate RAG system
        rag_scores = []
        for rag_entry in rag_data:
            rag_eval = self.evaluate_rag_system(
                rag_entry.get('query', ''),
                rag_entry.get('sources', []),
                rag_entry.get('response', '')
            )
            rag_scores.append(rag_eval)
        
        avg_rag_accuracy = np.mean([score['retrieval_accuracy'] for score in rag_scores]) if rag_scores else 0.0
        avg_source_relevance = np.mean([score['source_relevance'] for score in rag_scores]) if rag_scores else 0.0
        avg_knowledge_coverage = np.mean([score['knowledge_coverage'] for score in rag_scores]) if rag_scores else 0.0
        
        # Evaluate system performance
        performance_metrics = self.evaluate_system_performance(processing_log)
        
        # Create comprehensive metrics
        metrics = EvaluationMetrics(
            classification_accuracy=classification_metrics['accuracy'],
            classification_precision=classification_metrics['precision'],
            classification_recall=classification_metrics['recall'],
            classification_f1=classification_metrics['f1'],
            
            response_relevance_score=avg_response_quality,
            response_helpfulness_score=avg_response_quality,  # Simplified
            response_completeness_score=avg_response_quality,  # Simplified
            response_clarity_score=avg_response_quality,  # Simplified
            
            avg_response_time_seconds=performance_metrics['avg_response_time'],
            successful_resolutions=performance_metrics['successful_resolutions'],
            escalation_rate=performance_metrics['escalation_rate'],
            automation_rate=performance_metrics['automation_rate'],
            
            rag_retrieval_accuracy=avg_rag_accuracy,
            rag_source_relevance=avg_source_relevance,
            knowledge_coverage=avg_knowledge_coverage,
            
            human_intervention_rate=performance_metrics['escalation_rate'],
            user_satisfaction_score=0.85  # Placeholder - would come from user feedback
        )
        
        # Save evaluation
        self._save_evaluation(metrics)
        
        return metrics
    
    def _save_evaluation(self, metrics: EvaluationMetrics):
        """Save evaluation results"""
        evaluation_record = {
            "timestamp": datetime.now().isoformat(),
            "metrics": asdict(metrics)
        }
        
        self.evaluation_history.append(evaluation_record)
        
        # Save to file
        os.makedirs("data", exist_ok=True)
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.evaluation_history, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save evaluation metrics: {e}")
    
    def get_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_history:
            return {"message": "No evaluation data available"}
        
        latest_metrics = self.evaluation_history[-1]['metrics']
        
        # Performance categorization
        def categorize_score(score, thresholds=[0.3, 0.6, 0.8]):
            if score >= thresholds[2]:
                return "Excellent"
            elif score >= thresholds[1]:
                return "Good"
            elif score >= thresholds[0]:
                return "Fair"
            else:
                return "Needs Improvement"
        
        report = {
            "evaluation_summary": {
                "classification_performance": categorize_score(latest_metrics['classification_accuracy']),
                "response_quality": categorize_score(latest_metrics['response_relevance_score']),
                "system_efficiency": categorize_score(latest_metrics['automation_rate']),
                "rag_performance": categorize_score(latest_metrics['rag_retrieval_accuracy']),
                "overall_score": (
                    latest_metrics['classification_accuracy'] +
                    latest_metrics['response_relevance_score'] +
                    latest_metrics['automation_rate'] +
                    latest_metrics['rag_retrieval_accuracy']
                ) / 4
            },
            "detailed_metrics": latest_metrics,
            "recommendations": self._generate_recommendations(latest_metrics),
            "trends": self._analyze_trends() if len(self.evaluation_history) > 1 else {},
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        if metrics['classification_accuracy'] < 0.8:
            recommendations.append("Consider retraining the classification model with more diverse data")
        
        if metrics['response_relevance_score'] < 0.7:
            recommendations.append("Improve response quality by enhancing prompt engineering")
        
        if metrics['automation_rate'] < 0.6:
            recommendations.append("Review escalation criteria to increase automation rate")
        
        if metrics['rag_retrieval_accuracy'] < 0.7:
            recommendations.append("Update knowledge base and improve document indexing")
        
        if metrics['escalation_rate'] > 0.4:
            recommendations.append("Consider expanding automated response capabilities")
        
        if not recommendations:
            recommendations.append("System is performing well! Continue monitoring.")
        
        return recommendations
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in evaluation metrics over time"""
        if len(self.evaluation_history) < 2:
            return {}
        
        recent_metrics = [entry['metrics'] for entry in self.evaluation_history[-5:]]
        
        trends = {}
        key_metrics = ['classification_accuracy', 'response_relevance_score', 'automation_rate', 'escalation_rate']
        
        for metric in key_metrics:
            values = [m[metric] for m in recent_metrics]
            if len(values) > 1:
                trend = "improving" if values[-1] > values[0] else "declining" if values[-1] < values[0] else "stable"
                trends[metric] = {
                    "trend": trend,
                    "change": values[-1] - values[0],
                    "current": values[-1]
                }
        
        return trends