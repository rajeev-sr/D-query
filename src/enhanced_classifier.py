from src.model_inference import QueryClassifier
from src.rag_system import RAGSystem
from typing import Dict, Any
import re
import torch
import logging
from typing import List

logger = logging.getLogger(__name__)

class EnhancedQueryClassifier(QueryClassifier):
    def __init__(self, model_path="models/fine_tuned", docs_dir="data/institute_docs"):
        super().__init__(model_path)
        self.rag_system = RAGSystem(docs_dir)
        self.rag_enabled = False
        
        # Setup RAG system
        self._setup_rag()
    
    def _setup_rag(self):
        """Setup RAG system"""
        try:
            if self.rag_system.setup_knowledge_base():
                self.rag_enabled = True
                print("RAG system enabled")
            else:
                print("RAG system disabled - continuing without context")
        except Exception as e:
            print(f"RAG setup failed: {e}")
    
    def classify_and_respond_with_context(self, query: str, max_length: int = 512) -> Dict:
        """Enhanced classification with RAG context"""
        
        # Get base classification first
        base_result = self.classify_and_respond(query, max_length)
        
        if "error" in base_result:
            return base_result
        
        # Enhance with RAG context if available
        if self.rag_enabled:
            try:
                context_result = self.rag_system.retrieve_context(query)
                
                # Regenerate response with context
                enhanced_result = self._generate_with_context(query, context_result, base_result)
                
                # Add RAG metadata
                enhanced_result['rag_context'] = {
                    'sources': context_result.get('sources', []),
                    'context_confidence': context_result.get('confidence', 0),
                    'context_used': bool(context_result.get('context'))
                }
                
                return enhanced_result
                
            except Exception as e:
                print(f"RAG enhancement failed: {e}")
                # Return base result as fallback
                base_result['rag_context'] = {'error': str(e)}
                return base_result
        
        else:
            # Return base result without RAG
            base_result['rag_context'] = {'enabled': False}
            return base_result
    
    def _generate_with_context(self, query: str, context_result: Dict, base_result: Dict) -> Dict:
        """Generate response using both model and RAG context"""
        
        context = context_result.get('context', '')
        
        if not context or context_result.get('confidence', 0) < 0.3:
            # Low quality context - use base result
            return base_result
        
        # Create enhanced prompt with context
        enhanced_instruction = f"""You are an institute assistant with access to official documentation. 
            Use the following official information to answer the student query accurately.

            Official Information:
            {context}

            Student Query: {query}

            Provide a helpful and accurate response based on the official information above."""
                    
        # Clean input without special tokens
        inputs = self.tokenizer(
            enhanced_instruction,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            # Generate enhanced response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.5,  # Lower temperature for more factual responses
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = full_response.split(enhanced_instruction)[-1].strip()
            
            # Clean up the generated response
            generated_part = self._clean_response(generated_part)
            
            # Create enhanced result
            enhanced_result = base_result.copy()
            
            # Use RAG-enhanced response if it's better quality
            if len(generated_part) > 20 and not self._is_generic_response(generated_part):
                # Format as professional email
                professional_response = self._format_professional_email(generated_part, context_result)
                enhanced_result['response'] = professional_response
                enhanced_result['enhanced'] = True
                enhanced_result['confidence'] = min(1.0, base_result.get('confidence', 0.5) + 0.2)
            else:
                # Fallback to combining base response with context facts
                enhanced_result['response'] = self._format_professional_email(
                    self._combine_response_with_facts(base_result.get('response', ''), context_result),
                    context_result
                )
                enhanced_result['enhanced'] = True
            
            return enhanced_result
            
        except Exception as e:
            print(f"Enhanced generation error: {e}")
            return base_result
    
    def _is_generic_response(self, response: str) -> bool:
        """Check if response is too generic"""
        generic_phrases = [
            "i don't know", "not sure", "can't help", "contact", "sorry"
        ]
        return any(phrase in response.lower() for phrase in generic_phrases)
    
    def _combine_response_with_facts(self, base_response: str, context_result: Dict) -> str:
        """Combine base response with factual information"""
        
        context = context_result.get('context', '')
        sources = context_result.get('sources', [])
        
        if not context:
            return base_response
        
        # Extract key facts from context
        facts = self._extract_key_facts(context)
        
        # Combine response
        combined = base_response
        
        if facts:
            combined += "\n\nBased on official documentation:\n"
            for fact in facts[:3]:  # Limit to top 3 facts
                combined += f"• {fact}\n"
        
        if sources:
            combined += f"\nSources: {', '.join(sources[:2])}"  # Limit sources
        
        return combined
    
    def _extract_key_facts(self, context: str) -> List[str]:
        """Extract key facts from context"""
        
        # Simple fact extraction based on patterns
        facts = []
        
        # Look for dates
        date_matches = re.findall(r'([A-Z][a-z]+ \d{1,2}(?:-\d{1,2})?, \d{4})', context)
        for date in date_matches[:2]:
            facts.append(f"Date: {date}")
        
        # Look for fees/amounts
        fee_matches = re.findall(r'\$[\d,]+(?:\.\d{2})?', context)
        for fee in fee_matches[:2]:
            facts.append(f"Fee: {fee}")
        
        # Look for deadlines
        deadline_matches = re.findall(r'[Dd]eadline[:\s]+([^.\n]+)', context)
        for deadline in deadline_matches[:2]:
            facts.append(f"Deadline: {deadline.strip()}")
        
        # Look for contact information
        contact_matches = re.findall(r'[Cc]ontact[:\s]+([^.\n]+)', context)
        for contact in contact_matches[:1]:
            facts.append(f"Contact: {contact.strip()}")
        
        return facts[:4]  # Return maximum 4 facts
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response by removing unwanted tokens and text"""
        import re
        
        # Remove common unwanted patterns
        cleaned = response
        
        # Remove special tokens that might appear
        cleaned = re.sub(r'<\|.*?\|>', '', cleaned)
        cleaned = re.sub(r'<.*?>', '', cleaned)
        
        # Remove "startoftext", "endoftext" etc.
        cleaned = re.sub(r'(?i)(start|end)oftext', '', cleaned)
        
        # Remove repeated patterns
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        
        # Remove trailing dots or incomplete sentences
        cleaned = re.sub(r'\.\.\.$', '', cleaned.strip())
        
        # Remove "You are an institute assistant" if it appears
        cleaned = re.sub(r'(?i)you are an? institute assistant.*?\n', '', cleaned)
        
        # Remove instruction text that might leak through
        cleaned = re.sub(r'(?i)use the following.*?accurately\.\s*', '', cleaned)
        cleaned = re.sub(r'(?i)provide a helpful.*?above\.\s*', '', cleaned)
        
        return cleaned.strip()
    
    def _format_professional_email(self, response: str, context_info: dict = None) -> str:
        """Format the response into a professional email structure"""
        
        # Clean the response first
        response = self._clean_response(response)
        
        # Professional email template
        email_parts = []
        
        # Greeting
        email_parts.append("Dear Student,")
        email_parts.append("")
        email_parts.append("Thank you for your inquiry regarding our institute programs and services.")
        email_parts.append("")
        
        # Main content - structure the response
        if context_info and context_info.get('sources'):
            # Group content by source/topic
            content_sections = self._organize_content(response, context_info.get('sources', []))
            for section in content_sections:
                email_parts.append(section)
                email_parts.append("")
        else:
            # Add response as structured content
            if response.strip():
                email_parts.append("**Information Requested:**")
                email_parts.append("")
                # Format response with bullet points if it contains multiple pieces of info
                formatted_content = self._format_content_sections(response)
                email_parts.append(formatted_content)
                email_parts.append("")
        
        # Additional help offer
        email_parts.append("If you need more specific information or have additional questions, please feel free to contact our support team.")
        email_parts.append("")
        
        # Signature
        email_parts.append("Best regards,")
        email_parts.append("AI Assistant")
        email_parts.append("Institute Support Team")
        email_parts.append("")
        email_parts.append("---")
        email_parts.append("This is an automated response. If you need further assistance, please contact the support team directly.")
        
        return "\n".join(email_parts)
    
    def _format_content_sections(self, content: str) -> str:
        """Format content into organized sections"""
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Clean up line formatting
            line = line.replace('• From', '• **Source:**').replace(': - ', ':\n  - ')
            
            # Add proper bullet formatting
            if line.startswith('- '):
                line = f"  {line}"
            elif not line.startswith('•') and not line.startswith('  '):
                line = f"• {line}"
                
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _structure_with_llm(self, raw_content: str) -> str:
        """Use LLM to structure raw content into professional format"""
        if not raw_content or len(raw_content.strip()) < 10:
            return "Thank you for your inquiry. We have received your message and will provide you with the relevant information."
        
        # Create a structured prompt for the LLM
        structuring_prompt = f"""Transform the following raw information into a clear, professional, and well-structured email response for a student inquiry:

            Raw Information:
            {raw_content}

            Requirements:
            1. Write in a professional, helpful tone
            2. Structure information clearly with proper paragraphs
            3. Remove any redundant or repetitive information
            4. Make it concise but comprehensive
            5. Use proper grammar and formatting
            6. Start with acknowledging their query
            7. Organize information logically (fees, dates, procedures, etc.)
            8. End with an offer for further assistance

            Professional Response:"""

        try:
            # Generate structured response using the model
            if hasattr(self, 'model') and self.model:
                inputs = self.tokenizer.encode(
                    structuring_prompt, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=300,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )
                
                # Decode and clean the response
                structured_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the generated part (after the prompt)
                if "Professional Response:" in structured_response:
                    structured_response = structured_response.split("Professional Response:")[-1].strip()
                
                # Clean up the response
                structured_response = self._clean_response(structured_response)
                
                # Validate the structured response
                if len(structured_response) > 20 and self._is_professional_content(structured_response):
                    return structured_response
                
        except Exception as e:
            logger.debug(f"LLM structuring failed: {e}")
        
        # Fallback: Basic manual structuring
        return self._basic_structure_content(raw_content)
    
    def _basic_structure_content(self, content: str) -> str:
        """Basic content structuring as fallback"""
        if not content:
            return "Thank you for your inquiry. We have received your message and will provide you with the relevant information shortly."
        
        # Clean and organize the content
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if not lines:
            return "Thank you for your inquiry. We have received your message."
        
        structured_parts = []
        
        # Add acknowledgment
        structured_parts.append("Thank you for your inquiry regarding our institute programs and services.")
        structured_parts.append("")
        
        # Process and organize information
        fee_info = []
        date_info = []
        general_info = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['fee', 'cost', 'payment', 'tuition', '$', 'rs.', 'rupee']):
                fee_info.append(line)
            elif any(keyword in line.lower() for keyword in ['date', 'deadline', 'semester', 'academic', 'calendar']):
                date_info.append(line)
            else:
                general_info.append(line)
        
        # Add organized information
        if fee_info:
            structured_parts.append("**Fee Information:**")
            for info in fee_info[:3]:  # Limit to prevent overwhelming
                structured_parts.append(f"• {info}")
            structured_parts.append("")
        
        if date_info:
            structured_parts.append("**Important Dates:**")
            for info in date_info[:3]:
                structured_parts.append(f"• {info}")
            structured_parts.append("")
        
        if general_info:
            structured_parts.append("**Additional Information:**")
            for info in general_info[:3]:
                structured_parts.append(f"• {info}")
            structured_parts.append("")
        
        structured_parts.append("If you need more specific information or have additional questions, please feel free to contact our support team.")
        
        return '\n'.join(structured_parts)
    
    def _is_professional_content(self, content: str) -> bool:
        """Check if content is professional and appropriate"""
        if not content or len(content) < 10:
            return False
        
        # Check for unprofessional patterns
        unprofessional_patterns = [
            r'(?i)\buh\b', r'(?i)\bum\b', r'(?i)\bokay\b',
            r'(?i)\byeah\b', r'(?i)\bnah\b', r'(?i)\bdunno\b'
        ]
        
        for pattern in unprofessional_patterns:
            if re.search(pattern, content):
                return False
        
        return True