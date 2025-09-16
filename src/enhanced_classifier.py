from src.model_inference import QueryClassifier
from src.rag_system import RAGSystem
from src.gemini_filter import GeminiQueryFilter
from typing import Dict, Any
import re
import torch
import logging
from typing import List
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

class EnhancedQueryClassifier(QueryClassifier):
    def __init__(self, model_path="models/fine_tuned", docs_dir="data/institute_docs"):
        super().__init__(model_path)
        self.rag_system = RAGSystem(docs_dir)
        self.rag_enabled = False
        
        # Initialize Gemini for intelligent response generation
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.gemini_enabled = True
            print("Gemini LLM enabled for intelligent response generation")
        else:
            self.gemini_enabled = False
            print("Gemini LLM disabled - API key not found")
        
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
                # Format as professional email with original query
                professional_response = self._format_professional_email(generated_part, context_result, query)
                enhanced_result['response'] = professional_response
                enhanced_result['enhanced'] = True
                enhanced_result['confidence'] = min(1.0, base_result.get('confidence', 0.5) + 0.2)
            else:
                # Fallback to combining base response with context facts
                enhanced_result['response'] = self._format_professional_email(
                    self._combine_response_with_facts(base_result.get('response', ''), context_result),
                    context_result,
                    query
                )
                enhanced_result['enhanced'] = True
            
            return enhanced_result
            
        except Exception as e:
            print(f"Enhanced generation error: {e}")
            return base_result

    def _generate_intelligent_response_from_context(self, query: str, rag_context: str, context_info: Dict) -> str:
        """Generate intelligent response using Gemini LLM with RAG context"""
        if not self.gemini_enabled:
            return self._format_rag_context(rag_context)
        
        try:
            # Create comprehensive prompt for intelligent response generation
            prompt = f"""You are a professional AI assistant for an educational institution. You need to respond to a student query using the provided official documentation.

STUDENT QUERY:
{query}

OFFICIAL DOCUMENTATION CONTEXT:
{rag_context}

INSTRUCTIONS:
1. Analyze the student's query carefully
2. Use only the information from the official documentation provided
3. Structure your response professionally and concisely  
4. Be specific and accurate - avoid generic statements
5. If the documentation doesn't contain relevant information, acknowledge this politely
6. Keep the response focused and under 300 words
7. Use a helpful, professional tone appropriate for academic institution communication

RESPONSE FORMAT RULES:
- Start directly with the relevant information
- DO NOT use asterisks (*), markdown formatting, or special characters
- Use plain text formatting only - no bold, italics, or bullet symbols
- For lists, use simple numbered format (1., 2., 3.) or dash format (- item)
- Include specific details like dates, numbers, procedures when available
- End with an offer for further assistance if needed
- Make the response clean and readable in plain email format

Generate a professional response using only plain text formatting:"""

            # Get intelligent response from Gemini
            response = self.gemini_model.generate_content(prompt)
            
            if response and response.text and len(response.text.strip()) > 50:
                # Clean up any remaining asterisks or markdown formatting
                clean_response = self._clean_email_formatting(response.text.strip())
                return clean_response
            else:
                # Fallback to template-based formatting if Gemini fails
                return self._format_rag_context(rag_context)
                
        except Exception as e:
            logger.error(f"Error generating intelligent response with Gemini: {e}")
            # Fallback to template-based formatting
            return self._format_rag_context(rag_context)
    
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
    
    def _organize_content(self, content: str, sources: list = None) -> str:
        """Organize content into professional sections"""
        if not content or len(content.strip()) < 10:
            return [content] if content else [""]
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        organized_lines = []
        
        current_section = []
        for line in lines:
            # Check if this is a section header (contains keywords like "Fee", "Payment", etc.)
            if any(keyword in line.lower() for keyword in ['fee', 'payment', 'tuition', 'deadline', 'exam', 'admission']):
                if current_section:
                    organized_lines.extend(current_section)
                    current_section = []
                organized_lines.append(f"\n**{line.strip()}:**")
            else:
                if line.startswith('-') or line.startswith('•'):
                    current_section.append(f"  {line}")
                elif len(line) > 30:  # Longer lines as paragraphs
                    current_section.append(f"\n{line}")
                else:  # Short lines as bullet points
                    current_section.append(f"  • {line}")
        
        # Add remaining content
        if current_section:
            organized_lines.extend(current_section)
        
        # Return as sections for consistent formatting
        if organized_lines:
            return ['\n'.join(organized_lines).strip()]
        else:
            return [content.strip()] if content else [""]
    
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
    
    def _clean_email_formatting(self, text: str) -> str:
        """Clean email formatting to remove asterisks and markdown formatting"""
        if not text:
            return text
            
        # Remove asterisks used for bold formatting
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold** -> bold
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)   # *italic* -> italic
        
        # Remove other markdown formatting
        cleaned = re.sub(r'#{1,6}\s*(.*?)(?:\n|$)', r'\1\n', cleaned)  # Headers
        cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)  # Inline code
        cleaned = re.sub(r'```.*?\n(.*?)\n```', r'\1', cleaned, flags=re.DOTALL)  # Code blocks
        
        # Clean up bullet points - replace with simple dash format
        cleaned = re.sub(r'^\s*[\*\+\-•]\s+', '- ', cleaned, flags=re.MULTILINE)
        
        # Clean up excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Multiple newlines -> double newline
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned)  # Trim whitespace
        
        return cleaned

    def _format_professional_email(self, response: str, context_info: dict = None, original_query: str = None) -> str:
        """Format the response into a professional email structure"""
        
        # Clean the response first
        response = self._clean_response(response)
        
        # Use original query if available, otherwise use response as query
        query_for_llm = original_query if original_query else response
        
        # Professional email template
        email_parts = []
        
        # Greeting
        email_parts.append("Dear Student,")
        email_parts.append("")
        
        # Context-aware greeting based on available information
        if context_info and context_info.get('sources'):
            # Analyze sources to provide specific greeting
            sources_text = ' '.join(context_info.get('sources', []))
            if 'fee' in sources_text.lower() or 'payment' in sources_text.lower():
                email_parts.append("Thank you for your inquiry regarding fee information.")
            elif 'exam' in sources_text.lower() or 'academic' in sources_text.lower():
                email_parts.append("Thank you for your inquiry regarding academic information.")
            elif 'computer science' in sources_text.lower() or 'course' in sources_text.lower():
                email_parts.append("Thank you for your inquiry regarding our course programs.")
            else:
                email_parts.append("Thank you for your inquiry.")
        else:
            email_parts.append("Thank you for your inquiry.")
        email_parts.append("")
        
        # Main content - Use intelligent LLM generation with RAG context
        # Check if we have RAG sources available (context_result uses different structure than rag_context)
        has_sources = context_info and context_info.get('sources') and len(context_info.get('sources', [])) > 0
        has_context = context_info and (context_info.get('context_used', True) or context_info.get('context', ''))
        
        if has_sources:
            # Use RAG context as primary content - get actual content from context field
            rag_context = context_info.get('context', '')
            
            if rag_context and len(rag_context.strip()) > 50:
                # Generate intelligent response using Gemini LLM with RAG context
                # This implements the proper pipeline: Query → RAG → LLM → Structured Response
                intelligent_response = self._generate_intelligent_response_from_context(
                    query_for_llm,  # Use the actual query for LLM processing
                    rag_context, 
                    context_info
                )
                
                email_parts.append(intelligent_response)
                email_parts.append("")
            else:
                # Fallback when context is empty but sources exist
                email_parts.append("We have received your inquiry and will provide you with the relevant information.")
                sources_list = context_info.get('sources', [])
                if sources_list:
                    email_parts.append(f"Relevant documents found: {', '.join(sources_list[:3])}")
                email_parts.append("")
        else:
            # Add response as structured content when no RAG context available
            if response.strip() and len(response.strip()) > 10:
                # Try to generate intelligent response even without RAG context
                if self.gemini_enabled:
                    try:
                        simple_prompt = f"""You are a professional AI assistant for an educational institution. 
                        Respond professionally to this student query: {query_for_llm}
                        
                        IMPORTANT: Use only plain text formatting - no asterisks, markdown, or special characters.
                        Keep the response concise, helpful, and professional. If you don't have specific information, 
                        acknowledge this and suggest contacting the support team."""
                        
                        gemini_response = self.gemini_model.generate_content(simple_prompt)
                        if gemini_response and gemini_response.text and len(gemini_response.text.strip()) > 30:
                            # Clean any formatting from the response
                            clean_response = self._clean_email_formatting(gemini_response.text.strip())
                            email_parts.append(clean_response)
                        else:
                            # Fallback to basic formatting without asterisks
                            email_parts.append("Information Requested:")
                            email_parts.append("")
                            formatted_content = self._format_content_sections(response)
                            email_parts.append(formatted_content)
                    except Exception as e:
                        logger.error(f"Error generating fallback response: {e}")
                        email_parts.append("Information Requested:")
                        email_parts.append("")
                        formatted_content = self._format_content_sections(response)
                        email_parts.append(formatted_content)
                else:
                    # No Gemini available - use basic formatting without asterisks
                    email_parts.append("Information Requested:")
                    email_parts.append("")
                    formatted_content = self._format_content_sections(response)
                    email_parts.append(formatted_content)
                email_parts.append("")
            else:
                # Fallback when no good response or context
                email_parts.append("We have received your inquiry and will provide you with the relevant information.")
                email_parts.append("")
            if response.strip() and len(response.strip()) > 10:
                email_parts.append("Information Requested:")
                email_parts.append("")
                # Format response with bullet points if it contains multiple pieces of info
                formatted_content = self._format_content_sections(response)
                email_parts.append(formatted_content)
                email_parts.append("")
            else:
                # Fallback when no good response or context
                email_parts.append("We have received your inquiry and will provide you with the relevant information.")
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
    
    def _format_rag_context(self, rag_context: str) -> list:
        """Format RAG context content into structured email sections"""
        email_sections = []
        
        # The actual format is: "From filename: content continues on same line..."
        # Split by "From " to get separate document sections
        sections = rag_context.split('From ')
        sections_processed = 0
        max_sections = 2  # Limit to 2 sections to keep emails concise
        
        for section in sections[1:]:  # Skip first empty section
            if sections_processed >= max_sections:
                break
                
            if not section.strip() or len(section.strip()) < 50:
                continue
            
            # Find the first colon to separate filename from content
            colon_idx = section.find(':')
            if colon_idx == -1:
                continue
                
            filename = section[:colon_idx].strip()
            content = section[colon_idx+1:].strip()
            
            if not content or len(content) < 30:
                continue
            
            # Add section header based on document type - make it more specific
            if 'fee' in filename.lower():
                email_sections.append("Fee Information:")
            elif 'course' in filename.lower() or 'study' in filename.lower():
                email_sections.append("Course Information:")
            elif 'academic' in filename.lower() or 'calendar' in filename.lower():
                email_sections.append("Academic Calendar:")
            elif 'handbook' in filename.lower():
                email_sections.append("Student Guidelines:")
            elif 'brochure' in filename.lower():
                email_sections.append("Institute Information:")
            else:
                email_sections.append(f"{filename.replace('.pdf', '')} Information:")
            
            # Format content into readable chunks
            formatted_content = self._format_document_content(content)
            email_sections.append(formatted_content)
            email_sections.append("")  # Add spacing
            sections_processed += 1
        
        # Add helpful footer if we have content
        if email_sections:
            email_sections.append("---")
            email_sections.append("Need more specific information?** Please reply with your specific questions!")
        
        return email_sections
    
    def _format_document_content(self, content: str, query_keywords: list = None) -> str:
        """Format document content for email display - provide complete relevant information"""
        if not content:
            return "Information is available - please contact our support team for specific details."
        
        # Clean and structure the content
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # If we have query keywords, try to find relevant lines first
        relevant_lines = []
        other_lines = []
        
        if query_keywords:
            for line in lines:
                line_lower = line.lower()
                if any(keyword.lower() in line_lower for keyword in query_keywords):
                    relevant_lines.append(line)
                else:
                    other_lines.append(line)
        else:
            relevant_lines = lines
        
        formatted_lines = []
        
        # Add relevant lines first (complete, no truncation)
        for line in relevant_lines[:5]:  # Max 5 relevant lines
            if line and len(line) > 10:
                # Clean up the line but don't truncate
                clean_line = line.replace(' - ', ' — ').strip()
                if not clean_line.startswith(('•', '-', '*')):
                    formatted_lines.append(f"• {clean_line}")
                else:
                    formatted_lines.append(clean_line)
        
        # Add other important lines if we have space
        if len(formatted_lines) < 3:
            for line in other_lines[:2]:
                if line and len(line) > 10:
                    clean_line = line.replace(' - ', ' — ').strip()
                    if not clean_line.startswith(('•', '-', '*')):
                        formatted_lines.append(f"• {clean_line}")
                    else:
                        formatted_lines.append(clean_line)
        
        if formatted_lines:
            return '\n'.join(formatted_lines)
        else:
            return "Detailed information is available in our official documentation."
    
    def _extract_fee_info(self, content: str) -> str:
        """Extract and format fee information"""
        lines = content.split('\n')
        fee_info = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['fee', 'cost', 'payment', 'tuition', '$', 'rs.', 'rupee']):
                if line and not line.startswith('From '):
                    # Clean up formatting
                    line = line.replace(' - ', '\n• ')
                    if not line.startswith('•'):
                        line = f"• {line}"
                    fee_info.append(line)
        
        if fee_info:
            return '\n'.join(fee_info[:8])  # Limit to prevent overwhelming
        return "Fee information is available through the student portal or accounts office."
    
    def _extract_academic_info(self, content: str) -> str:
        """Extract and format academic calendar information"""
        lines = content.split('\n')
        academic_info = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['semester', 'exam', 'classes', 'results', 'calendar']):
                if line and not line.startswith('From '):
                    # Clean up formatting
                    line = line.replace(' - ', '\n• ')
                    if not line.startswith('•'):
                        line = f"• {line}"
                    academic_info.append(line)
        
        if academic_info:
            return '\n'.join(academic_info[:8])  # Limit to prevent overwhelming
        return "Academic calendar information is available on the student portal."
    
    def _extract_course_info(self, content: str) -> str:
        """Extract and format course information"""
        lines = content.split('\n')
        course_info = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['course', 'program', 'computer', 'science', 'curriculum']):
                if line and not line.startswith('From '):
                    # Clean up formatting
                    if not line.startswith('•'):
                        line = f"• {line}"
                    course_info.append(line)
        
        if course_info:
            return '\n'.join(course_info[:6])  # Limit to prevent overwhelming
        return "Course information is available through the academic department or student portal."
    
    def _clean_source_content(self, content: str) -> str:
        """Clean and format source content for email display"""
        # Remove file references and clean up
        content = re.sub(r'From [^:]+:\s*', '', content)
        
        # Split into manageable chunks
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Format as bullet points
        formatted_lines = []
        for line in lines[:6]:  # Limit lines
            if line and len(line) > 10:
                if not line.startswith('•'):
                    formatted_lines.append(f"• {line}")
                else:
                    formatted_lines.append(line)
    
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
            structured_parts.append("Fee Information:")
            for info in fee_info[:3]:  # Limit to prevent overwhelming
                structured_parts.append(f"• {info}")
            structured_parts.append("")
        
        if date_info:
            structured_parts.append("Important Dates:")
            for info in date_info[:3]:
                structured_parts.append(f"• {info}")
            structured_parts.append("")
        
        if general_info:
            structured_parts.append("Additional Information:")
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