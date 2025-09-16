#src/model_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import json

class QueryClassifier:
    def __init__(self, model_path="models/fine_tuned"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = self._setup_cuda_safely()
        
        self.load_model()

    def _setup_cuda_safely(self):  # Add this method
        """Setup CUDA with proper error handling"""
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return torch.device("cpu")
        
        try:
            # Test CUDA functionality
            torch.cuda.empty_cache()
            test_tensor = torch.tensor([1.0]).cuda()
            _ = test_tensor + 1
            test_tensor.cpu()
            del test_tensor
            torch.cuda.empty_cache()
            
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA test failed: {e}, falling back to CPU")
            return torch.device("cpu")
        

    def _validate_model_outputs(self):
        """Validate model doesn't produce NaN or inf values"""
        try:
            test_input = self.tokenizer(
                "Test input",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50
            )
            
            if self.device.type == "cuda":
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
            
            with torch.no_grad():
                outputs = self.model(**test_input)
                logits = outputs.logits
                
                # Check for NaN or inf values
                if torch.isnan(logits).any():
                    print("Model produces NaN values")
                    return False
                if torch.isinf(logits).any():
                    print("Model produces inf values")  
                    return False
                    
                print("Model validation passed")
                return True
                
        except Exception as e:
            print(f"Model validation failed: {e}")
            return False
    
    def load_model(self):
        """Load the fine-tuned model"""
        try:
            print(f"Loading model from {self.model_path}")
            
            # Load tokenizer with proper configuration
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Fix tokenizer configuration
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model with CUDA-safe settings
            model_kwargs = {
                "dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if self.device.type == "cuda":
                model_kwargs["device_map"] = {"": 0}  # Force to GPU 0
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Move to device if needed
            if self.device.type == "cpu":
                base_model = base_model.to(self.device)
            
            self.model = base_model
            self.model.eval()
            if not self._validate_model_outputs():
                print("Model validation failed, loading base model instead...")
                return self._load_base_model_fallback()
                    
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
            
    def _load_base_model_fallback(self):
        """Load base model when fine-tuned model is corrupted"""
        try:
            print("Loading base model as fallback...")
            
            # Try different base models
            base_models = ["microsoft/DialoGPT-medium", "gpt2", "distilgpt2"]
            
            for model_name in base_models:
                try:
                    print(f"Trying {model_name}...")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    
                    if self.device.type == "cuda":
                        self.model = self.model.to(self.device)
                    
                    self.model.eval()
                    
                    # Test the model
                    if self._validate_model_outputs():
                        print(f"Successfully loaded {model_name}")
                        return True
                    else:
                        print(f"{model_name} validation failed")
                        
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"Base model fallback failed: {e}")
            return False
    def classify_and_respond(self, query, max_length=512):
        """Classify query and generate response"""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        # Clean input
        query = query.strip()
        if not query:
            return {"error": "Empty query"}
        
        # Use simpler, more effective prompt format based on testing
        prompt = f"Question: {query}\nAnswer:"
        
        try:
            # Tokenize with safe parameters
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=128,  # Much shorter
                padding=True
            )
            
            # Move to device safely
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with conservative settings but enough tokens for proper output
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=100,  # Enough for "Category: X Response: Y"
                    num_return_sequences=1,
                    do_sample=False,  # Greedy only
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0
                )
            
            # Decode and clean response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = full_response.replace(prompt, "").strip()
            
            # Clean up common artifacts from fine-tuned models
            generated_part = generated_part.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
            generated_part = generated_part.replace('You are an institute assistant', '')
            generated_part = generated_part.replace('with access to official documentation', '')
            generated_part = generated_part.replace('Use the following official information', '')
            generated_part = ' '.join(generated_part.split())  # Normalize whitespace
            generated_part = generated_part.strip()
            
            # Check if response is garbage
            if self._is_garbage_response(generated_part):
                print(f"Detected garbage response: '{generated_part[:100]}...', using fallback")
                generated_part = self._get_rule_based_response(query)
            else:
                print(f"Model generated: '{generated_part[:100]}...'")
            
            # Parse result
            result = self._parse_model_output(generated_part, query)
            result['confidence'] = self._calculate_confidence(generated_part)
            
            return result
            
        except Exception as e:
            print(f"Generation error: {e}")
            return {
                "category": "other",
                "response": self._get_rule_based_response(query),
                "confidence": 0.6,
                "action": "review_needed",
                "fallback": True
            }
    
    def _parse_model_output(self, output, query=""):
        """Parse model output to extract category and response"""
        # Default values
        result = {
            "category": "other",
            "response": output.strip(),
            "action": "review_needed"
        }
        
        # Try to extract structured format first
        category_match = re.search(r'Category:\s*(\w+)', output, re.IGNORECASE)
        if category_match:
            result["category"] = category_match.group(1).lower()
        
        response_match = re.search(r'Response:\s*(.*?)(?=\n\n|\Z)', output, re.DOTALL | re.IGNORECASE)
        if response_match:
            raw_response = response_match.group(1).strip()
            result["response"] = self._clean_and_format_response(raw_response)
        else:
            # If no structured format, use entire output as response and classify by content
            raw_response = output.strip()
            result["response"] = self._clean_and_format_response(raw_response)
            # Pass both query and response for better classification
            result["category"] = self._classify_by_content(raw_response, query)
        
        # Determine action based on category and content
        result["action"] = self._determine_action(result["category"], result["response"])
        
        return result
    
    def _clean_and_format_response(self, response: str) -> str:
        """Clean and format response as professional email"""
        import re
        
        # Clean unwanted patterns
        cleaned = response
        
        # Remove special tokens and artifacts
        cleaned = re.sub(r'<\|.*?\|>', '', cleaned)
        cleaned = re.sub(r'<.*?>', '', cleaned)
        cleaned = re.sub(r'(?i)(start|end)oftext', '', cleaned)
        cleaned = re.sub(r'You are an? institute assistant.*?accurately\.\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Remove duplicate whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        # If the response is too short or seems incomplete, use rule-based format
        if len(cleaned) < 20 or not self._is_complete_response(cleaned):
            return cleaned
        
        # Don't double-format if it's already formatted as an email
        if cleaned.startswith('Dear ') or 'Best regards' in cleaned:
            return cleaned
        
        # Format as professional email
        return self._format_as_professional_email(cleaned)
    
    def _is_complete_response(self, response: str) -> bool:
        """Check if response is complete and meaningful"""
        if not response or len(response) < 10:
            return False
        
        # Check for incomplete patterns
        incomplete_patterns = [
            r'\.\.\.$',  # ends with ...
            r'[a-z]$',   # ends mid-sentence without punctuation
            r'Category:\s*$',  # just "Category:"
            r'Response:\s*$'   # just "Response:"
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, response):
                return False
        
        return True
    
    def _classify_by_content(self, response, query=""):
        """Classify response based on content keywords from both query and response"""
        # Combine query and response for classification
        combined_text = f"{query} {response}".lower()
        
        # Academic and exam related
        if any(word in combined_text for word in ['exam', 'test', 'grade', 'result', 'academic', 'calendar', 'schedule']):
            return 'academic'
        
        # Fee and payment related
        if any(word in combined_text for word in ['fee', 'payment', 'pay', 'cost', 'charge', 'account', 'tuition', 'bill']):
            return 'fee'
        
        # Registration and enrollment
        if any(word in combined_text for word in ['register', 'enrollment', 'admission', 'apply', 'enroll']):
            return 'registration'
        
        # Technical support
        if any(word in combined_text for word in ['password', 'login', 'technical', 'support', 'help', 'system', 'portal']):
            return 'technical'
        
        # Administrative
        if any(word in combined_text for word in ['office', 'document', 'certificate', 'verification', 'transcript']):
            return 'administrative'
        
        return 'other'
    
    def _determine_action(self, category, response):
        """Determine appropriate action based on classification"""

        categories = [
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
        ]
        
        # Simple rule-based action determination
        if category in categories and len(response) > 20:
            return "auto_respond"
        elif category in ['technical', 'other']:
            return "review_needed"
        else:
            return "escalate"
    
    def _calculate_confidence(self, output):
        """Calculate confidence score based on output quality"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if category is clearly identified
        if re.search(r'Category:\s*\w+', output, re.IGNORECASE):
            confidence += 0.2
        
        # Higher confidence for longer, more detailed responses
        if len(output.split()) > 15:
            confidence += 0.1
        elif len(output.split()) > 8:
            confidence += 0.05
        
        # Higher confidence if it contains helpful patterns
        if any(pattern in output.lower() for pattern in [
            'please check', 'you can', 'visit the', 'through the', 'contact the'
        ]):
            confidence += 0.15
        
        # Boost confidence for rule-based responses on matching queries
        if "student portal" in output or "office" in output or "registration" in output:
            confidence += 0.2
        
        return min(1.0, confidence)
        
        # Higher confidence if response is structured
        if re.search(r'Response:\s*.+', output, re.IGNORECASE):
            confidence += 0.2
        
        # Lower confidence if response is very short or generic
        if len(output.strip()) < 30:
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    

    def clear_gpu_cache(self):
        """Clear GPU cache manually"""
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("GPU cache cleared")
            except Exception as e:
                print(f"Could not clear GPU cache: {e}")

    def get_gpu_memory_info(self):
        """Get GPU memory information"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            return {"allocated": allocated, "reserved": reserved}
        return None
    
    def _generate_cpu_fallback(self, query):
        """Generate response on CPU"""
        try:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            return self.classify_and_respond(query, max_length=256)
        except Exception as e:
            return self._get_error_fallback(query)

    def _get_fallback_response(self, query):
        """Rule-based fallback response"""
        query_lower = query.lower()
        if 'password' in query_lower or 'login' in query_lower:
            return "Click 'Forgot Password' on the login page to reset your password."
        elif 'fee' in query_lower or 'payment' in query_lower:
            return "You can pay fees through the student portal or contact the accounts office."
        elif 'exam' in query_lower or 'test' in query_lower:
            return "Please check the academic calendar for exam schedules."
        elif 'assignment' in query_lower or 'submission' in query_lower:
            return "Submit assignments through the course portal before the deadline."
        else:
            return "Thank you for your question. Please contact the support team for assistance."

    def _get_error_fallback(self, query):
        """Error fallback response"""
        return {
            "category": "other",
            "response": self._get_fallback_response(query),
            "confidence": 0.5,
            "action": "review_needed",
            "fallback": True
        }
    
    def _is_garbage_response(self, response):
        """Check if response contains garbage characters - updated for fine-tuned model"""
        if not response:
            return True
        
        # Clean the response of known special tokens first
        cleaned_response = response.replace('<|startoftext|>', '').replace('<|endoftext|>', '').strip()
        
        if not cleaned_response:
            return True
        
        # If response contains expected structured output, be much more lenient
        has_category = 'category:' in cleaned_response.lower()
        has_response = 'response:' in cleaned_response.lower()
        if has_category or has_response:
            # For structured responses, only check if it's completely empty or nonsensical
            if len(cleaned_response.split()) >= 2:  # At least 2 words
                return False
        
        # More lenient check for special characters (exclude common punctuation and formatting)
        allowed_special = set(' .,!?-:;()[]{}"\'\n\t')
        special_chars = sum(1 for c in cleaned_response if not c.isalnum() and c not in allowed_special)
        if len(cleaned_response) > 0 and special_chars / len(cleaned_response) > 0.5:  # Increased threshold
            return True
        
        # Check for repeated patterns (more lenient)
        if len(cleaned_response) >= 10 and len(set(cleaned_response[:20])) < 3:  # Only flag if very repetitive
            return True
        
        # Check for minimum meaningful content
        if len(cleaned_response.split()) < 2:  # At least 2 words
            return True
        
        return False

    def _get_rule_based_response(self, query):
        """Generate rule-based response with proper email formatting"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['password', 'login', 'forgot']):
            content = "To reset your password, please click on 'Forgot Password' on the login page and follow the instructions sent to your registered email address."
        elif any(word in query_lower for word in ['exam', 'test', 'schedule']):
            content = "Please check the academic calendar on the student portal for exam schedules and important dates. You can also contact your course coordinator for specific exam information."
        elif any(word in query_lower for word in ['fee', 'payment', 'pay']):
            content = "You can pay your fees through the online student portal using various payment methods. Alternatively, you can visit the accounts office during working hours (9 AM - 5 PM, Monday to Friday)."
        elif any(word in query_lower for word in ['assignment', 'submission', 'submit']):
            content = "Assignments can be submitted through the course portal before the specified deadline. Please check your course syllabus for submission guidelines and deadlines."
        elif any(word in query_lower for word in ['register', 'registration', 'enroll']):
            content = "Course registration is available through the student portal during the designated registration periods. Please ensure you meet all prerequisites before registering."
        else:
            content = "Thank you for your question. For specific assistance regarding your query, please contact the student support office or visit our help desk during office hours."
        
        # Format as professional email
        return self._format_as_professional_email(content)
    
    def _format_as_professional_email(self, content: str) -> str:
        """Format content as a professional email"""
        email_parts = [
            "Dear Student,",
            "",
            f"Thank you for your inquiry. {content}",
            "",
            "If you need further assistance, please don't hesitate to contact us.",
            "",
            "Best regards,",
            "Student Support Team",
            "Institute Administration"
        ]
        
        return '\n'.join(email_parts)