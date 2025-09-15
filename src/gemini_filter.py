# src/gemini_filter.py
import google.generativeai as genai
import os
from typing import Dict, List, Tuple
import logging
from datetime import datetime

class GeminiQueryFilter:
    def __init__(self, api_key=None):
        """Initialize Gemini API for email query filtering"""
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        try:
            # Use the updated Gemini model name
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logging.info("Gemini model initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model: {e}")
            raise
        
        # Query filtering prompt template
        self.filter_prompt = """
You are an intelligent email filter for an educational institution's query handling system.

Your task is to analyze an email and determine if it contains a GENUINE QUERY or REQUEST that needs human/AI assistance.

EMAIL TO ANALYZE:
From: {sender}
Subject: {subject}
Body: {body}

CLASSIFICATION CRITERIA:

✅ ACCEPT as QUERY if the email contains:
- Questions about academic procedures (admissions, exams, grades, courses)
- Requests for information (schedules, deadlines, requirements)
- Administrative help needed (fees, documents, certificates)
- Technical support requests (login issues, portal problems)
- Clarification requests about policies or procedures
- Complaints that need resolution
- Requests for appointments or meetings
- Help with applications or submissions

❌ REJECT (not a query) if the email is:
- Spam or promotional content
- Automated notifications or alerts
- Simple acknowledgments or "thank you" messages
- Personal conversations not related to institution services
- Mass forwards or chain messages
- News updates or newsletters
- Social invitations or casual messages
- Advertisements or marketing content

RESPONSE FORMAT:
Return ONLY one of these responses:
- "ACCEPT" if it's a genuine query/request
- "REJECT" if it's not a query

Your response:
"""
    
    def is_query_related(self, email: Dict) -> Tuple[bool, str]:
        """
        Analyze email content using Gemini to determine if it's query-related
        
        Args:
            email: Dict containing 'sender', 'subject', 'body' keys
            
        Returns:
            Tuple of (is_query: bool, reason: str)
        """
        try:
            # Prepare email content
            sender = email.get('sender', 'Unknown')
            subject = email.get('subject', 'No Subject')
            body = email.get('body', 'No Content')
            
            # Create prompt with email content
            prompt = self.filter_prompt.format(
                sender=sender,
                subject=subject,
                body=body[:1000]  # Limit body to first 1000 chars
            )
            
            # Get Gemini analysis
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                logging.warning("Empty response from Gemini")
                return True, "Empty response - defaulting to accept"
            
            result = response.text.strip().upper()
            
            if "ACCEPT" in result:
                return True, "Classified as query by Gemini"
            elif "REJECT" in result:
                return False, "Classified as non-query by Gemini"
            else:
                logging.warning(f"Unexpected Gemini response: {response.text}")
                return True, "Unclear response - defaulting to accept"
                
        except Exception as e:
            logging.error(f"Error in Gemini query filtering: {e}")
            # Default to accepting emails if there's an error
            return True, f"Error in filtering - defaulting to accept: {str(e)}"
    
    def filter_query_emails(self, emails: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter emails to separate query-related from non-query emails
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            Tuple of (query_emails, non_query_emails)
        """
        query_emails = []
        non_query_emails = []
        
        for i, email in enumerate(emails):
            try:
                logging.info(f"Filtering email {i+1}/{len(emails)}: {email.get('subject', 'No Subject')[:50]}...")
                
                is_query, reason = self.is_query_related(email)
                
                # Add filtering metadata to email
                email['gemini_filtered'] = True
                email['gemini_filter_result'] = 'query' if is_query else 'non_query'
                email['gemini_filter_reason'] = reason
                email['gemini_filtered_at'] = datetime.now().isoformat()
                
                if is_query:
                    query_emails.append(email)
                    logging.info(f"✅ Accepted: {reason}")
                else:
                    non_query_emails.append(email)
                    logging.info(f"❌ Rejected: {reason}")
                    
            except Exception as e:
                logging.error(f"Error filtering email {i+1}: {e}")
                # Default to accepting on error
                email['gemini_filtered'] = True
                email['gemini_filter_result'] = 'query'
                email['gemini_filter_reason'] = f"Error in filtering: {str(e)}"
                email['gemini_filtered_at'] = datetime.now().isoformat()
                query_emails.append(email)
        
        logging.debug(f"Gemini filtering complete: {len(query_emails)} queries, {len(non_query_emails)} non-queries")
        return query_emails, non_query_emails
    
    def get_filter_stats(self, emails: List[Dict]) -> Dict:
        """Get statistics about filtered emails"""
        total = len(emails)
        filtered = len([e for e in emails if e.get('gemini_filtered')])
        queries = len([e for e in emails if e.get('gemini_filter_result') == 'query'])
        non_queries = len([e for e in emails if e.get('gemini_filter_result') == 'non_query'])
        
        return {
            'total_emails': total,
            'filtered_emails': filtered,
            'query_emails': queries,
            'non_query_emails': non_queries,
            'query_rate': (queries / total * 100) if total > 0 else 0,
            'filter_rate': (filtered / total * 100) if total > 0 else 0
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the filter with sample emails
    logging.basicConfig(level=logging.INFO)
    
    sample_emails = [
        {
            'sender': 'student@iitbhilai.ac.in',
            'subject': 'Question about exam schedule',
            'body': 'Hi, I wanted to ask about the final exam schedule for this semester. When will it be published?'
        },
        {
            'sender': 'marketing@company.com',
            'subject': 'Special Offer - 50% Off!',
            'body': 'Limited time offer! Get 50% off on our premium products. Click here to buy now!'
        },
        {
            'sender': 'student@iitbhilai.ac.in', 
            'subject': 'Thank you',
            'body': 'Thank you for your help yesterday. Have a great day!'
        }
    ]
    
    try:
        # Initialize filter (requires GEMINI_API_KEY environment variable)
        filter_system = GeminiQueryFilter()
        
        # Filter emails
        queries, non_queries = filter_system.filter_query_emails(sample_emails)
        
        print(f"\nFiltering Results:")
        print(f"Query emails: {len(queries)}")
        print(f"Non-query emails: {len(non_queries)}")
        
        # Show results
        print("\n=== QUERY EMAILS ===")
        for email in queries:
            print(f"✅ {email['subject']}: {email['gemini_filter_reason']}")
            
        print("\n=== NON-QUERY EMAILS ===") 
        for email in non_queries:
            print(f"❌ {email['subject']}: {email['gemini_filter_reason']}")
            
    except Exception as e:
        print(f"Error testing filter: {e}")
        print("Make sure to set GEMINI_API_KEY environment variable")