import re
from typing import List, Dict

class EmailProcessor:
    def __init__(self, student_domains=None):
        # Common student email patterns
        self.student_domains = student_domains or [
            '@iitbhilai.ac.in',
        ]

        self.student_patterns = [
            r'^[a-zA-Z0-9._%+-]+@iitbhilai\.ac\.in$',  # Any email at iitbhilai.ac.in
        ]
    
    def is_student_email(self, sender: dict) -> bool:
        """Check if email is from a student. Accepts sender or full email dict."""
        # Accept either sender string or email dict
        if isinstance(sender, dict):
            sender_str = sender.get('sender', '').lower()
            body_str = sender.get('body', '').lower()
        else:
            sender_str = str(sender).lower()
            body_str = ''

        # Check domain patterns in sender
        for domain in self.student_domains:
            if domain.lower() in sender_str:
                return True

        # Check regex patterns in sender
        for pattern in self.student_patterns:
            if re.search(pattern, sender_str):
                return True

        # Check student ID pattern in body (8-digit number)
        if re.search(r'\b\d{8}\b', body_str):
            return True
        
        #check B.tech , M.tech in body
        if re.search(r'\b(b\.tech|m\.tech|btech|mtech)\b', body_str):
            return True
        
        #check "student" keyword in body
        if re.search(r'\bstudent\b', body_str):
            return True

        return False
    
    def filter_student_emails(self, emails: List[Dict]) -> List[Dict]:
        """Filter emails to only include student emails"""
        student_emails = []
        
        for email in emails:
            if self.is_student_email(email):
                student_emails.append(email)
        
        return student_emails
    
    def categorize_emails(self, emails: List[Dict]) -> Dict:
        """Basic categorization of emails"""
        categories = {
            'academic': [],
            'administrative': [],
            'technical': [],
            'other': []
        }
        
        academic_keywords = ['exam', 'assignment', 'grade', 'homework', 'quiz', 'test']
        admin_keywords = [
            'registration', 'enrollment', 'fee', 'certificate', 'transcript',
            'admission', 'scholarship', 'hostel', 'leave', 'attendance',
            'payment', 'refund', 'disciplinary', 'convocation', 'identity card',
            'bonafide', 'migration', 'withdrawal', 'rules', 'notice', 'office',
            'document', 'approval', 'clearance', 'committee', 'council', 'quota',
            'mess', 'canteen', 'library', 'holiday', 'schedule', 'deadline',
            'reminder', 'update', 'announcement', 'policy', 'guidelines', 'form',
            'submission', 'verification', 'process', 'status', 'result', 'marksheet',
            'hostel allotment', 'room change', 'discipline', 'security', 'medical',
            'insurance', 'transport', 'bus', 'parking', 'gatepass', 'visitor', 'parent'
        ]
        tech_keywords = ['login', 'password', 'access', 'website', 'portal', 'system']
        
        for email in emails:
            subject_body = (email['subject'] + ' ' + email['body']).lower()
            
            if any(keyword in subject_body for keyword in admin_keywords):
                categories['administrative'].append(email)
            elif any(keyword in subject_body for keyword in academic_keywords):
                categories['academic'].append(email)
            elif any(keyword in subject_body for keyword in tech_keywords):
                categories['technical'].append(email)
            else:
                categories['other'].append(email)
        
        return categories