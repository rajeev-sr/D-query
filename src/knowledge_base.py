# src/knowledge_base.py
import os
import json
import pandas as pd
from pathlib import Path
import PyPDF2
import docx
from typing import List, Dict, Any
import re

class DocumentProcessor:
    def __init__(self, docs_directory="data/institute_docs"):
        self.docs_dir = Path(docs_directory)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample documents if none exist
        self._create_sample_documents()
    
    def _create_sample_documents(self):
        """Create sample institute documents for testing"""
        
        sample_docs = {
            "academic_calendar.txt": """
            ACADEMIC CALENDAR 2025-26

            SEMESTER 1:
            - Classes Start: August 15, 2025
            - Mid-term Exams: October 1-5, 2025
            - End-term Exams: December 10-20, 2025
            - Results Declaration: January 5, 2025

            SEMESTER 2:
            - Classes Start: January 15, 2025
            - Mid-term Exams: March 15-20, 2025
            - End-term Exams: May 15-25, 2025
            - Results Declaration: June 10, 2025

            IMPORTANT DATES:
            - Registration Deadline: 1 week before semester start
            - Fee Payment Deadline: 2 weeks after registration
            - Assignment Submission: 1 week before end-term exams
            """,
                        
                        "fee_structure.txt": """
            FEE STRUCTURE 2025-26

            UNDERGRADUATE PROGRAMS:
            - Tuition Fee: $5,000 per semester
            - Library Fee: $200 per semester
            - Laboratory Fee: $300 per semester
            - Sports Fee: $100 per semester
            - Total: $5,600 per semester

            PAYMENT METHODS:
            - Online Payment: Available 24/7 through student portal
            - Bank Transfer: Account details available in student handbook
            - Installments: Available in 2 installments per semester

            LATE PAYMENT PENALTY:
            - 1-7 days late: $50 penalty
            - 8-15 days late: $100 penalty
            - More than 15 days: Academic hold on account
            """,
                        
                        "technical_support.txt": """
            TECHNICAL SUPPORT GUIDE

            STUDENT PORTAL ACCESS:
            - URL: https://portal.university.edu
            - Username: Your student ID
            - Password: Default is DOB (DDMMYYYY), change on first login

            COMMON ISSUES:
            1. Forgot Password:
            - Click "Forgot Password" on login page
            - Enter student ID and registered email
            - Check email for reset link

            2. Cannot Access Courses:
            - Ensure you are registered for the semester
            - Contact academic office if registration issues persist

            3. WiFi Connection:
            - Network: UniversityWiFi
            - Password: Available at IT helpdesk
            - Contact IT for connection issues

            SUPPORT CONTACTS:
            - IT Helpdesk: it-help@university.edu
            - Phone: (555) 123-4567
            - Hours: Monday-Friday 9 AM - 6 PM
            """,
                        
                        "assignment_guidelines.txt": """
            ASSIGNMENT SUBMISSION GUIDELINES

            SUBMISSION PROCESS:
            1. Login to student portal
            2. Navigate to course page
            3. Click "Submit Assignment"
            4. Upload file (PDF format preferred)
            5. Click "Submit" and wait for confirmation

            FILE REQUIREMENTS:
            - Format: PDF, DOC, DOCX
            - Size limit: 10 MB per file
            - Naming: StudentID_CourseName_AssignmentNumber.pdf

            DEADLINE POLICY:
            - Submissions accepted until 11:59 PM on due date
            - Late submissions: 10% penalty per day
            - No submissions accepted after 1 week

            PLAGIARISM POLICY:
            - All submissions checked for plagiarism
            - First offense: Warning and resubmission
            - Second offense: Zero grade for assignment
            - Third offense: Course failure
            """
                    }
        
        # Create sample documents
        for filename, content in sample_docs.items():
            filepath = self.docs_dir / filename
            if not filepath.exists():
                with open(filepath, 'w') as f:
                    f.write(content)
        
        print(f"Sample documents created in {self.docs_dir}")
    
    def process_all_documents(self) -> List[Dict]:
        """Process all documents in the directory"""
        documents = []
        
        for file_path in self.docs_dir.glob('*'):
            if file_path.is_file():
                try:
                    doc = self.process_document(file_path)
                    if doc:
                        documents.append(doc)
                        print(f"Processed: {file_path.name}")
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
        
        print(f"Total documents processed: {len(documents)}")
        return documents
    
    def process_document(self, file_path: Path) -> Dict:
        """Process individual document"""
        
        content = ""
        doc_type = file_path.suffix.lower()
        
        try:
            if doc_type == '.txt':
                content = self._process_txt(file_path)
            elif doc_type == '.pdf':
                content = self._process_pdf(file_path)
            elif doc_type in ['.doc', '.docx']:
                content = self._process_docx(file_path)
            else:
                print(f"Unsupported file type: {doc_type}")
                return None
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
        
        if not content.strip():
            return None
        
        # Split into chunks
        chunks = self._split_into_chunks(content)
        
        return {
            'filename': file_path.name,
            'content': content,
            'chunks': chunks,
            'doc_type': doc_type,
            'metadata': {
                'size': len(content),
                'num_chunks': len(chunks)
            }
        }
    
    def _process_txt(self, file_path: Path) -> str:
        """Process text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _process_pdf(self, file_path: Path) -> str:
        """Process PDF file"""
        content = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                content += page.extract_text() + "\n"
        return content
    
    def _process_docx(self, file_path: Path) -> str:
        """Process Word document"""
        doc = docx.Document(file_path)
        content = ""
        for paragraph in doc.paragraphs:
            content += paragraph.text + "\n"
        return content
    
    def _split_into_chunks(self, content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split content into overlapping chunks"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks