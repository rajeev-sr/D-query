import os
import csv
import time
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    TextLoader, 
    DirectoryLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# Set up your Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
os.environ["GOOGLE_API_KEY"] = api_key

class TrainingDataGenerator:
    def __init__(self, api_key: str):
        """Initialize the training data generator"""
        os.environ["GOOGLE_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Common student names and email patterns
        self.student_names = [
            "RAJEEV KUMAR", "PRIYA SHARMA", "AMIT SINGH", "SNEHA PATEL", 
            "ROHIT GUPTA", "ANITA VERMA", "DEEPAK KUMAR", "KAVYA REDDY",
            "VIKASH YADAV", "POOJA AGARWAL", "SURESH CHANDRA", "MEERA JAIN",
            "RAVI PRAKASH", "SUNITA DEVI", "MANISH KUMAR", "NISHA SINGH"
        ]
        
        self.email_domains = ["@iitbhilai.ac.in"]

        
        
        # Query categories
        self.categories = [
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

    def load_documents(self, document_paths: List[str]) -> List[Document]:
        """Load documents from various file types"""
        all_documents = []
        
        for path in document_paths:
            if os.path.isfile(path):
                # Single file
                documents = self._load_single_file(path)
                all_documents.extend(documents)
            elif os.path.isdir(path):
                # Directory
                documents = self._load_directory(path)
                all_documents.extend(documents)
        
        return all_documents

    def _load_single_file(self, file_path: str) -> List[Document]:
        """Load a single file based on its extension"""
        _, ext = os.path.splitext(file_path.lower())
        
        try:
            if ext == '.pdf':
                loader = PyMuPDFLoader(file_path)
            elif ext in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
            
            else:
                print(f"Unsupported file type: {ext}")
                return []
            
            return loader.load()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def _load_directory(self, dir_path: str) -> List[Document]:
        """Load all supported files from a directory"""
        documents = []
        
        # Load PDFs manually to use our custom PDF loader
        pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(dir_path, pdf_file)
            pdf_docs = self._load_single_file(pdf_path)
            documents.extend(pdf_docs)
        
        # Load text files
        try:
            for ext in ['*.txt', '*.md']:
                txt_loader = DirectoryLoader(
                    dir_path, 
                    glob=ext, 
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': 'utf-8'}
                )
                documents.extend(txt_loader.load())
        except Exception as e:
            print(f"Error loading text files: {e}")
        
        return documents

    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000) -> List[Document]:
        """Split documents into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
        )
        
        return text_splitter.split_documents(documents)

    def generate_synthetic_queries(self, document_chunk: Document, num_queries: int = 1) -> List[Dict]:
        print("Generating synthetic queries")
        """Generate synthetic student queries based on document content"""
        
        prompt_template = ChatPromptTemplate.from_template("""
        Based on the following document content, generate {num_queries} realistic student email queries that students might ask about this information.

        Document Content:
        {content}

        For each query, create a realistic student email with:
        1. A relevant subject line
        2. A polite email body from a student perspective
        3. Student ID (format: ID-1234XXX0 where X is random digits)
        4. The query should be something that can be answered using the document content
                                                           

        Generate the queries in the following JSON format:
        [
            {{
                "subject": "subject line",
                "body": "email body with student introduction and question",
                "query_type": "one of: [
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
            }}
        ]

        Make the queries diverse and realistic. Students should introduce themselves briefly and ask specific questions.
        """)

        try:
            response = self.llm.invoke(
                prompt_template.format_messages(
                    content=document_chunk.page_content,  # Limit content length
                    num_queries=num_queries
                )
            )
            
            # Parse the JSON response
            queries_json = response.content.strip()
            if queries_json.startswith('```json'):
                queries_json = queries_json[7:-3]
            elif queries_json.startswith('```'):
                queries_json = queries_json[3:-3]
            
            queries = json.loads(queries_json)
            return queries
            
        except Exception as e:
            print(f"Error generating queries: {e}")
            return []

    def generate_response(self, query: str, document_content: str) -> str:
        print("Generating response for query")
        """Generate appropriate response for a query using document content"""
        
        prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful department assistant. A student has sent the following email query:

        Query: {query}

        Based on the following department information, provide a helpful and professional response:

        Department Information:
        {content}

        Guidelines for response:
        1. Be polite and professional
        2. Address the student's query directly
        3. Use information from the department documents
        4. If the information is not available in the documents, politely mention that you'll forward their query to the concerned authority
        5. Keep the response concise but complete
        6. Do **not** start with “Thank you for your email” or similar filler phrases.
        7. Do **not** mention the student ID.
        8. Use a professional email format
        9. Use Department Name  : Administration Office, IIT Bhilai                                                    

        Response:
        """)

        try:
            response = self.llm.invoke(
                prompt_template.format_messages(
                    query=query,
                    content=document_content  # Limit content length
                )
            )
            return response.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def create_synthetic_data(self, documents: List[Document], num_samples: int = 100) -> List[Dict]:
        """Create synthetic training data from documents"""
        
        print(f"Creating {num_samples} synthetic training samples...")
        training_data = []
        
        # Chunk documents
        chunked_docs = self.chunk_documents(documents)
        
        if not chunked_docs:
            print("No documents found or loaded!")
            return []
        
        samples_per_chunk = max(1, num_samples // len(chunked_docs))
        
        for i, chunk in enumerate(chunked_docs):
            if len(training_data) >= num_samples:
                break
            # when i%3==0 sleep for 60 seconds
            if i%3==0 and i!=0:
                time.sleep(60)
                print("Taking a short break to avoid rate limits...")
      
            print(f"Processing chunk {i+1}/{len(chunked_docs)}")
            
            # Generate queries for this chunk
            queries = self.generate_synthetic_queries(chunk, num_queries=samples_per_chunk)
            
            for query_data in queries:
                if len(training_data) >= num_samples:
                    break
                
                # Create full query text
                full_query = f"{query_data['subject']}\n{query_data['body']}"
                
                # Generate response
                response = self.generate_response(full_query, chunk.page_content)
                
                # Create synthetic sender info
                sender_name = random.choice(self.student_names)
                sender_email = f"{sender_name.lower().replace(' ', '')}{random.choice(self.email_domains)}"
                
                # Generate timestamp (random time in last 6 months)
                base_date = datetime.now()
                random_days = random.randint(0, 180)
                random_hours = random.randint(8, 18)
                random_minutes = random.randint(0, 59)
                timestamp = base_date - timedelta(days=random_days)
                timestamp = timestamp.replace(hour=random_hours, minute=random_minutes, second=0)
                
                training_sample = {
                    "query": full_query,
                    "sender": f"{sender_name} <{sender_email}>",
                    "category": query_data.get("query_type", "unknown"),
                    "response": response,
                    "timestamp": timestamp.strftime("%a, %d %b %Y %H:%M:%S %z")
                }
                
                training_data.append(training_sample)
        
        return training_data

    def save_to_csv(self, training_data: List[Dict], filename: str = "training_data.csv"):
        """Save training data to CSV file"""
        
        if not training_data:
            print("No training data to save!")
            return
        
        df = pd.DataFrame(training_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Training data saved to {filename}")
        print(f"Total samples: {len(training_data)}")
        
        # Print sample data
        print("\nSample data:")
        # print(df.head(2).to_string())

def main():
    """Main function to generate training data"""

    generator = TrainingDataGenerator(api_key)
    
    # Specify document paths (files or directories)
    document_paths = [
        "data/admin_data",
    ]
    
    print("Loading documents...")
    documents = generator.load_documents(document_paths)
    
    if not documents:
        print("No documents loaded! Please check your document paths.")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    # Generate training data
    num_samples = int(input("Enter number of training samples to generate (default: 50): ") or "50")
    training_data = generator.create_synthetic_data(documents, num_samples)
    
    # Save to CSV
    output_file = "training_data.csv"
    generator.save_to_csv(training_data, output_file)

if __name__ == "__main__":
    main()