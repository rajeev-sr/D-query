from src.knowledge_base import DocumentProcessor
from src.vector_database import VectorDatabase
from typing import List, Dict, Any
import json

class RAGSystem:
    def __init__(self, docs_dir="data/institute_docs", db_path="data/vector_db"):
        self.doc_processor = DocumentProcessor(docs_dir)
        self.vector_db = VectorDatabase(db_path)
        self.knowledge_loaded = False
        
    def setup_knowledge_base(self, force_reload=False):
        """Setup the knowledge base"""
        
        # Check if knowledge base already exists
        stats = self.vector_db.get_stats()
        
        if not force_reload and stats.get('total_chunks', 0) > 0:
            print(f"Knowledge base already loaded: {stats['total_chunks']} chunks")
            self.knowledge_loaded = True
            return True
        
        print("Setting up knowledge base...")
        
        # Process documents
        documents = self.doc_processor.process_all_documents()
        
        if not documents:
            print("No documents found to process")
            return False
        
        # Add to vector database
        self.vector_db.add_documents(documents)
        
        # Verify setup
        stats = self.vector_db.get_stats()
        print(f"Knowledge base setup complete: {stats}")
        
        self.knowledge_loaded = True
        return True
    
    def retrieve_context(self, query: str, max_results: int = 3) -> Dict:
        """Retrieve relevant context for a query"""
        
        if not self.knowledge_loaded:
            print("Knowledge base not loaded. Setting up...")
            if not self.setup_knowledge_base():
                return {'context': '', 'sources': [], 'error': 'Knowledge base setup failed'}
        
        # Search for relevant documents
        search_results = self.vector_db.search(query, n_results=max_results)
        
        if not search_results['results']:
            return {
                'context': 'No relevant information found in knowledge base.',
                'sources': [],
                'confidence': 0.0
            }
        
        # Combine relevant chunks into context
        context_parts = []
        sources = []
        total_similarity = 0
        
        for result in search_results['results']:
            if result['similarity_score'] > 0.3:  # Minimum relevance threshold
                context_parts.append(f"From {result['source']}: {result['content']}")
                sources.append(result['source'])
                total_similarity += result['similarity_score']
        
        context = "\n\n".join(context_parts)
        confidence = total_similarity / len(search_results['results']) if search_results['results'] else 0
        
        return {
            'context': context,
            'sources': list(set(sources)),  # Remove duplicates
            'confidence': confidence,
            'num_sources': len(set(sources))
        }
    
    def test_retrieval(self):
        """Test the RAG system with sample queries"""
        
        test_queries = [
            "When are the final exams?",
            "How do I pay fees?",
            "I forgot my password",
            "Assignment submission guidelines",
            "What is the late fee penalty?"
        ]
        
        print("Testing RAG retrieval system...")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            result = self.retrieve_context(query)
            
            print(f"Sources found: {result.get('num_sources', 0)}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            
            if result['context']:
                print(f"Context preview: {result['context'][:200]}...")
            else:
                print("No relevant context found")
            
            print("-" * 50)