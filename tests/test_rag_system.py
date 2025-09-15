# test_rag_system.py
from src.rag_system import RAGSystem
import os

def test_rag_system():
    print("=== TESTING RAG SYSTEM ===")
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = RAGSystem()
    
    # Setup knowledge base
    print("Setting up knowledge base...")
    if not rag.setup_knowledge_base():
        print("Failed to setup knowledge base")
        return False
    
    # Get statistics
    stats = rag.vector_db.get_stats()
    print(f"Knowledge base stats: {stats}")
    
    # Test retrieval
    print("\nTesting retrieval...")
    rag.test_retrieval()
    
    # Test specific query
    print("\nTesting specific query...")
    query = "When is the exam and what are the dates?"
    result = rag.retrieve_context(query)
    
    print(f"Query: {query}")
    print(f"Sources: {result['sources']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Context:\n{result['context'][:300]}...")
    
    print("\nRAG system test completed!")
    return True

if __name__ == "__main__":
    test_rag_system()