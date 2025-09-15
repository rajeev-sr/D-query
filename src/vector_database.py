import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path

class VectorDatabase:
    def __init__(self, db_path="data/vector_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Setup embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Collection name
        self.collection_name = "institute_knowledge"
        self.collection = None
        
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or load collection"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Loaded existing collection: {self.collection_name}")
            
        except Exception:
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Institute knowledge base for student queries"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to vector database"""
        
        all_texts = []
        all_ids = []
        all_metadata = []
        
        for doc_idx, doc in enumerate(documents):
            for chunk_idx, chunk in enumerate(doc['chunks']):
                # Create unique ID
                doc_id = f"{doc['filename']}_{doc_idx}_{chunk_idx}"
                
                # Prepare metadata
                metadata = {
                    'filename': doc['filename'],
                    'doc_type': doc['doc_type'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(doc['chunks']),
                    'content_length': len(chunk)
                }
                
                all_texts.append(chunk)
                all_ids.append(doc_id)
                all_metadata.append(metadata)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            batch_metadata = all_metadata[i:i + batch_size]
            
            self.collection.add(
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_metadata
            )
        
        print(f"Added {len(all_texts)} chunks to vector database")
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant documents"""
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'distances', 'metadatas']
            )
            
            # Format results
            formatted_results = []
            
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'source': results['metadatas'][0][i]['filename']
                })
            
            return {
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results)
            }
            
        except Exception as e:
            print(f"Search error: {e}")
            return {'query': query, 'results': [], 'total_results': 0}
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        except Exception as e:
            return {'error': str(e)}