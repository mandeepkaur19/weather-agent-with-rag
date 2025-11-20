"""Vector store implementation using Qdrant for storing and retrieving embeddings."""
import hashlib
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, EMBEDDING_MODEL, OPENAI_API_KEY


class VectorStore:
    """Vector store for managing embeddings in Qdrant."""
    
    def __init__(self, collection_name: Optional[str] = None):
        """Initialize VectorStore with Qdrant client and embeddings model."""
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists, create if it doesn't."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI text-embedding-3-small dimension
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            raise Exception(f"Failed to ensure collection exists: {str(e)}")
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional list of metadata dictionaries for each text
        """
        if not texts:
            return
        
        # Generate embeddings
        embeddings_list = self.embeddings.embed_documents(texts)
        
        # Prepare points with unique IDs
        points = []
        for idx, (text, embedding) in enumerate(zip(texts, embeddings_list)):
            metadata = metadatas[idx] if metadatas else {}
            metadata["text"] = text
            
            # Generate unique integer ID using hash of text
            # This ensures same text gets same ID (idempotent)
            text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            # Add index to ensure uniqueness even for identical texts
            point_id = text_hash + idx
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )
            )
        
        # Upsert points
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        except Exception as e:
            raise Exception(f"Failed to add documents to vector store: {str(e)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing text and metadata
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in Qdrant
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.payload.get("text", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                    "score": result.score
                })
            
            return formatted_results
        except Exception as e:
            raise Exception(f"Failed to search vector store: {str(e)}")
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
        except Exception as e:
            raise Exception(f"Failed to clear collection: {str(e)}")

