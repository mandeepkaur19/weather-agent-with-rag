"""Unit tests for RAGService."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_service import RAGService
from vector_store import VectorStore


class TestRAGService:
    """Test cases for RAGService."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore."""
        store = MagicMock(spec=VectorStore)
        return store
    
    @pytest.fixture
    def rag_service(self, mock_vector_store):
        """Create a RAGService instance for testing."""
        with patch('rag_service.ChatOpenAI'):
            return RAGService(mock_vector_store)
    
    def test_query_with_results(self, rag_service, mock_vector_store):
        """Test RAG query with retrieved results."""
        # Mock vector store search
        mock_vector_store.search.return_value = [
            {
                "text": "This is a test document about AI.",
                "metadata": {"source": "test.pdf"},
                "score": 0.95
            }
        ]
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Based on the document, AI is a test topic."
        rag_service.llm = Mock()
        rag_service.llm.invoke = Mock(return_value=mock_response)
        
        # Test
        result = rag_service.query("What is AI?")
        
        # Assertions
        assert "answer" in result
        assert len(result["sources"]) > 0
        mock_vector_store.search.assert_called_once()
    
    def test_query_no_results(self, rag_service, mock_vector_store):
        """Test RAG query with no retrieved results."""
        # Mock empty search results
        mock_vector_store.search.return_value = []
        
        # Test
        result = rag_service.query("What is AI?")
        
        # Assertions
        assert "couldn't find" in result["answer"].lower()
        assert len(result["sources"]) == 0

