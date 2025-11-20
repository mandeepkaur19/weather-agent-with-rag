"""Unit tests for AIAgent."""
import pytest
from unittest.mock import Mock, MagicMock
from agent import AIAgent
from weather_service import WeatherService
from rag_service import RAGService


class TestAIAgent:
    """Test cases for AIAgent."""
    
    @pytest.fixture
    def mock_weather_service(self):
        """Create a mock WeatherService."""
        service = MagicMock(spec=WeatherService)
        service.get_weather.return_value = {
            "city": "London",
            "country": "GB",
            "temperature": 15.5,
            "feels_like": 14.2,
            "humidity": 65,
            "pressure": 1013,
            "description": "clear sky",
            "wind_speed": 3.5,
            "units": "metric"
        }
        service.format_weather_response.return_value = "Weather in London: 15.5°C"
        return service
    
    @pytest.fixture
    def mock_rag_service(self):
        """Create a mock RAGService."""
        service = MagicMock(spec=RAGService)
        service.query.return_value = {
            "answer": "This is a test answer from RAG.",
            "sources": [],
            "retrieved_chunks": []
        }
        return service
    
    @pytest.fixture
    def agent(self, mock_weather_service, mock_rag_service):
        """Create an AIAgent instance for testing."""
        from unittest.mock import patch
        with patch('agent.ChatOpenAI'):
            return AIAgent(mock_weather_service, mock_rag_service)
    
    def test_weather_routing(self, agent, mock_weather_service):
        """Test that weather queries are routed correctly."""
        result = agent.process_query("What's the weather in London?")
        
        assert result["route"] == "weather"
        assert "response" in result
        mock_weather_service.get_weather.assert_called()
    
    def test_rag_routing(self, agent, mock_rag_service):
        """Test that non-weather queries are routed to RAG."""
        result = agent.process_query("What is machine learning?")
        
        assert result["route"] == "rag"
        assert "response" in result
        mock_rag_service.query.assert_called()
    
    def test_city_extraction(self, agent):
        """Test city name extraction from queries."""
        test_cases = [
            ("What's the weather in London?", "London"),
            ("Temperature in New York", "New York"),
            ("Weather for Paris", "Paris"),
        ]
        
        for query, expected_city in test_cases:
            city = agent._extract_city_from_query(query)
            assert expected_city.lower() in city.lower() or city.lower() in expected_city.lower()

