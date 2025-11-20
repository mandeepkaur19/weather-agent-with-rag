"""Unit tests for WeatherService."""
import pytest
from unittest.mock import Mock, patch
from weather_service import WeatherService


class TestWeatherService:
    """Test cases for WeatherService."""
    
    @pytest.fixture
    def weather_service(self):
        """Create a WeatherService instance for testing."""
        return WeatherService(api_key="test_api_key")
    
    @patch('weather_service.requests.get')
    def test_get_weather_success(self, mock_get, weather_service):
        """Test successful weather data retrieval."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {
                "temp": 15.5,
                "feels_like": 14.2,
                "humidity": 65,
                "pressure": 1013
            },
            "weather": [{"description": "clear sky"}],
            "wind": {"speed": 3.5},
            "visibility": 10000
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test
        result = weather_service.get_weather("London")
        
        # Assertions
        assert result["city"] == "London"
        assert result["country"] == "GB"
        assert result["temperature"] == 15.5
        assert result["description"] == "clear sky"
        mock_get.assert_called_once()
    
    @patch('weather_service.requests.get')
    def test_get_weather_api_error(self, mock_get, weather_service):
        """Test handling of API errors."""
        # Mock API error
        mock_get.side_effect = Exception("API Error")
        
        # Test
        with pytest.raises(Exception) as exc_info:
            weather_service.get_weather("London")
        
        assert "Failed to fetch weather data" in str(exc_info.value)
    
    def test_format_weather_response(self, weather_service):
        """Test weather response formatting."""
        weather_data = {
            "city": "London",
            "country": "GB",
            "temperature": 15.5,
            "feels_like": 14.2,
            "humidity": 65,
            "pressure": 1013,
            "description": "clear sky",
            "wind_speed": 3.5,
            "visibility": 10,
            "units": "metric"
        }
        
        result = weather_service.format_weather_response(weather_data)
        
        assert "London" in result
        assert "15.5°C" in result
        assert "clear sky" in result
        assert "65%" in result

