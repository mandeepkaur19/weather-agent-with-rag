"""Weather service for fetching real-time weather data from OpenWeatherMap API."""
import requests
from typing import Dict, Optional
from config import OPENWEATHERMAP_API_KEY


class WeatherService:
    """Service to fetch weather data from OpenWeatherMap API."""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize WeatherService with API key."""
        self.api_key = api_key or OPENWEATHERMAP_API_KEY
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is required")
    
    def get_weather(self, city: str, units: str = "metric") -> Dict:
        """
        Fetch weather data for a given city.
        
        Args:
            city: Name of the city
            units: Temperature units (metric, imperial, kelvin)
            
        Returns:
            Dictionary containing weather information
        """
        params = {
            "q": city,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "description": data["weather"][0]["description"],
                "wind_speed": data.get("wind", {}).get("speed", 0),
                "visibility": data.get("visibility", 0) / 1000 if data.get("visibility") else None,
                "units": units
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch weather data: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected API response format: {str(e)}")
    
    def format_weather_response(self, weather_data: Dict) -> str:
        """
        Format weather data into a readable string.
        
        Args:
            weather_data: Dictionary containing weather information
            
        Returns:
            Formatted weather string
        """
        unit_symbol = "°C" if weather_data["units"] == "metric" else "°F"
        visibility = weather_data.get("visibility")
        visibility_text = f"{visibility:.1f} km" if visibility is not None else "—"

        wind_speed = weather_data.get("wind_speed", 0)
        wind_speed_kmh = wind_speed * 3.6

        response_lines = [
            f"**🌦 Weather · {weather_data['city']}, {weather_data['country']}**",
            "",
            f"- **Temperature:** {weather_data['temperature']:.1f}{unit_symbol} (feels like {weather_data['feels_like']:.1f}{unit_symbol})",
            f"- **Conditions:** {weather_data['description'].title()}",
            f"- **Humidity / Pressure:** {weather_data['humidity']}% · {weather_data['pressure']} hPa",
            f"- **Wind:** {wind_speed:.1f} m/s ({wind_speed_kmh:.1f} km/h)",
            f"- **Visibility:** {visibility_text}",
        ]

        return "\n".join(response_lines)

