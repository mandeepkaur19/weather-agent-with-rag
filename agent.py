"""LangGraph agent implementation with decision node for weather vs RAG routing."""
from typing import Literal, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from weather_service import WeatherService
from rag_service import RAGService
from config import LLM_MODEL, OPENAI_API_KEY


class AgentState(TypedDict):
    """State schema for the LangGraph agent."""
    messages: Annotated[list, "list of messages"]
    query: str
    route: str
    response: str


class AIAgent:
    """LangGraph agent that routes between weather API and RAG service."""
    
    def __init__(self, weather_service: WeatherService, rag_service: RAGService):
        """Initialize AIAgent with weather and RAG services."""
        self.weather_service = weather_service
        self.rag_service = rag_service
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route", self._route_query)
        workflow.add_node("weather", self._handle_weather)
        workflow.add_node("rag", self._handle_rag)
        
        # Set entry point
        workflow.set_entry_point("route")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "route",
            self._should_use_weather,
            {
                "weather": "weather",
                "rag": "rag"
            }
        )
        
        # Add edges to end
        workflow.add_edge("weather", END)
        workflow.add_edge("rag", END)
        
        return workflow.compile()
    
    def _route_query(self, state: AgentState) -> AgentState:
        """Route node: receives the query and prepares for routing."""
        return state
    
    def _should_use_weather(self, state: AgentState) -> Literal["weather", "rag"]:
        """
        Decision node: determines whether to use weather API or RAG.
        
        Args:
            state: Current agent state
            
        Returns:
            "weather" or "rag" based on query intent
        """
        query = state.get("query", "").lower()
        
        # Keywords that suggest weather query
        weather_keywords = [
            "weather", "temperature", "forecast", "humidity", 
            "wind", "rain", "snow", "climate", "temperature in",
            "weather in", "how's the weather", "what's the weather"
        ]
        
        # Check if query contains weather-related keywords
        if any(keyword in query for keyword in weather_keywords):
            return "weather"
        
        # Otherwise, use RAG
        return "rag"
    
    def _handle_weather(self, state: AgentState) -> AgentState:
        """
        Weather node: handles weather-related queries.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with weather response
        """
        query = state.get("query", "")
        
        try:
            # Extract city name from query
            # Simple extraction - look for "in [city]" or "for [city]"
            city = self._extract_city_from_query(query)
            
            if not city:
                response = "I couldn't identify a city name in your query. Please specify a city, for example: 'What's the weather in London?'"
            else:
                weather_data = self.weather_service.get_weather(city)
                response = self.weather_service.format_weather_response(weather_data)
            
            state["response"] = response
            state["route"] = "weather"
            return state
            
        except Exception as e:
            state["response"] = f"Sorry, I encountered an error fetching weather data: {str(e)}"
            state["route"] = "weather"
            return state
    
    def _handle_rag(self, state: AgentState) -> AgentState:
        """
        RAG node: handles document-based queries.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with RAG response
        """
        query = state.get("query", "")
        
        try:
            result = self.rag_service.query(query)
            state["response"] = result["answer"]
            state["route"] = "rag"
            return state
            
        except Exception as e:
            state["response"] = f"Sorry, I encountered an error processing your query: {str(e)}"
            state["route"] = "rag"
            return state
    
    def _extract_city_from_query(self, query: str) -> str:
        """
        Extract city name from query string.
        
        Args:
            query: User query string
            
        Returns:
            Extracted city name or empty string
        """
        query_lower = query.lower()
        
        # Common patterns
        patterns = [
            "in ",
            "for ",
            "at ",
            "weather ",
            "temperature "
        ]
        
        for pattern in patterns:
            if pattern in query_lower:
                # Extract text after pattern
                parts = query_lower.split(pattern, 1)
                if len(parts) > 1:
                    city_part = parts[1].strip()
                    # Remove question marks and other punctuation
                    city_part = city_part.rstrip("?.,!").strip()
                    # Take first word or phrase (handle multi-word cities)
                    words = city_part.split()
                    # Try to get city name (first 1-3 words)
                    if len(words) >= 1:
                        # Common multi-word cities
                        if len(words) >= 2 and words[0] in ["new", "san", "los", "saint", "st"]:
                            city = " ".join(words[:2])
                        else:
                            city = words[0]
                        return city.title()
        
        return ""
    
    def process_query(self, query: str) -> dict:
        """
        Process a user query through the agent.
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary containing response and metadata
        """
        initial_state = {
            "messages": [],
            "query": query,
            "route": "",
            "response": ""
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "response": final_state["response"],
            "route": final_state["route"]
        }

