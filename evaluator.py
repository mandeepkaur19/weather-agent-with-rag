"""LangSmith evaluation module for LLM response evaluation."""
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from config import LANGSMITH_API_KEY, LANGSMITH_PROJECT


class ResponseEvaluator:
    """Evaluator for LLM responses using LangSmith."""
    
    def __init__(self):
        """Initialize ResponseEvaluator with LangSmith client."""
        self.client = Client(api_key=LANGSMITH_API_KEY) if LANGSMITH_API_KEY else None
        self.project_name = LANGSMITH_PROJECT
    
    def evaluate_response(self, query: str, response: str, route: str, metadata: dict = None):
        """
        Log and evaluate a response using LangSmith.
        
        Args:
            query: User query
            response: LLM response
            route: Route taken (weather or rag)
            metadata: Optional metadata dictionary
        """
        if not self.client:
            print("LangSmith API key not configured. Skipping evaluation.")
            return
        
        try:
            # Create a run for evaluation
            run = self.client.create_run(
                name=f"{route}_query",
                run_type="chain",
                inputs={"query": query},
                outputs={"response": response, "route": route},
                project_name=self.project_name,
                metadata=metadata or {}
            )
            
            # Evaluate response quality
            evaluation = self._evaluate_quality(query, response, route)
            
            if run is None:
                print("LangSmith run was not created; skipping feedback logging.")
                return {
                    "run_id": None,
                    "evaluation": evaluation
                }
            
            # Log evaluation
            self.client.create_feedback(
                run_id=run.id,
                key="response_quality",
                value=evaluation["score"],
                comment=evaluation["comment"]
            )
            
            return {
                "run_id": run.id,
                "evaluation": evaluation
            }
            
        except Exception as e:
            print(f"Error in LangSmith evaluation: {str(e)}")
            return None
    
    def _evaluate_quality(self, query: str, response: str, route: str) -> dict:
        """
        Evaluate response quality based on heuristics.
        
        Args:
            query: User query
            response: LLM response
            route: Route taken
            
        Returns:
            Dictionary with score and comment
        """
        score = 0.0
        comments = []
        
        # Check response length
        if len(response) > 10:
            score += 0.3
        else:
            comments.append("Response too short")
        
        # Check if response addresses the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        
        if overlap > 0:
            score += 0.3
        else:
            comments.append("Response may not address query")
        
        # Check for error messages
        error_indicators = ["error", "failed", "sorry", "couldn't", "unable"]
        has_error = any(indicator in response.lower() for indicator in error_indicators)
        
        if not has_error or "sorry" in response.lower():
            score += 0.2
        else:
            comments.append("Response contains error indicators")
        
        # Route-specific checks
        if route == "weather":
            weather_indicators = ["temperature", "humidity", "wind", "weather"]
            if any(indicator in response.lower() for indicator in weather_indicators):
                score += 0.2
        elif route == "rag":
            if len(response) > 50:  # RAG responses should be more detailed
                score += 0.2
        
        # Normalize score to 0-1
        score = min(1.0, score)
        
        return {
            "score": score,
            "comment": "; ".join(comments) if comments else "Good response"
        }
    
    def get_tracer(self):
        """Get LangChain tracer for automatic logging."""
        if not self.client:
            return None
        
        return LangChainTracer(
            project_name=self.project_name,
            client=self.client
        )

