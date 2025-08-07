from typing import Any
from .base import BaseTool
from duckduckgo_search import DDGS

class WebSearchTool(BaseTool):
    """
    A tool for searching the web using the DuckDuckGo Search API.
    This tool is accessible and doesn't require an API key.
    """

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def signature(self) -> str:
        return f"{self.name}(query: str, num_results: int = 5) -> str"

    @property
    def description(self) -> str:
        return (
            "Searches the web for the given query using DuckDuckGo and returns "
            "a specified number of top results. Use this to find current information, "
            "answer questions about recent events, or gather context on a topic."
        )

    def run(self, query: str, num_results: int = 5, **kwargs: Any) -> str:
        """
        Executes the web search.

        Args:
            query: The string to search for.
            num_results: The maximum number of results to return.

        Returns:
            A formatted string containing the search results, or an error message.
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            
            if not results:
                return f"No results found for query: '{query}'"

            # Format the results into a clean string for the LLM
            formatted_results = []
            for i, res in enumerate(results):
                formatted_results.append(
                    f"{i+1}. Title: {res.get('title', 'N/A')}\n"
                    f"   Snippet: {res.get('body', 'N/A')}\n"
                    f"   URL: {res.get('href', 'N/A')}"
                )
            
            return "\n---\n".join(formatted_results)

        except Exception as e:
            return f"Error performing web search for query '{query}': {e}"


