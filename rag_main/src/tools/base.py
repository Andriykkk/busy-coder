from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTool(ABC):
    """
    Abstract Base Class for all tools that the agent can use.

    This class defines the essential interface that every tool must implement.
    The 'signature' and 'description' are crucial for the LLM to understand
    how and when to use the tool.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the tool (e.g., 'code_executor')."""
        pass

    @property
    @abstractmethod
    def signature(self) -> str:
        """
        The function signature of the tool, telling the LLM how to call it.
        Example: 'code_executor(code: str) -> str'
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A brief description of what the tool does and when to use it."""
        pass

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """
        The core logic of the tool. This method executes the tool's function.
        It receives arguments as keyword arguments and must return a single
        string as the observation.
        """
        pass

    def get_schema(self) -> Dict[str, str]:
        """Returns a dictionary schema representing the tool."""
        return {
            "name": self.name,
            "signature": self.signature,
            "description": self.description,
        }