from pydantic import BaseModel, Field
from typing import Dict, Any

class Action(BaseModel):
    """
    Represents a single action to be taken by the agent.

    This is a structured representation of the LLM's output, parsed
    into a specific tool call with its corresponding arguments.
    """
    tool_name: str = Field(
        ...,
        description="The name of the tool to be executed (e.g., 'code_executor', 'web_search')."
    )
    tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of arguments to pass to the tool."
    )
    thought: str = Field(
        "",
        description="The agent's reasoning or thought process that led to this action."
    )
    raw_llm_output: str = Field(
        "",
        description="The raw, unparsed output from the language model, useful for debugging."
    )

    def __str__(self):
        """Provides a clean, readable string representation of the action."""
        if self.tool_name == "finish":
            return f"Finish with args: {self.tool_args}"
        return f"Tool: {self.tool_name}, Args: {self.tool_args}"