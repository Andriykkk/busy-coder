from pydantic import BaseModel, Field
from typing import List, Tuple

from .action import Action # Import the Action model from the same directory

class State(BaseModel):
    """
    Represents the complete state of the agent-environment interaction at any point.

    This object is passed back and forth between the agent and the environment,
    accumulating the history of the interaction.
    """
    task: str = Field(
        ...,
        description="The initial, high-level task assigned by the user."
    )
    history: List[Tuple[str, Action, str]] = Field(
        default_factory=list,
        description="A list of (thought, action, observation) triplets, recording the interaction history."
    )
    observation: str = Field(
        "",
        description="The most recent observation from the environment after the last action."
    )
    finished: bool = Field(
        False,
        description="A flag indicating whether the task has been completed."
    )
    success: bool = Field(
        False,
        description="A flag indicating whether the task was completed successfully."
    )

    @property
    def summary(self) -> str:
        """Provides a brief summary of the current state."""
        if self.finished:
            status = "Succeeded" if self.success else "Failed"
            return f"Task finished with status: {status}"
        return f"Task ongoing. Last observation: {self.observation[:100]}..."

    def add_to_history(self, thought: str, action: Action, observation: str):
        """Helper method to append a new step to the history."""
        self.history.append((thought, action, observation))
        self.observation = observation