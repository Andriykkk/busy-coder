from typing import List, Tuple
from .action import Action

class State:
    """
    Represents the state of the agent-environment interaction, including turn limits.
    """
    def __init__(self, task: str, observation: str, interaction_limit: int = 5, solution_limit: int = 2):
        self.task: str = task
        self.history: List[Tuple[str, Action, str]] = []
        self.observation: str = observation
        
        # Core state flags
        self.finished: bool = False
        self.success: bool = False
        
        # Turn counting and limits
        self.interaction_limit: int = interaction_limit
        self.solution_limit: int = solution_limit
        self.interaction_count: int = 0
        self.solution_attempts: int = 0

    def add_to_history(self, thought: str, action: Action, observation: str):
        """Adds a new step to the history and increments the interaction counter."""
        self.history.append((thought, action, observation))
        self.observation = observation
        self.interaction_count += 1
        
        if action.tool_name == "finish":
            self.solution_attempts += 1

    @property
    def interactions_left(self) -> int:
        return self.interaction_limit - self.interaction_count

    @property
    def solutions_left(self) -> int:
        return self.solution_limit - self.solution_attempts

    @property
    def summary(self) -> str:
        return (
            f"Task: {self.task}\n"
            f"Finished: {self.finished}, Success: {self.success}\n"
            f"Turns Taken: {self.interaction_count}/{self.interaction_limit}\n"
            f"Solutions Attempted: {self.solution_attempts}/{self.solution_limit}"
        )

