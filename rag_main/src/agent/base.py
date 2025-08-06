from abc import ABC, abstractmethod
from ..datamodels.state import State
from ..datamodels.action import Action

class BaseAgent(ABC):
    """
    Abstract Base Class for all agents.

    This class defines the core interface for an agent: the ability to
    receive the current state of the world and decide on the next action.
    """

    @abstractmethod
    def act(self, state: State) -> Action:
        """
        The core method of the agent. It takes the current state and returns
        the next action to be taken.

        Args:
            state: The current State object, containing the task and history.

        Returns:
            An Action object representing the agent's decided next move.
        """
        pass