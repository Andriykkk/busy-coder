from typing import List, Dict, Any

from ..datamodels.state import State
from ..datamodels.action import Action
from ..tools.base import BaseTool

class Sandbox:
    """
    The Sandbox environment manages the state and the execution of tools.

    It acts as the intermediary between the agent and the tools, taking an
    action from the agent and returning the resulting observation.
    """

    def __init__(self, tools: List[BaseTool]):
        """
        Initializes the environment with a list of available tools.

        Args:
            tools: A list of tool instances that inherit from BaseTool.
        """
        self._tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        LOGGER.info(f"Sandbox initialized with tools: {list(self._tools.keys())}")

    def reset(self, task: str) -> State:
        """
        Resets the environment to a new initial state for a given task.

        Args:
            task: The high-level task description from the user.

        Returns:
            A new State object initialized for the task.
        """
        LOGGER.info(f"Resetting sandbox for new task: {task}")
        initial_observation = "Environment is ready. What is the first step?"
        return State(task=task, observation=initial_observation)

    def step(self, state: State, action: Action) -> State:
        """
        Executes a single step in the environment based on the agent's action.

        Args:
            state: The current state of the interaction.
            action: The action to be executed.

        Returns:
            The new state after executing the action.
        """
        # Handle the special 'finish' action
        if action.tool_name == "finish":
            state.finished = True
            # You could add logic here to check if the answer is correct
            state.success = True
            state.observation = f"Task marked as finished by the agent with args: {action.tool_args}"
            LOGGER.info(state.observation)
            return state

        # Find and execute the specified tool
        tool = self._tools.get(action.tool_name)
        if tool is None:
            observation = f"Error: Tool '{action.tool_name}' not found. Available tools are: {list(self._tools.keys())}"
            LOGGER.error(observation)
        else:
            try:
                LOGGER.info(f"Executing tool: {action.tool_name} with args: {action.tool_args}")
                observation = tool.run(**action.tool_args)
            except Exception as e:
                observation = f"Error executing tool '{action.tool_name}': {e}"
                LOGGER.error(observation)

        # Update the state history
        state.add_to_history(action.thought, action, observation)
        return state

# It's good practice to have a logger instance available for environment-specific logs
import logging
LOGGER = logging.getLogger(__name__)