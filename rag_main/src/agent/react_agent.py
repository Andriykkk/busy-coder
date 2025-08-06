import re
from typing import List, Dict, Any

from ..datamodels.state import State
from ..datamodels.action import Action
from ..tools.base import BaseTool
from .base import BaseAgent

# A placeholder for a generic LLM client. In a real implementation, this
# would be an instance of a class that handles API calls to OpenAI, Anthropic, etc.
LLMClient = Any

class ReactAgent(BaseAgent):
    """
    A ReAct (Reasoning + Acting) agent that uses an LLM to decide on the next
    action based on the current state and available tools.
    """

    def __init__(self, llm_client: LLMClient, tools: List[BaseTool]):
        """
        Initializes the ReactAgent.

        Args:
            llm_client: A client for interacting with a large language model.
            tools: A list of tools available to the agent.
        """
        self.llm_client = llm_client
        self.tools = tools

    def act(self, state: State) -> Action:
        """
        The core logic of the ReAct agent.
        1. Creates a prompt from the current state and tools.
        2. Sends the prompt to the LLM.
        3. Parses the LLM's output into a structured Action.
        """
        prompt = self._create_prompt(state)
        
        # In a real implementation, you would call the LLM here.
        # For now, we'll use a placeholder response.
        # llm_output = self.llm_client.generate(prompt)
        llm_output = self._get_placeholder_response(state) # Placeholder

        return self._parse_output(llm_output)

    def _create_prompt(self, state: State) -> str:
        """
        Creates the full prompt to be sent to the LLM.
        """
        # 1. System instruction and tool descriptions
        tool_schemas = [tool.get_schema() for tool in self.tools]
        tool_text = "\n".join([
            f"- {s['name']}{s['signature']}: {s['description']}" for s in tool_schemas
        ])
        
        prompt = (
            "You are a helpful assistant that can solve complex tasks by breaking them down "
            "into smaller steps. You have access to the following tools:\n"
            f"{tool_text}\n\n"
            "To solve the task, you must use the following format, thinking step-by-step:\n"
            "Thought: [Your reasoning and plan for the next step.]\n"
            "Action: [A single tool call from the list above, or 'finish(answer=...)' when you are done.]\n\n"
            "--- Start of Task ---\n"
        )

        # 2. The task and history
        prompt += f"Task: {state.task}\n"
        for thought, action, observation in state.history:
            prompt += f"Thought: {thought}\n"
            prompt += f"Action: {action}\n"
            prompt += f"Observation: {observation}\n"
        
        # 3. The final prompt for the next action
        prompt += "Thought: [Your next thought here]\n"
        prompt += "Action: [Your next action here]\n"
        
        return prompt

    def _parse_output(self, llm_output: str) -> Action:
        """
        Parses the raw LLM output into a structured Action object.
        """
        thought_match = re.search(r"Thought:\s*(.*)", llm_output, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*)", llm_output, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action_str = action_match.group(1).strip() if action_match else ""

        if not action_str:
            # Handle cases where parsing fails
            return Action(tool_name="finish", tool_args={"answer": "Error: Could not parse LLM output."})

        tool_name_match = re.match(r"(\w+)\(|", action_str)
        tool_name = tool_name_match.group(1) if tool_name_match else ""

        args_str_match = re.search(r"\((.*)\)", action_str)
        args_str = args_str_match.group(1) if args_str_match else ""
        
        # Simple argument parsing, can be improved with more robust logic
        try:
            # This is a simplified parser. A real one would handle quotes, etc.
            tool_args = dict(arg.split("=") for arg in args_str.split(", ")) if args_str else {}
            # Clean up quotes from values
            tool_args = {k: v.strip("'\"") for k, v in tool_args.items()}
        except ValueError:
            tool_args = {"raw_args": args_str} # Fallback for parsing failure

        return Action(
            tool_name=tool_name,
            tool_args=tool_args,
            thought=thought,
            raw_llm_output=llm_output
        )

    def _get_placeholder_response(self, state: State) -> str:
        """
        A placeholder to simulate LLM responses for development.
        """
        if not state.history:
            return 'Thought: I need to find out what a "CodeAct RAG project" is. I will use the web search tool.\nAction: web_search(query="CodeAct RAG project")'
        else:
            return 'Thought: I have the search results. Now I will use the code executor to write a summary to a file.\nAction: code_executor(code="with open(\'summary.txt\', \'w\') as f: f.write(\'This is a summary.\')")'

import logging
LOGGER = logging.getLogger(__name__)