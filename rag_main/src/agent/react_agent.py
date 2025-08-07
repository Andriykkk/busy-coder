import re
import torch
from typing import List, Dict, Any

from ..datamodels.state import State
from ..datamodels.action import Action
from ..tools.base import BaseTool
from .base import BaseAgent

# The system prompt is crucial and should match the one used during fine-tuning.
SYSTEM_PROMPT = """You are a helpful assistant assigned with the task of problem-solving. To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions to assist you throughout the process.

At each turn, you should first provide your step-by-step thinking for solving the task. After that, you have two options:

1) Interact with a Python programming environment and receive the corresponding output. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>.
2) Directly provide a solution that adheres to the required format for the given task. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>.

You have 5 chances to interact with the environment or propose a solution. You can only propose a solution 2 times."""


class ReactAgent(BaseAgent):
    """
    A ReAct agent fine-tuned on the code-act dataset format.
    It uses a conversational prompt and parses actions from <execute> tags.
    """

    def __init__(self, llm_client: Dict[str, Any], tools: List[BaseTool]):
        """
        Args:
            llm_client: A dictionary containing the 'model' and 'tokenizer'.
            tools: A list of tools available to the agent (used for schema).
        """
        self.model = llm_client["model"]
        self.tokenizer = llm_client["tokenizer"]
        self.tools = tools

    def act(self, state: State) -> Action:
        """
        The core logic of the agent.
        1. Creates a conversational prompt from the current state.
        2. Generates a response from the LLM.
        3. Parses the LLM's output to extract thought and action.
        """
        prompt = self._create_prompt(state)
        
        # --- This is the real LLM call ---
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        llm_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # The model's output includes the prompt, so we need to remove it.
        response_text = llm_output[len(prompt):].strip()

        return self._parse_output(response_text)

    def _create_prompt(self, state: State) -> str:
        """
        Creates the full conversational prompt in the format the model was trained on.
        """
        # 1. Start with the system prompt
        prompt = f"### System: {SYSTEM_PROMPT}\n"
        
        # 2. Add the initial user task
        prompt += f"### User: \nTask:\n{state.task}\n"

        # 3. Reconstruct the conversation from history
        for thought, action, observation in state.history:
            # Assistant's turn
            assistant_response = thought
            if action.tool_name == "code_executor":
                assistant_response += f"\n<execute>\n{action.tool_args.get('code', '')}\n</execute>"
            elif action.tool_name == "finish":
                 assistant_response += f"<solution> {action.tool_args.get('answer', '')} </solution>"
            
            prompt += f"### Assistant: {assistant_response}\n"
            
            # Environment's observation (as a User turn)
            prompt += f"### User: \nObservation:\n{observation}\n"

        # 4. Cue the model for its next response
        prompt += "### Assistant: "
        return prompt

    def _parse_output(self, llm_output: str) -> Action:
        """
        Parses the raw LLM output to find <execute> or <solution> tags.
        """
        # Try to find an <execute> tag for code execution
        execute_match = re.search(r"<execute>(.*?)</execute>", llm_output, re.DOTALL)
        if execute_match:
            code = execute_match.group(1).strip()
            # The thought is everything outside the tag
            thought = re.sub(r"<execute>.*?</execute>", "", llm_output, flags=re.DOTALL).strip()
            return Action(
                tool_name="code_executor",
                tool_args={"code": code},
                thought=thought,
                raw_llm_output=llm_output
            )

        # Try to find a <solution> tag for the final answer
        solution_match = re.search(r"<solution>(.*?)</solution>", llm_output, re.DOTALL)
        if solution_match:
            answer = solution_match.group(1).strip()
            thought = re.sub(r"<solution>.*?</solution>", "", llm_output, flags=re.DOTALL).strip()
            return Action(
                tool_name="finish",
                tool_args={"answer": answer},
                thought=thought,
                raw_llm_output=llm_output
            )

        # If no tags are found, treat the entire output as a thought and finish.
        return Action(
            tool_name="finish",
            tool_args={"answer": f"No action taken. Assistant response: {llm_output}"},
            thought=llm_output,
            raw_llm_output=llm_output
        )

