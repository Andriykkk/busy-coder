import re
import torch
from typing import List, Dict, Any

from ..datamodels.state import State
from ..datamodels.action import Action
from ..tools.base import BaseTool
from .base import BaseAgent

class ReactAgent(BaseAgent):
    """
    A ReAct agent fine-tuned on the code-act dataset format.
    It uses a conversational prompt and parses actions from <execute> tags.
    """

    def __init__(self, llm_client: Dict[str, Any], tools: List[BaseTool], max_new_tokens: int = 512, stop_on_eos: bool = True):
        """
        Args:
            llm_client: A dictionary containing the 'model' and 'tokenizer'.
            tools: A list of tools available to the agent.
            max_new_tokens: The maximum number of tokens to generate.
            stop_on_eos: Whether to stop generation on the EOS token.
        """
        self.model = llm_client["model"]
        self.tokenizer = llm_client["tokenizer"]
        self.tools = tools
        self.max_new_tokens = max_new_tokens
        self.stop_on_eos = stop_on_eos

    def act(self, state: State) -> Action:
        """
        The core logic of the agent.
        """
        prompt = self._create_prompt(state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # --- Configure generation parameters ---
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
        }
        if self.stop_on_eos and self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            # Suppress warning by setting pad_token_id
            gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # The model's output includes the prompt, so we need to remove it.
        response_text = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        return self._parse_output(response_text.strip())

    def _create_prompt(self, state: State) -> str:
        """
        Creates the full conversational prompt, dynamically including the
        number of remaining chances.
        """
        # 1. Create the dynamic system prompt for this turn
        system_prompt = (
            "You are a helpful assistant assigned with the task of problem-solving. "
            "To achieve this, you will be using an interactive coding environment "
            "equipped with a variety of tool functions to assist you throughout the process.\n\n"
            "At each turn, you should first provide your step-by-step thinking for solving the task. "
            "After that, you have two options:\n\n"
            "1) Interact with a Python programming environment and receive the corresponding output. "
            'Your code should be enclosed using "<execute>" tag, for example: <execute> print(\"Hello World!\") </execute>.\n' 
            "2) Directly provide a solution that adheres to the required format for the given task. "
            'Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>.\n\n' 
            f"You have {state.interactions_left} chances to interact with the environment or propose a solution. "
            f"You can only propose a solution {state.solutions_left} times."
        )
        
        prompt = f"### System: {system_prompt}\n"
        
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
            # We also include the remaining chances in the observation for the model's context
            obs_text = (
                f"Observation:\n{observation}\n"
                f"You have {state.interactions_left - len(state.history)} steps left and "
                f"{state.solutions_left - (1 if action.tool_name == 'finish' else 0)} chances to propose solution left."
            )
            prompt += f"### User: \n{obs_text}\n"

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

