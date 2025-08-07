import argparse
import json
import logging
from typing import Any, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.agent.react_agent import ReactAgent
from src.datamodels.state import State
from src.environment.sandbox import Sandbox
from src.tools.code_executor import CodeExecutorTool
from src.tools.web_search import WebSearchTool

# Configure basic logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("CodeActAgent")


def interactive_loop(agent: ReactAgent, env: Sandbox, task: str):
    """
    The core interaction loop for the agent.
    """
    LOGGER.info(f"Starting new task: {task}")
    state = env.reset(task)
    LOGGER.info(f"Initial observation:\n\033[94m{state.observation}\033[0m")

    while not state.finished:
        action = agent.act(state)
        LOGGER.info(f"Agent action:\n\033[92m{action}\033[0m")
        state = env.step(state, action)
        LOGGER.info(f"Observation:\n\033[94m{state.observation}\033[0m")

    LOGGER.info(f"Task finished. Final state: {state.summary}")
    return state


def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(description="Run the CodeAct RAG agent.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to the experiment configuration JSON file.",
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            exp_config: Dict[str, Any] = json.load(f)
    except FileNotFoundError:
        LOGGER.error(f"Configuration file not found at: {args.config}")
        return

    # --- Component Initialization ---
    LOGGER.info("Initializing components...")

    # 1. Load Model and Tokenizer
    model_path = exp_config.get("model_path")
    if not model_path:
        LOGGER.error("`model_path` not found in config. Please specify the path to the merged model.")
        return

    LOGGER.info(f"Loading model from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    except Exception as e:
        LOGGER.error(f"Failed to load model or tokenizer: {e}")
        return
    
    LOGGER.info("Model and tokenizer loaded successfully.")

    # 2. Initialize Tools
    tools = [
        CodeExecutorTool(sandbox_url=exp_config.get("sandbox_url", "http://localhost:8888")),
        WebSearchTool(),
    ]

    # 3. Initialize Environment and Agent
    env = Sandbox(tools)
    agent = ReactAgent(llm_client={"model": model, "tokenizer": tokenizer}, tools=tools)

    LOGGER.info("Initialization complete.")

    # --- Start Interactive Session ---
    try:
        while True:
            task = input("\033[95m\nEnter your task (or 'exit' to quit):\n> \033[0m")
            if task.lower() == 'exit':
                break
            interactive_loop(agent, env, task)
    except KeyboardInterrupt:
        LOGGER.info("\nInterrupted by user. Exiting.")


if __name__ == "__main__":
    main()
