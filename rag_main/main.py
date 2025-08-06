import argparse
import json
import logging
from typing import List, Dict, Any

# Configure basic logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("CodeActAgent")


def interactive_loop(agent: Any, env: Any, task: str):
    """
    The core interaction loop for the agent.
    This function will orchestrate the conversation between the agent and the environment.
    """
    LOGGER.info(f"Starting new task: {task}")

    # Reset the environment with the new task
    # state = env.reset(task)
    # LOGGER.info(f"Initial observation:\n\033[94m{state.observation}\033[0m")

    # Loop until the state is marked as finished
    # while not state.finished:
    #     # 1. Get action from the agent
    #     action = agent.act(state)
    #     LOGGER.info(f"Agent action:\n\033[92m{action}\033[0m")

    #     # 2. Execute the action in the environment
    #     state = env.step(action)
    #     LOGGER.info(f"Observation:\n\033[94m{state.observation}\033[0m")

    # LOGGER.info(f"Task finished. Final state: {state.summary}")
    # return state
    pass # Placeholder until we implement the other components


def main():
    """
    Main entry point for the application.
    Parses arguments, loads configuration, initializes components, and starts the loop.
    """
    parser = argparse.ArgumentParser(description="Run the CodeAct RAG agent.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration JSON file.",
    )
    args = parser.parse_args()

    LOGGER.info(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            exp_config: Dict[str, Any] = json.load(f)
    except FileNotFoundError:
        LOGGER.error(f"Configuration file not found at: {args.config}")
        return
    except json.JSONDecodeError:
        LOGGER.error(f"Invalid JSON in configuration file: {args.config}")
        return

    # --- Component Initialization (will be implemented later) ---
    LOGGER.info("Initializing components...")

    # 1. Initialize Agent (e.g., ReactAgent)
    # agent_config = exp_config.get("agent", {})
    # agent = initialize_agent(agent_config)
    agent = None # Placeholder

    # 2. Initialize Tools (e.g., CodeExecutor, WebSearch)
    # tool_configs = exp_config.get("tools", [])
    # tools = [initialize_tool(tc) for tc in tool_configs]
    tools = [] # Placeholder

    # 3. Initialize Environment
    # env_config = exp_config.get("environment", {})
    # env = SandboxEnvironment(tools, **env_config)
    env = None # Placeholder

    LOGGER.info("Initialization complete.")

    # --- Start Interactive Session ---
    try:
        while True:
            task = input("\033[95m\nEnter your task (or 'exit' to quit):\n> \033[0m")
            if task.lower() == 'exit':
                LOGGER.info("Exiting.")
                break
            interactive_loop(agent, env, task)
    except KeyboardInterrupt:
        LOGGER.info("\nInterrupted by user. Exiting.")


if __name__ == "__main__":
    main()