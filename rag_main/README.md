# CodeAct RAG Project

This project implements a multi-turn agent that can reason and act to solve complex tasks by interacting with a sandboxed environment.

## Project Structure

- `main.py`: The main entry point to run the agent.
- `src/`: Contains the core source code.
  - `agent/`: The agent's reasoning logic (ReAct).
  - `datamodels/`: Defines the `State` and `Action` data structures.
  - `environment/`: The environment that manages tools and state.
  - `tools/`: The tools the agent can use (code execution, web search).
- `Dockerfile`: Defines the sandboxed environment for code execution.
- `docker-compose.yml`: Orchestrates the application and sandbox services.
- `configs/`: Contains configuration files for experiments.
- `requirements.txt`: Project dependencies.

## How to Run

1.  Set up your environment variables (e.g., `OPENAI_API_KEY`).
2.  Build and run the services: `docker-compose up --build`
3.  Interact with the agent through the command line.
