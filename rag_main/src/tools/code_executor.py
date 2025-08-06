# This file will implement the tool for executing Python code.

# Responsibilities:
# 1. Tool Definition:
#    - Inherit from `BaseTool`.
#    - Define the `name`, `signature`, and `description`.

# 2. Sandbox Communication:
#    - The `run` method will take a string of Python code.
#    - It will be responsible for sending this code to the sandboxed environment
#      (e.g., by making an HTTP request to the Jupyter Kernel Gateway running in Docker).
#    - It should wait for the execution to complete.

# 3. Result Handling:
#    - It will receive the `stdout`, `stderr`, or any plot/image output from the sandbox.
#    - It will format this result into a single observation string to be returned to the agent.
#    - For example: "Execution Result:\n--- stdout ---\n... \n--- stderr ---\n..."