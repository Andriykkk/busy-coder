# This file defines the Docker environment for sandboxed code execution.

# Base Image:
# - Start from a data science image like `jupyter/datascience-notebook`
#   as it comes pre-installed with pandas, numpy, matplotlib, etc.

# Dependencies:
# - Add any other specific Python libraries the agent might need to install via pip.
#   (e.g., `pip install scikit-learn beautifulsoup4`)

# Entrypoint:
# - The command to start the Jupyter Kernel Gateway.
# - This will expose an endpoint that can receive and execute code.
#   CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888"]
