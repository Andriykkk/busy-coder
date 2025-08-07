# Use a pre-built data science image as the base
# This includes Python, Jupyter, and common libraries like pandas, numpy, scikit-learn, etc.
FROM jupyter/datascience-notebook:latest

# Add any additional libraries that the agent might need to use inside the sandbox
# For example, for web scraping or other specific tasks.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Add any system-level dependencies here if needed
    && rm -rf /var/lib/apt/lists/*
USER ${NB_UID}

RUN pip install --no-cache-dir \
    beautifulsoup4 \
    requests

# Expose the default port for the Jupyter Kernel Gateway
EXPOSE 8888

# The command to start the kernel gateway when the container runs.
# This makes the container listen for code execution requests.
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888"]