# Use a slim Python image for the main application
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The command to run the app will be provided by docker-compose
# This Dockerfile just sets up the environment.
COPY . .
