#!/bin/bash

# Vector DB Backend - Docker Run Script

# Build the Docker image
echo "Building Docker image..."
docker build -t vector-db-backend .

# Run the container
echo "Starting container on port 8000..."
docker run -p 8000:8000 --name vector-db-backend-container --rm vector-db-backend

