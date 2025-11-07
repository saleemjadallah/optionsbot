#!/bin/bash

# Build and Run Options Trading Bot Frontend Docker Container
# ===========================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Options Trading Bot Frontend Docker Image...${NC}"

# Build the Docker image
docker build -t options-trader-frontend:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully!${NC}"
else
    echo -e "${RED}✗ Docker build failed!${NC}"
    exit 1
fi

# Option to run the container
read -p "Do you want to run the container now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Starting the container...${NC}"

    # Stop and remove existing container if it exists
    docker stop options-trader-frontend 2>/dev/null || true
    docker rm options-trader-frontend 2>/dev/null || true

    # Run the container
    docker run -d \
        --name options-trader-frontend \
        -p 8501:8501 \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/data:/app/data \
        --restart unless-stopped \
        options-trader-frontend:latest

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Container started successfully!${NC}"
        echo -e "${GREEN}Access the dashboard at: http://localhost:8501${NC}"
        echo
        echo "Useful commands:"
        echo "  View logs:        docker logs -f options-trader-frontend"
        echo "  Stop container:   docker stop options-trader-frontend"
        echo "  Start container:  docker start options-trader-frontend"
        echo "  Remove container: docker rm options-trader-frontend"
    else
        echo -e "${RED}✗ Failed to start container!${NC}"
        exit 1
    fi
fi