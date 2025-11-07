#!/bin/bash

# Options Trading Bot Startup Script
# ==================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Options Trading Bot System Startup${NC}"
echo -e "${GREEN}=====================================${NC}\n"

# Function to check if a service is running
check_service() {
    local service=$1
    local port=$2
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $service is running on port $port"
        return 0
    else
        echo -e "${RED}✗${NC} $service is not running on port $port"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0

    echo -e "${YELLOW}Waiting for $service to start...${NC}"
    while [ $attempt -lt $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $service is ready!"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    echo -e "${RED}✗${NC} $service failed to start"
    return 1
}

# Parse command line arguments
MODE=${1:-local}  # local or docker

if [ "$MODE" == "docker" ]; then
    echo -e "${YELLOW}Starting in Docker mode...${NC}\n"

    # Check if .env file exists
    if [ ! -f .env ]; then
        echo -e "${YELLOW}Creating .env file from template...${NC}"
        cp .env.example .env
        echo -e "${RED}Please update .env file with your API credentials${NC}"
        exit 1
    fi

    # Stop existing containers
    echo -e "${YELLOW}Stopping existing containers...${NC}"
    docker-compose down

    # Start services
    echo -e "${YELLOW}Starting Docker services...${NC}"
    docker-compose up -d

    # Wait for services
    wait_for_service "PostgreSQL" 5432
    wait_for_service "Redis" 6379
    wait_for_service "Backend API" 8000
    wait_for_service "Frontend Dashboard" 8501

    echo -e "\n${GREEN}=====================================${NC}"
    echo -e "${GREEN}All services started successfully!${NC}"
    echo -e "${GREEN}=====================================${NC}\n"
    echo -e "Dashboard: ${GREEN}http://localhost:8501${NC}"
    echo -e "API Docs: ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "\nTo view logs: ${YELLOW}docker-compose logs -f${NC}"
    echo -e "To stop: ${YELLOW}docker-compose down${NC}"

elif [ "$MODE" == "local" ]; then
    echo -e "${YELLOW}Starting in Local Development mode...${NC}\n"

    # Check Python version
    python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo -e "Python version: ${GREEN}$python_version${NC}"

    # Check if virtual environment exists
    if [ ! -d "options_trader_env" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv options_trader_env
    fi

    # Activate virtual environment
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source options_trader_env/bin/activate

    # Install requirements
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -q -r requirements.txt
    pip install -q fastapi uvicorn[standard] websockets streamlit streamlit-option-menu

    # Start Redis if not running
    if ! check_service "Redis" 6379; then
        echo -e "${YELLOW}Starting Redis...${NC}"
        if command -v redis-server &> /dev/null; then
            redis-server --daemonize yes
            wait_for_service "Redis" 6379
        else
            echo -e "${RED}Redis is not installed. Please install Redis or use Docker mode.${NC}"
            exit 1
        fi
    fi

    # Start PostgreSQL if not running
    if ! check_service "PostgreSQL" 5432; then
        echo -e "${YELLOW}PostgreSQL is not running.${NC}"
        echo -e "${YELLOW}Please start PostgreSQL or use Docker mode.${NC}"
        echo -e "${YELLOW}Continuing with mock data mode...${NC}\n"
    fi

    # Start Backend API
    echo -e "${YELLOW}Starting Backend API...${NC}"
    cd backend
    uvicorn api.endpoints:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    cd ..

    # Wait for backend to start
    wait_for_service "Backend API" 8000

    # Start Frontend Dashboard
    echo -e "${YELLOW}Starting Frontend Dashboard...${NC}"
    cd frontend
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
    FRONTEND_PID=$!
    cd ..

    # Wait for frontend to start
    wait_for_service "Frontend Dashboard" 8501

    echo -e "\n${GREEN}=====================================${NC}"
    echo -e "${GREEN}All services started successfully!${NC}"
    echo -e "${GREEN}=====================================${NC}\n"
    echo -e "Dashboard: ${GREEN}http://localhost:8501${NC}"
    echo -e "API Docs: ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "\nProcess IDs:"
    echo -e "  Backend: $BACKEND_PID"
    echo -e "  Frontend: $FRONTEND_PID"
    echo -e "\nTo stop: Press ${YELLOW}Ctrl+C${NC}"

    # Wait for user interrupt
    trap "echo -e '\n${YELLOW}Shutting down services...${NC}'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
    wait

elif [ "$MODE" == "test" ]; then
    echo -e "${YELLOW}Running Integration Tests...${NC}\n"

    # Check if backend is running
    if ! check_service "Backend API" 8000; then
        echo -e "${RED}Backend API is not running!${NC}"
        echo -e "Please start the backend first with: ${YELLOW}./start_trading_system.sh local${NC}"
        exit 1
    fi

    # Run tests
    python3 tests/test_api_integration.py

else
    echo -e "${RED}Invalid mode: $MODE${NC}"
    echo -e "Usage: $0 [local|docker|test]"
    echo -e "  local  - Run services locally (default)"
    echo -e "  docker - Run services in Docker"
    echo -e "  test   - Run integration tests"
    exit 1
fi