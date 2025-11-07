#!/bin/bash

# Options Trading System - Complete Startup Script
# ================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=8501
BACKEND_HOST="0.0.0.0"
FRONTEND_HOST="0.0.0.0"
VENV_NAME="options_trader_env"
PROJECT_ROOT=$(pwd)

# Process IDs
BACKEND_PID=""
FRONTEND_PID=""

# ASCII Art Banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════╗"
    echo "║     OPTIONS TRADING SYSTEM - STARTUP SCRIPT    ║"
    echo "╚════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Logging functions
log_info() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ℹ️  ${NC}$1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️  ${NC}$1"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ❌ ${NC}$1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ✅ ${NC}$1"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    python_major=$(python3 -c 'import sys; print(sys.version_info.major)')
    python_minor=$(python3 -c 'import sys; print(sys.version_info.minor)')

    if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
        log_error "Python 3.8+ is required (found $python_version)"
        exit 1
    fi
    log_success "Python $python_version detected"

    # Check for required commands
    local required_commands=("curl" "nc")
    for cmd in "${required_commands[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            log_warning "$cmd is not installed (optional but recommended)"
        fi
    done
}

# Setup virtual environment
setup_venv() {
    if [ ! -d "$VENV_NAME" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv $VENV_NAME
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi

    log_info "Activating virtual environment..."
    source $VENV_NAME/bin/activate

    # Upgrade pip
    pip install --quiet --upgrade pip
}

# Install dependencies
install_dependencies() {
    log_info "Installing/updating dependencies..."

    # Use fixed requirements if it exists, otherwise use regular requirements
    if [ -f "requirements-fixed.txt" ]; then
        log_info "Using requirements-fixed.txt..."
        pip install --quiet -r requirements-fixed.txt
        log_success "Core dependencies installed from fixed requirements"
    elif [ -f "requirements.txt" ]; then
        log_warning "Attempting to install from requirements.txt (may have compatibility issues)..."
        pip install --quiet -r requirements.txt 2>/dev/null || {
            log_warning "Some packages failed to install, continuing with essential packages..."
        }
    fi

    # Install essential packages separately to ensure they're available
    log_info "Installing essential server packages..."
    pip install --quiet --upgrade pip
    pip install --quiet fastapi uvicorn[standard] websockets streamlit streamlit-option-menu 2>/dev/null || {
        log_warning "Some packages already installed or had issues, continuing..."
    }

    log_success "Essential dependencies installed"
}

# Check if port is available
check_port() {
    local port=$1
    local service=$2

    if nc -z localhost $port 2>/dev/null; then
        log_warning "Port $port is already in use (for $service)"

        # Find process using the port
        if command -v lsof &> /dev/null; then
            local pid=$(lsof -ti:$port)
            if [ ! -z "$pid" ]; then
                log_info "Process $pid is using port $port"
                read -p "Kill this process? (y/n): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    kill -9 $pid 2>/dev/null || true
                    sleep 2
                    log_success "Process killed"
                else
                    log_error "Cannot start $service on port $port"
                    return 1
                fi
            fi
        fi
    else
        log_success "Port $port is available for $service"
    fi
    return 0
}

# Wait for service to be ready
wait_for_service() {
    local service=$1
    local port=$2
    local url=$3
    local max_attempts=30
    local attempt=1

    log_info "Waiting for $service to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            # Additional health check if URL provided
            if [ ! -z "$url" ]; then
                if curl -s -f "$url" > /dev/null 2>&1; then
                    log_success "$service is ready and responding!"
                    return 0
                fi
            else
                log_success "$service is ready on port $port!"
                return 0
            fi
        fi

        printf "."
        sleep 2
        ((attempt++))
    done

    echo
    log_error "$service failed to start after $max_attempts attempts"
    return 1
}

# Start Backend API
start_backend() {
    log_info "Starting Backend API Server..."

    # Check if backend directory exists
    if [ ! -d "backend" ]; then
        log_error "Backend directory not found"
        return 1
    fi

    # Check port availability
    if ! check_port $BACKEND_PORT "Backend API"; then
        return 1
    fi

    # Start the backend server (use simple endpoints for now)
    cd backend
    nohup python3 -m uvicorn api.endpoints_simple:app \
        --host $BACKEND_HOST \
        --port $BACKEND_PORT \
        --reload \
        > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    cd ..

    log_info "Backend API starting with PID: $BACKEND_PID"

    # Wait for backend to be ready
    if wait_for_service "Backend API" $BACKEND_PORT "http://localhost:$BACKEND_PORT/docs"; then
        log_success "Backend API is running at http://localhost:$BACKEND_PORT"
        log_info "API Documentation: http://localhost:$BACKEND_PORT/docs"
        return 0
    else
        log_error "Backend API failed to start"
        return 1
    fi
}

# Start Frontend Dashboard
start_frontend() {
    log_info "Starting Frontend Dashboard..."

    # Check if frontend directory exists
    if [ ! -d "frontend" ]; then
        log_error "Frontend directory not found"
        return 1
    fi

    # Check port availability
    if ! check_port $FRONTEND_PORT "Frontend Dashboard"; then
        return 1
    fi

    # Check which app file exists
    local app_file=""
    if [ -f "frontend/app.py" ]; then
        app_file="app.py"
    elif [ -f "frontend/multipage_app.py" ]; then
        app_file="multipage_app.py"
    else
        log_error "No Streamlit app file found in frontend directory"
        return 1
    fi

    # Start the frontend server
    cd frontend
    nohup streamlit run $app_file \
        --server.port=$FRONTEND_PORT \
        --server.address=$FRONTEND_HOST \
        --server.headless=true \
        > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..

    log_info "Frontend Dashboard starting with PID: $FRONTEND_PID"

    # Wait for frontend to be ready
    if wait_for_service "Frontend Dashboard" $FRONTEND_PORT "http://localhost:$FRONTEND_PORT/_stcore/health"; then
        log_success "Frontend Dashboard is running at http://localhost:$FRONTEND_PORT"
        return 0
    else
        log_error "Frontend Dashboard failed to start"
        return 1
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    mkdir -p logs data .streamlit
    log_success "Directories created"
}

# Show system status
show_status() {
    echo
    echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║            SYSTEM STATUS SUMMARY               ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"
    echo

    # Check Backend
    if nc -z localhost $BACKEND_PORT 2>/dev/null; then
        echo -e "${GREEN}✅ Backend API:${NC}        http://localhost:$BACKEND_PORT"
        echo -e "${GREEN}   API Docs:${NC}          http://localhost:$BACKEND_PORT/docs"
    else
        echo -e "${RED}❌ Backend API:${NC}        Not Running"
    fi

    # Check Frontend
    if nc -z localhost $FRONTEND_PORT 2>/dev/null; then
        echo -e "${GREEN}✅ Frontend Dashboard:${NC} http://localhost:$FRONTEND_PORT"
    else
        echo -e "${RED}❌ Frontend Dashboard:${NC} Not Running"
    fi

    echo
    if [ ! -z "$BACKEND_PID" ]; then
        echo -e "${BLUE}Process IDs:${NC}"
        echo -e "  Backend PID:  $BACKEND_PID"
        echo -e "  Frontend PID: $FRONTEND_PID"
    fi
    echo
}

# Monitor logs
monitor_logs() {
    log_info "Monitoring logs (Press Ctrl+C to stop monitoring)..."
    echo
    echo -e "${YELLOW}Backend Logs:${NC}"
    tail -f logs/backend.log &
    local BACKEND_TAIL_PID=$!

    echo
    echo -e "${YELLOW}Frontend Logs:${NC}"
    tail -f logs/frontend.log &
    local FRONTEND_TAIL_PID=$!

    # Wait for user interrupt
    wait

    # Clean up tail processes
    kill $BACKEND_TAIL_PID $FRONTEND_TAIL_PID 2>/dev/null || true
}

# Graceful shutdown
shutdown_services() {
    echo
    log_warning "Shutting down services..."

    if [ ! -z "$BACKEND_PID" ]; then
        log_info "Stopping Backend API (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
    fi

    if [ ! -z "$FRONTEND_PID" ]; then
        log_info "Stopping Frontend Dashboard (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi

    # Kill any remaining processes on the ports
    if command -v lsof &> /dev/null; then
        lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
        lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    fi

    log_success "All services stopped"
    exit 0
}

# Stop all services
stop_all() {
    log_warning "Stopping all trading system services..."

    # Find and kill backend processes
    pkill -f "uvicorn.*api.endpoints:app" 2>/dev/null || true

    # Find and kill frontend processes
    pkill -f "streamlit run" 2>/dev/null || true

    # Kill processes on specific ports
    if command -v lsof &> /dev/null; then
        lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
        lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    fi

    log_success "All services stopped"
}

# Display help
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  start       Start all services (default)"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show system status"
    echo "  logs        Monitor system logs"
    echo "  help        Show this help message"
    echo
    echo "Options:"
    echo "  --backend-only    Start only the backend API"
    echo "  --frontend-only   Start only the frontend dashboard"
    echo "  --no-install      Skip dependency installation"
    echo
    echo "Examples:"
    echo "  $0                    # Start all services"
    echo "  $0 start              # Start all services"
    echo "  $0 stop               # Stop all services"
    echo "  $0 restart            # Restart all services"
    echo "  $0 status             # Check service status"
    echo "  $0 logs               # Monitor logs"
    echo "  $0 --backend-only     # Start only backend"
}

# Parse command line arguments
parse_arguments() {
    COMMAND=${1:-start}
    BACKEND_ONLY=false
    FRONTEND_ONLY=false
    NO_INSTALL=false

    # Parse remaining arguments
    shift || true
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backend-only)
                BACKEND_ONLY=true
                shift
                ;;
            --frontend-only)
                FRONTEND_ONLY=true
                shift
                ;;
            --no-install)
                NO_INSTALL=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Main execution
main() {
    # Parse arguments
    parse_arguments "$@"

    # Handle commands
    case $COMMAND in
        start)
            # Print banner
            print_banner

            # Setup trap for cleanup
            trap shutdown_services SIGINT SIGTERM

            # Check requirements
            check_requirements

            # Create directories
            create_directories

            # Setup virtual environment
            setup_venv

            # Install dependencies (unless skipped)
            if [ "$NO_INSTALL" = false ]; then
                install_dependencies
            fi

            # Start services based on options
            if [ "$FRONTEND_ONLY" = true ]; then
                start_frontend
            elif [ "$BACKEND_ONLY" = true ]; then
                start_backend
            else
                # Start both services
                if start_backend; then
                    if start_frontend; then
                        log_success "All services started successfully!"
                    else
                        log_error "Frontend failed to start"
                        shutdown_services
                    fi
                else
                    log_error "Backend failed to start"
                    exit 1
                fi
            fi

            # Show status
            show_status

            # Keep script running
            log_info "Press Ctrl+C to stop all services"
            echo

            # Wait indefinitely
            while true; do
                sleep 1
            done
            ;;

        stop)
            stop_all
            ;;

        restart)
            log_info "Restarting all services..."
            stop_all
            sleep 3
            exec "$0" start "${@:2}"
            ;;

        status)
            show_status
            ;;

        logs)
            if [ ! -f "logs/backend.log" ] || [ ! -f "logs/frontend.log" ]; then
                log_error "Log files not found. Are the services running?"
                exit 1
            fi
            monitor_logs
            ;;

        help|--help|-h)
            show_help
            ;;

        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"