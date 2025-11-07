#!/bin/bash

# Deployment Script for Options Trading Bot Frontend
# ==================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_MODE=${1:-development}
COMPOSE_FILE="docker-compose.yml"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Options Trading Bot Frontend Deployment${NC}"
echo -e "${BLUE}   Mode: ${YELLOW}${DEPLOYMENT_MODE}${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠ .env file not found. Creating from template...${NC}"
    if [ -f .env.template ]; then
        cp .env.template .env
        echo -e "${RED}✗ Please edit the .env file with your actual values before continuing.${NC}"
        exit 1
    else
        echo -e "${RED}✗ .env.template file not found!${NC}"
        exit 1
    fi
fi

# Create required directories
echo -e "${GREEN}Creating required directories...${NC}"
mkdir -p logs data nginx/ssl monitoring/grafana/dashboards monitoring/grafana/provisioning

# Create external network if it doesn't exist
echo -e "${GREEN}Setting up Docker network...${NC}"
docker network create trading_network 2>/dev/null || echo "Network already exists"

# Function to deploy based on mode
deploy() {
    local mode=$1

    case $mode in
        development)
            echo -e "${GREEN}Starting development environment...${NC}"
            docker-compose up -d dashboard redis
            ;;

        staging)
            echo -e "${GREEN}Starting staging environment...${NC}"
            docker-compose up -d dashboard redis
            ;;

        production)
            echo -e "${GREEN}Starting production environment...${NC}"
            # Check for SSL certificates
            if [ ! -f nginx/ssl/cert.pem ] || [ ! -f nginx/ssl/key.pem ]; then
                echo -e "${YELLOW}⚠ SSL certificates not found in nginx/ssl/${NC}"
                echo -e "${YELLOW}  Please add cert.pem and key.pem files${NC}"
                read -p "Continue without HTTPS? (y/n): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            fi
            docker-compose --profile production up -d
            ;;

        monitoring)
            echo -e "${GREEN}Starting with monitoring stack...${NC}"
            docker-compose --profile monitoring up -d
            ;;

        full)
            echo -e "${GREEN}Starting full stack with all services...${NC}"
            docker-compose --profile production --profile monitoring up -d
            ;;

        *)
            echo -e "${RED}✗ Invalid deployment mode: $mode${NC}"
            echo "Usage: $0 [development|staging|production|monitoring|full]"
            exit 1
            ;;
    esac
}

# Build images
echo -e "${GREEN}Building Docker images...${NC}"
docker-compose build

# Deploy based on mode
deploy $DEPLOYMENT_MODE

# Wait for services to be healthy
echo -e "${GREEN}Waiting for services to be healthy...${NC}"
sleep 5

# Check service status
echo -e "${GREEN}Checking service status...${NC}"
docker-compose ps

# Display access information
echo
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   Deployment Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo
echo -e "${BLUE}Access points:${NC}"
echo -e "  Dashboard:   ${GREEN}http://localhost:8501${NC}"

if [[ $DEPLOYMENT_MODE == "production" ]] || [[ $DEPLOYMENT_MODE == "full" ]]; then
    echo -e "  HTTPS:       ${GREEN}https://localhost${NC}"
fi

if [[ $DEPLOYMENT_MODE == "monitoring" ]] || [[ $DEPLOYMENT_MODE == "full" ]]; then
    echo -e "  Prometheus:  ${GREEN}http://localhost:9090${NC}"
    echo -e "  Grafana:     ${GREEN}http://localhost:3000${NC}"
fi

echo -e "  Redis:       ${GREEN}localhost:6379${NC}"
echo
echo -e "${BLUE}Useful commands:${NC}"
echo -e "  View logs:       ${YELLOW}docker-compose logs -f dashboard${NC}"
echo -e "  Stop services:   ${YELLOW}docker-compose down${NC}"
echo -e "  Restart:         ${YELLOW}docker-compose restart dashboard${NC}"
echo -e "  Shell access:    ${YELLOW}docker exec -it options_trading_dashboard bash${NC}"
echo

# Health check
echo -e "${GREEN}Performing health check...${NC}"
sleep 3
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Dashboard is healthy and running!${NC}"
else
    echo -e "${YELLOW}⚠ Dashboard health check failed. Check logs for details.${NC}"
    echo -e "  Run: ${YELLOW}docker-compose logs dashboard${NC}"
fi