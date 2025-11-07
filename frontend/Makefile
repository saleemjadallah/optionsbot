# Makefile for Options Trading Bot Frontend
# =========================================

.PHONY: help build up down restart logs shell clean dev prod monitoring full

# Default target
help:
	@echo "Options Trading Bot Frontend - Make Commands"
	@echo "==========================================="
	@echo ""
	@echo "Development:"
	@echo "  make dev        - Start development environment"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo ""
	@echo "Production:"
	@echo "  make prod       - Deploy production environment"
	@echo "  make monitoring - Deploy with monitoring stack"
	@echo "  make full       - Deploy full stack (prod + monitoring)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make logs       - View dashboard logs"
	@echo "  make shell      - Access dashboard shell"
	@echo "  make clean      - Clean up containers and volumes"
	@echo "  make status     - Show service status"
	@echo ""
	@echo "Testing:"
	@echo "  make test       - Run tests"
	@echo "  make health     - Check service health"

# Build Docker images
build:
	docker-compose build

# Development environment
dev:
	./deploy.sh development

# Production environment
prod:
	./deploy.sh production

# Monitoring stack
monitoring:
	./deploy.sh monitoring

# Full stack deployment
full:
	./deploy.sh full

# Start services
up:
	docker-compose up -d

# Stop services
down:
	docker-compose down

# Restart services
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f dashboard

# Access shell
shell:
	docker exec -it options_trading_dashboard bash

# Clean up
clean:
	docker-compose down -v
	rm -rf logs/* data/*
	@echo "Cleaned up containers, volumes, and temporary files"

# Show status
status:
	docker-compose ps

# Health check
health:
	@echo "Checking service health..."
	@curl -s -f http://localhost:8501/_stcore/health > /dev/null && \
		echo "✓ Dashboard is healthy" || \
		echo "✗ Dashboard health check failed"
	@docker exec options_trading_redis redis-cli ping > /dev/null 2>&1 && \
		echo "✓ Redis is healthy" || \
		echo "✗ Redis health check failed"

# Run tests
test:
	@echo "Running tests..."
	@if [ -d "tests" ]; then \
		docker-compose run --rm dashboard pytest tests/; \
	else \
		echo "No tests directory found"; \
	fi

# Update dependencies
update-deps:
	docker-compose run --rm dashboard pip list --outdated
	@echo "Run 'make update-deps-force' to update all dependencies"

# Force update dependencies
update-deps-force:
	docker-compose run --rm dashboard pip install --upgrade -r requirements.txt

# Backup data
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/backup-$$(date +%Y%m%d-%H%M%S).tar.gz logs/ data/
	@echo "Backup created in backups/"

# Monitor resource usage
monitor:
	docker stats --no-stream

# Network inspection
network:
	docker network inspect trading_network

# Volume inspection
volumes:
	docker volume ls | grep frontend
	@echo ""
	@echo "Volume details:"
	@docker volume inspect frontend_dashboard_cache || true
	@docker volume inspect frontend_redis_data || true