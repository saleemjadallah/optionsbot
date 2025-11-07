"""
Entry point for Railway deployment.
Starts the FastAPI ensemble service.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import and run the service
from service.main import run

if __name__ == "__main__":
    run()
