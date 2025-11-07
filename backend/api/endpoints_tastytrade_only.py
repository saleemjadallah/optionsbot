"""
Lightweight FastAPI Backend - Tastytrade Only

This is a simplified version of the backend that only includes
Tastytrade integration endpoints. Use this if you just want to
test Tastytrade authentication without all the trading bot dependencies.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.tastytrade_endpoints import router as tastytrade_router

# Create app
app = FastAPI(title="Tastytrade API")

# Include Tastytrade router
app.include_router(tastytrade_router)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://dashboard:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Tastytrade API Server",
        "docs": "/docs",
        "health": "/api/tastytrade/health"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tastytrade-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
