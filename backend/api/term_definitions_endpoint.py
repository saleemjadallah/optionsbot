"""
Trading Term Definitions API Endpoint
======================================

FastAPI endpoint for generating AI-powered trading term definitions
using OpenAI's GPT model.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import os
from openai import OpenAI
from datetime import datetime, timedelta
import json
from pathlib import Path

router = APIRouter(prefix="/api", tags=["term-definitions"])

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables")
    client = None
else:
    client = OpenAI(api_key=openai_api_key)


class TermDefinitionResponse(BaseModel):
    """Response model for term definitions"""
    term: str
    definition: str
    generated_at: str
    cached: bool


class DefinitionCache:
    """Simple in-memory cache for term definitions"""

    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.cache_duration = timedelta(days=30)  # Cache for 30 days
        self._load_persistent_cache()

    def _get_cache_file(self) -> Path:
        """Get path to persistent cache file"""
        return Path(__file__).parent.parent / "data" / "term_definitions_cache.json"

    def _load_persistent_cache(self):
        """Load cache from disk"""
        cache_file = self._get_cache_file()
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Convert timestamp strings back to datetime
                    for key, value in data.items():
                        value['timestamp'] = datetime.fromisoformat(value['timestamp'])
                    self.cache = data
            except Exception as e:
                print(f"Error loading cache: {e}")

    def _save_persistent_cache(self):
        """Save cache to disk"""
        cache_file = self._get_cache_file()
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Convert datetime to string for JSON serialization
            serializable_cache = {}
            for key, value in self.cache.items():
                serializable_cache[key] = {
                    'definition': value['definition'],
                    'timestamp': value['timestamp'].isoformat()
                }

            with open(cache_file, 'w') as f:
                json.dump(serializable_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def get(self, term: str) -> Optional[str]:
        """Get cached definition if available and not expired"""
        key = term.lower().strip()
        if key in self.cache:
            cached_item = self.cache[key]
            if datetime.now() - cached_item['timestamp'] < self.cache_duration:
                return cached_item['definition']
            else:
                # Expired, remove from cache
                del self.cache[key]
        return None

    def set(self, term: str, definition: str):
        """Cache a definition"""
        key = term.lower().strip()
        self.cache[key] = {
            'definition': definition,
            'timestamp': datetime.now()
        }
        self._save_persistent_cache()


# Global cache instance
definition_cache = DefinitionCache()


def generate_definition_with_openai(term: str) -> str:
    """
    Generate a trading term definition using OpenAI GPT.

    Args:
        term: The trading term to define

    Returns:
        AI-generated definition string
    """
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured"
        )

    try:
        # Create a specific prompt for trading terminology
        prompt = f"""You are a financial markets expert specializing in options trading.
Provide a clear, concise definition of the trading term: "{term}"

Requirements:
- Keep it under 100 words
- Focus on practical understanding for options traders
- Include what it measures or represents
- Mention why it's important or how it's used
- Use simple, direct language
- Avoid overly technical jargon unless necessary

Definition:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using cost-effective model
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in options trading and financial markets. Provide clear, concise definitions of trading terminology."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.3,  # Lower temperature for more consistent definitions
        )

        definition = response.choices[0].message.content.strip()
        return definition

    except Exception as e:
        print(f"Error generating definition with OpenAI: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate definition: {str(e)}"
        )


@router.get("/term-definition/{term}", response_model=TermDefinitionResponse)
async def get_term_definition(term: str) -> TermDefinitionResponse:
    """
    Get definition for a trading term.

    Args:
        term: The trading term to define

    Returns:
        TermDefinitionResponse with definition and metadata
    """
    # Check cache first
    cached_definition = definition_cache.get(term)

    if cached_definition:
        return TermDefinitionResponse(
            term=term,
            definition=cached_definition,
            generated_at=datetime.now().isoformat(),
            cached=True
        )

    # Generate new definition
    try:
        definition = generate_definition_with_openai(term)

        # Cache the definition
        definition_cache.set(term, definition)

        return TermDefinitionResponse(
            term=term,
            definition=definition,
            generated_at=datetime.now().isoformat(),
            cached=False
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@router.delete("/term-definition/cache")
async def clear_definition_cache():
    """
    Clear the term definitions cache.

    Returns:
        Success message
    """
    definition_cache.cache.clear()
    definition_cache._save_persistent_cache()

    return {"message": "Cache cleared successfully"}


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "openai_configured": client is not None,
        "cache_size": len(definition_cache.cache)
    }
