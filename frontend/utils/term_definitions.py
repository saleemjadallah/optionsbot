"""
Trading Term Definitions Utility
=================================

Manages trading terminology definitions with AI-powered generation and caching.
Uses OpenAI API to generate clear, concise definitions for trading terms.
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional, Dict
import hashlib


class TermDefinitionManager:
    """Manages trading term definitions with OpenAI generation and local caching"""

    def __init__(self, cache_file: str = "term_definitions_cache.json"):
        """
        Initialize the term definition manager.

        Args:
            cache_file: Path to the JSON cache file for storing definitions
        """
        self.cache_file = Path(__file__).parent / cache_file
        self.cache = self._load_cache()
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

    def _load_cache(self) -> Dict[str, str]:
        """Load cached definitions from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Save cached definitions to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def get_definition(self, term: str) -> str:
        """
        Get definition for a trading term.

        First checks cache, then requests from backend API if not found.

        Args:
            term: The trading term to define

        Returns:
            Definition string
        """
        # Normalize term for consistent caching
        term_key = term.lower().strip()

        # Check cache first
        if term_key in self.cache:
            return self.cache[term_key]

        # Request from backend API
        try:
            response = requests.get(
                f"{self.backend_url}/api/term-definition/{term}",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                definition = data.get('definition', 'Definition not available')

                # Cache the definition
                self.cache[term_key] = definition
                self._save_cache()

                return definition
        except Exception as e:
            print(f"Error fetching definition for '{term}': {e}")

        # Fallback to default definition if API fails
        return self._get_default_definition(term)

    def _get_default_definition(self, term: str) -> str:
        """
        Provide basic fallback definitions for common trading terms.

        Args:
            term: The trading term

        Returns:
            Default definition string
        """
        defaults = {
            'delta': 'Delta measures the rate of change of an option\'s price relative to changes in the underlying asset\'s price. It ranges from 0 to 1 for calls and -1 to 0 for puts.',
            'gamma': 'Gamma measures the rate of change of delta. It shows how much delta will change when the underlying price moves by $1.',
            'theta': 'Theta measures time decay - how much an option\'s value decreases as time passes, all else being equal. Usually expressed as dollars lost per day.',
            'vega': 'Vega measures sensitivity to volatility. It shows how much an option\'s price changes when implied volatility changes by 1%.',
            'rho': 'Rho measures sensitivity to interest rate changes. It shows how much an option\'s price changes when interest rates change by 1%.',
            'implied volatility': 'IV represents the market\'s expectation of future volatility. Higher IV means options are more expensive.',
            'var': 'Value at Risk (VaR) estimates the maximum potential loss over a given time period at a specific confidence level.',
            'sharpe ratio': 'Risk-adjusted return metric calculated as (return - risk-free rate) / standard deviation. Higher values indicate better risk-adjusted performance.',
            'max drawdown': 'The largest peak-to-trough decline in portfolio value. It measures the maximum loss from a peak before a new peak is achieved.',
            'win rate': 'The percentage of trades that are profitable. A 70% win rate means 7 out of 10 trades were winners.',
            'profit factor': 'Ratio of gross profits to gross losses. A profit factor of 2.0 means you make $2 for every $1 lost.',
            'portfolio beta': 'Measures the portfolio\'s sensitivity to market movements. A beta of 1 moves with the market, >1 is more volatile, <1 is less volatile.',
            'sortino ratio': 'Similar to Sharpe ratio but only considers downside volatility, providing a better measure of downside risk.',
            'calmar ratio': 'Return divided by maximum drawdown. Measures return per unit of downside risk.',
            'expected shortfall': 'Also called Conditional VaR - the expected loss given that VaR has been exceeded.',
            'strike price': 'The predetermined price at which an option can be exercised.',
            'premium': 'The price paid to purchase an option contract.',
            'open interest': 'The total number of outstanding option contracts that have not been closed or exercised.',
            'bid/ask spread': 'The difference between the highest price a buyer is willing to pay (bid) and the lowest price a seller will accept (ask).',
            'iron condor': 'A strategy involving four options: selling an OTM call spread and an OTM put spread, profiting from low volatility.',
            'straddle': 'Buying both a call and put at the same strike, profiting from large price movements in either direction.',
            'strangle': 'Similar to straddle but using different strikes for call and put, typically OTM.',
            'butterfly spread': 'A neutral strategy using three strikes, with limited risk and profit potential.',
            'call spread': 'Buying and selling calls at different strikes to limit both risk and potential profit.',
            'put spread': 'Buying and selling puts at different strikes to limit both risk and potential profit.',
            'calendar spread': 'Buying and selling options with same strike but different expiration dates.',
            'volatility skew': 'The pattern where OTM puts often have higher implied volatility than OTM calls.',
            'leverage': 'Using borrowed capital or derivatives to increase exposure beyond what cash alone would allow.',
            'concentration risk': 'Risk from having too much capital allocated to a single position or correlated positions.',
            'ulcer index': 'Measures downside volatility, specifically the depth and duration of drawdowns.',
            'omega ratio': 'Ratio of probability-weighted gains to losses, considering all moments of the return distribution.',
            'conviction': 'The model\'s confidence level in a trade signal or prediction.',
            'edge': 'The expected advantage or positive expectancy in a trading strategy.'
        }

        term_lower = term.lower().strip()

        if term_lower in defaults:
            return defaults[term_lower]

        # Generic fallback
        return f"{term}: Definition loading... Please ensure backend service is running."


# Global instance for easy access
_term_manager = None


def get_term_definition(term: str) -> str:
    """
    Get definition for a trading term (convenience function).

    Args:
        term: The trading term to define

    Returns:
        Definition string
    """
    global _term_manager
    if _term_manager is None:
        _term_manager = TermDefinitionManager()
    return _term_manager.get_definition(term)


def clear_cache():
    """Clear the definitions cache"""
    global _term_manager
    if _term_manager is not None:
        _term_manager.cache = {}
        _term_manager._save_cache()
