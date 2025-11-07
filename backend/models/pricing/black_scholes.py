"""
Black-Scholes Options Pricing Model
===================================

The foundational analytical model for European options pricing with closed-form solutions.
Provides fast, reliable pricing and Greeks calculations for standard options.

Features:
- Analytical pricing for calls and puts
- Complete Greeks suite (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility calculation using Newton-Raphson
- Vectorized operations for batch pricing
- Production-ready with comprehensive validation
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Union, Optional
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class BlackScholesGreeks:
    """Container for option Greeks"""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

class BlackScholesModel:
    """
    Black-Scholes model for European options pricing.

    This implementation provides:
    - Fast analytical pricing
    - Complete Greeks calculations
    - IV solver
    - Batch operations support
    """

    def __init__(self, precision: int = 6):
        """
        Initialize Black-Scholes model.

        Args:
            precision: Decimal precision for calculations
        """
        self.precision = precision

    def price(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        if T <= 0:
            # Handle expired options
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return round(price, self.precision)

    def calculate_greeks(self, S: float, K: float, T: float, r: float,
                        sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for an option.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Dictionary containing all Greeks
        """
        if T <= 0:
            # Handle expired options
            return {
                'delta': 1.0 if (option_type.lower() == 'call' and S > K) else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Common terms
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        sqrt_T = np.sqrt(T)
        exp_rT = np.exp(-r * T)

        # Calculate Greeks
        if option_type.lower() == 'call':
            delta = N_d1
            theta = (-S * n_d1 * sigma / (2 * sqrt_T) -
                    r * K * exp_rT * N_d2) / 365  # Convert to daily
            rho = K * T * exp_rT * N_d2 / 100  # Per 1% change
        else:  # put
            delta = N_d1 - 1
            theta = (-S * n_d1 * sigma / (2 * sqrt_T) +
                    r * K * exp_rT * norm.cdf(-d2)) / 365
            rho = -K * T * exp_rT * norm.cdf(-d2) / 100

        # Greeks that are same for calls and puts
        gamma = n_d1 / (S * sigma * sqrt_T)
        vega = S * n_d1 * sqrt_T / 100  # Per 1% change in volatility

        return {
            'delta': round(delta, self.precision),
            'gamma': round(gamma, self.precision),
            'vega': round(vega, self.precision),
            'theta': round(theta, self.precision),
            'rho': round(rho, self.precision)
        }

    def implied_volatility(self, S: float, K: float, T: float, r: float,
                          market_price: float, option_type: str = 'call',
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            market_price: Market price of the option
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance

        Returns:
            Implied volatility
        """
        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
        sigma = max(0.001, min(5.0, sigma))  # Bound between 0.1% and 500%

        for i in range(max_iterations):
            # Calculate price and vega
            price = self.price(S, K, T, r, sigma, option_type)
            greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
            vega = greeks['vega'] * 100  # Convert back from percentage

            # Check convergence
            price_diff = market_price - price
            if abs(price_diff) < tolerance:
                return sigma

            # Newton-Raphson update
            if abs(vega) < 1e-10:
                # Vega too small, use bisection fallback
                return self._iv_bisection(S, K, T, r, market_price, option_type)

            sigma = sigma + price_diff / vega
            sigma = max(0.001, min(5.0, sigma))  # Keep in bounds

        # If no convergence, return last estimate
        warnings.warn(f"IV did not converge after {max_iterations} iterations")
        return sigma

    def _iv_bisection(self, S: float, K: float, T: float, r: float,
                     market_price: float, option_type: str,
                     tolerance: float = 1e-6) -> float:
        """
        Fallback IV calculation using bisection method.

        Args:
            S, K, T, r, market_price, option_type: As in implied_volatility
            tolerance: Convergence tolerance

        Returns:
            Implied volatility
        """
        # Set bounds
        vol_low = 0.001
        vol_high = 5.0
        vol_mid = 0.5

        max_iterations = 100
        for i in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2
            price_mid = self.price(S, K, T, r, vol_mid, option_type)

            if abs(price_mid - market_price) < tolerance:
                return vol_mid

            if price_mid < market_price:
                vol_low = vol_mid
            else:
                vol_high = vol_mid

        return vol_mid

    def batch_price(self, strikes: np.ndarray, S: float, T: float, r: float,
                   sigma: float, option_type: str = 'call') -> np.ndarray:
        """
        Calculate prices for multiple strikes (vectorized).

        Args:
            strikes: Array of strike prices
            S: Current stock price
            T: Time to expiry
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Array of option prices
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return np.maximum(0, S - strikes)
            else:
                return np.maximum(0, strikes - S)

        # Vectorized d1 and d2 calculation
        d1 = (np.log(S / strikes) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            prices = S * norm.cdf(d1) - strikes * np.exp(-r * T) * norm.cdf(d2)
        else:
            prices = strikes * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return np.round(prices, self.precision)

    def price_surface(self, strikes: np.ndarray, maturities: np.ndarray,
                     S: float, r: float, sigma: float,
                     option_type: str = 'call') -> np.ndarray:
        """
        Calculate option prices for a grid of strikes and maturities.

        Args:
            strikes: Array of strike prices
            maturities: Array of times to maturity
            S: Current stock price
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            2D array of option prices
        """
        K_grid, T_grid = np.meshgrid(strikes, maturities)
        prices = np.zeros_like(K_grid)

        for i, T in enumerate(maturities):
            prices[i, :] = self.batch_price(strikes, S, T, r, sigma, option_type)

        return prices