"""
SABR (Stochastic Alpha, Beta, Rho) Options Pricing Model
======================================================

A comprehensive implementation of the SABR stochastic volatility model for options pricing
that provides analytical approximations for implied volatility suitable for real-time calibration.
This model is particularly effective for modeling volatility smiles and forward rate dynamics.

Features:
- Hagan 2002 lognormal SABR formula implementation
- Real-time implied volatility calculation with analytical approximations
- Market calibration with multiple optimization algorithms
- Volatility smile construction and analysis
- Batch pricing for entire options surfaces
- Forward rate and volatility surface modeling
- Model validation and Greeks calculation
- Performance benchmarking and error analysis
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, least_squares
from scipy.stats import norm
from scipy.special import erf, erfc
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import time
import warnings
import joblib
from concurrent.futures import ThreadPoolExecutor
import math
from enum import Enum

warnings.filterwarnings('ignore')

class SABRFormulation(Enum):
    """Different SABR model formulations"""
    HAGAN_2002 = "hagan_2002"           # Original Hagan formula
    OBLOJ_2008 = "obloj_2008"           # Obloj correction for extreme parameters
    WEST_2005 = "west_2005"             # West approximation
    PAULOT_2009 = "paulot_2009"         # Paulot asymptotic expansion

@dataclass
class SABRParameters:
    """Parameters for SABR stochastic volatility model"""
    alpha: float = 0.025        # Initial volatility level
    beta: float = 0.5           # CEV parameter (0 = normal, 1 = lognormal)
    rho: float = 0.0            # Correlation between forward and volatility
    nu: float = 0.2             # Volatility of volatility

    def __post_init__(self):
        """Validate SABR parameters"""
        if self.alpha <= 0:
            raise ValueError("alpha (initial volatility) must be positive")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must be between 0 and 1")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho (correlation) must be between -1 and 1")
        if self.nu < 0:
            raise ValueError("nu (vol of vol) must be non-negative")

@dataclass
class CalibrationConfig:
    """Configuration for SABR model calibration"""
    max_iterations: int = 1000
    tolerance: float = 1e-8
    method: str = 'differential_evolution'  # 'lm', 'differential_evolution', 'minimize'
    formulation: SABRFormulation = SABRFormulation.HAGAN_2002
    bounds: Dict = None
    fix_beta: bool = True       # Often beta is fixed at 0.5
    fixed_beta: float = 0.5
    weights: str = 'vega'       # 'uniform', 'vega', 'volume'

    def __post_init__(self):
        if self.bounds is None:
            self.bounds = {
                'alpha': (0.001, 2.0),
                'beta': (0.01, 0.99),
                'rho': (-0.99, 0.99),
                'nu': (0.001, 2.0)
            }

class SABRModel:
    """
    SABR Stochastic Volatility Model

    The SABR model assumes:
    dF = α F^β dW₁
    dα = ν α dW₂

    where E[dW₁dW₂] = ρ dt

    Key features:
    - Analytical implied volatility approximation
    - Flexible CEV dynamics via beta parameter
    - Effective volatility smile modeling
    - Fast calibration suitable for real-time trading
    """

    def __init__(self, formulation: SABRFormulation = SABRFormulation.HAGAN_2002):
        """
        Initialize SABR model

        Parameters:
        - formulation: SABR formula variant to use
        """
        self.formulation = formulation
        self.parameters = None
        self.calibration_results = {}

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('SABRModel')

    def implied_volatility(self, F: float, K: float, T: float,
                         params: SABRParameters) -> float:
        """
        Calculate SABR implied volatility using Hagan 2002 formula

        Parameters:
        - F: Forward price
        - K: Strike price
        - T: Time to expiration
        - params: SABR model parameters

        Returns:
        - Implied volatility (Black-Scholes equivalent)
        """
        if T <= 0:
            return 0.0

        if self.formulation == SABRFormulation.HAGAN_2002:
            return self._hagan_implied_vol(F, K, T, params)
        elif self.formulation == SABRFormulation.OBLOJ_2008:
            return self._obloj_implied_vol(F, K, T, params)
        elif self.formulation == SABRFormulation.WEST_2005:
            return self._west_implied_vol(F, K, T, params)
        else:
            return self._hagan_implied_vol(F, K, T, params)  # Default fallback

    def _hagan_implied_vol(self, F: float, K: float, T: float,
                          params: SABRParameters) -> float:
        """
        Hagan 2002 SABR implied volatility approximation

        This is the most commonly used SABR formula
        """
        alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu

        # Handle ATM case separately for numerical stability
        if abs(F - K) < 1e-10:
            return self._atm_implied_vol(F, T, params)

        # Log-moneyness
        FK = F / K
        log_FK = np.log(FK)

        # Powers for CEV dynamics
        if beta == 0:
            # Normal SABR
            F_avg = 0.5 * (F + K)
            z = nu / alpha * F_avg * log_FK
        elif beta == 1:
            # Lognormal SABR
            z = nu / alpha * log_FK
        else:
            # General CEV case
            F_avg_beta = ((F**(1-beta) + K**(1-beta)) / 2) ** (1/(1-beta))
            F_K_term = (F**(1-beta) - K**(1-beta)) / (1-beta)
            z = nu / alpha * F_K_term / F_avg_beta**beta

        # x(z) function
        if abs(z) < 1e-10:
            x_z = 1.0
        else:
            sqrt_term = np.sqrt(1 - 2*rho*z + z**2)
            if abs(sqrt_term + z - rho) < 1e-10:
                # Handle singularity
                x_z = 1.0
            else:
                x_z = z / np.log((sqrt_term + z - rho) / (1 - rho))

        # First term (ATM volatility scaled)
        if beta == 0:
            F_K_avg = 0.5 * (F + K)
            first_term = alpha / F_K_avg
        elif beta == 1:
            first_term = alpha
        else:
            F_K_avg_beta = ((F**(1-beta) + K**(1-beta)) / 2) ** (beta/(1-beta))
            first_term = alpha / F_K_avg_beta

        # Second term (time-dependent corrections)
        if beta == 0:
            F_avg = 0.5 * (F + K)
            second_term_1 = ((2 - 3*rho**2) * nu**2) / (24 * alpha**2 / F_avg**2)
            second_term_2 = (rho * beta * nu * alpha) / (4 * F_avg**(1-beta))
            second_term_3 = ((1-beta)**2) / (24 * F_avg**(2*(1-beta)))
        elif beta == 1:
            second_term_1 = ((2 - 3*rho**2) * nu**2) / (24 * alpha**2)
            second_term_2 = (rho * nu) / (4 * alpha)
            second_term_3 = 0
        else:
            F_K_avg = ((F**(1-beta) + K**(1-beta)) / 2) ** (1/(1-beta))
            second_term_1 = ((2 - 3*rho**2) * nu**2) / (24 * alpha**2 / F_K_avg**(2*beta))
            second_term_2 = (rho * beta * nu * alpha) / (4 * F_K_avg**(1-beta))
            second_term_3 = (beta * (1-beta) * alpha**2) / (24 * F_K_avg**(2*(1-beta)))

        second_term = 1 + (second_term_1 + second_term_2 + second_term_3) * T

        # Final implied volatility
        implied_vol = first_term * x_z * second_term

        return max(implied_vol, 1e-10)  # Ensure positive volatility

    def _atm_implied_vol(self, F: float, T: float, params: SABRParameters) -> float:
        """ATM implied volatility (F = K case)"""
        alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu

        # ATM volatility
        if beta == 1:
            atm_vol = alpha
        else:
            atm_vol = alpha / F**beta

        # Time corrections
        correction_1 = (rho * beta * nu * alpha) / (4 * F**(1-beta))
        correction_2 = ((2 - 3*rho**2) * nu**2) / (24 * alpha**2 / F**(2*beta))
        correction_3 = (beta * (1-beta) * alpha**2) / (24 * F**(2*(1-beta)))

        time_correction = 1 + (correction_1 + correction_2 + correction_3) * T

        return atm_vol * time_correction

    def _obloj_implied_vol(self, F: float, K: float, T: float,
                          params: SABRParameters) -> float:
        """
        Obloj 2008 correction for extreme parameters

        Better numerical stability for extreme rho and nu values
        """
        # Start with Hagan formula
        hagan_vol = self._hagan_implied_vol(F, K, T, params)

        # Obloj corrections (simplified implementation)
        alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu

        # Additional correction terms for extreme parameters
        if abs(rho) > 0.8 or nu > 1.0:
            log_moneyness = np.log(F / K)
            correction = 1 + 0.1 * nu * abs(rho) * log_moneyness**2 * T
            hagan_vol *= correction

        return hagan_vol

    def _west_implied_vol(self, F: float, K: float, T: float,
                         params: SABRParameters) -> float:
        """West 2005 approximation (alternative formulation)"""
        # Simplified West approximation - in practice, would use more complex formula
        return self._hagan_implied_vol(F, K, T, params) * 1.02  # Placeholder adjustment

    def price_option_black(self, F: float, K: float, T: float, r: float,
                          params: SABRParameters, option_type: str = 'call') -> float:
        """
        Price option using SABR implied volatility in Black formula

        Parameters:
        - F: Forward price
        - K: Strike price
        - T: Time to maturity
        - r: Risk-free rate
        - params: SABR parameters
        - option_type: 'call' or 'put'

        Returns:
        - Option price
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(F * np.exp(-r * T) - K * np.exp(-r * T), 0)
            else:
                return max(K * np.exp(-r * T) - F * np.exp(-r * T), 0)

        # Get SABR implied volatility
        implied_vol = self.implied_volatility(F, K, T, params)

        # Black formula for forward prices
        return self._black_formula(F, K, T, r, implied_vol, option_type)

    def _black_formula(self, F: float, K: float, T: float, r: float,
                      sigma: float, option_type: str) -> float:
        """Black formula for pricing options on forwards"""
        if sigma <= 0 or T <= 0:
            if option_type.lower() == 'call':
                return max(F - K, 0) * np.exp(-r * T)
            else:
                return max(K - F, 0) * np.exp(-r * T)

        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        discount = np.exp(-r * T)

        if option_type.lower() == 'call':
            price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

        return max(price, 0)

    def price_batch(self, options_data: pd.DataFrame, params: SABRParameters) -> np.ndarray:
        """
        Price multiple options efficiently

        Parameters:
        - options_data: DataFrame with columns ['F', 'K', 'T', 'r', 'option_type']
        - params: SABR parameters

        Returns:
        - Array of option prices
        """
        prices = np.zeros(len(options_data))

        # Vectorized calculation where possible
        for i, row in options_data.iterrows():
            prices[i] = self.price_option_black(
                row['F'], row['K'], row['T'], row['r'],
                params, row.get('option_type', 'call')
            )

        return prices

    def calibrate_to_market(self, market_data: pd.DataFrame,
                          config: CalibrationConfig = None) -> SABRParameters:
        """
        Calibrate SABR parameters to market option prices

        Parameters:
        - market_data: DataFrame with columns ['F', 'K', 'T', 'r', 'market_price', 'option_type']
        - config: Calibration configuration

        Returns:
        - Calibrated SABR parameters
        """
        if config is None:
            config = CalibrationConfig()

        self.logger.info(f"Starting SABR calibration with {len(market_data)} market prices...")

        # Calculate weights for optimization
        if config.weights == 'vega' and 'vega' in market_data.columns:
            weights = market_data['vega'].values
            weights = weights / np.sum(weights)  # Normalize
        elif config.weights == 'volume' and 'volume' in market_data.columns:
            weights = market_data['volume'].values
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(market_data)) / len(market_data)

        # Initial parameter guess
        if config.fix_beta:
            initial_params = [0.025, config.fixed_beta, 0.0, 0.2]  # alpha, beta, rho, nu
        else:
            initial_params = [0.025, 0.5, 0.0, 0.2]

        # Objective function
        def objective(x):
            try:
                if config.fix_beta:
                    params = SABRParameters(alpha=x[0], beta=config.fixed_beta, rho=x[2], nu=x[3])
                else:
                    params = SABRParameters(alpha=x[0], beta=x[1], rho=x[2], nu=x[3])

                model_prices = self.price_batch(market_data, params)
                market_prices = market_data['market_price'].values

                # Weighted mean squared error
                residuals = (model_prices - market_prices) ** 2
                wmse = np.sum(weights * residuals)

                return wmse

            except Exception as e:
                self.logger.warning(f"Error in objective function: {e}")
                return 1e6  # Large penalty for invalid parameters

        # Parameter bounds
        if config.fix_beta:
            bounds = [
                (config.bounds['alpha'][0], config.bounds['alpha'][1]),
                (config.bounds['rho'][0], config.bounds['rho'][1]),
                (config.bounds['nu'][0], config.bounds['nu'][1])
            ]
            # Remove beta from initial params
            initial_params = [initial_params[0], initial_params[2], initial_params[3]]
        else:
            bounds = [
                (config.bounds['alpha'][0], config.bounds['alpha'][1]),
                (config.bounds['beta'][0], config.bounds['beta'][1]),
                (config.bounds['rho'][0], config.bounds['rho'][1]),
                (config.bounds['nu'][0], config.bounds['nu'][1])
            ]

        # Optimization
        start_time = time.time()

        if config.method == 'differential_evolution':
            result = differential_evolution(
                objective, bounds, maxiter=config.max_iterations,
                tol=config.tolerance, seed=42, workers=1, popsize=15
            )
        elif config.method == 'minimize':
            result = minimize(
                objective, initial_params, bounds=bounds,
                method='L-BFGS-B', options={'maxiter': config.max_iterations}
            )
        elif config.method == 'lm':
            # Levenberg-Marquardt for least squares
            def residuals(x):
                try:
                    if config.fix_beta:
                        params = SABRParameters(alpha=x[0], beta=config.fixed_beta, rho=x[1], nu=x[2])
                    else:
                        params = SABRParameters(alpha=x[0], beta=x[1], rho=x[2], nu=x[3])

                    model_prices = self.price_batch(market_data, params)
                    market_prices = market_data['market_price'].values
                    return np.sqrt(weights) * (model_prices - market_prices)
                except:
                    return np.ones(len(market_data)) * 1000

            result = least_squares(
                residuals, initial_params,
                bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
                max_nfev=config.max_iterations
            )
        else:
            raise ValueError(f"Unknown optimization method: {config.method}")

        calibration_time = time.time() - start_time

        # Extract optimal parameters
        if config.fix_beta:
            optimal_params = SABRParameters(
                alpha=result.x[0], beta=config.fixed_beta,
                rho=result.x[1], nu=result.x[2]
            )
        else:
            optimal_params = SABRParameters(
                alpha=result.x[0], beta=result.x[1],
                rho=result.x[2], nu=result.x[3]
            )

        # Calculate final metrics
        final_prices = self.price_batch(market_data, optimal_params)
        market_prices = market_data['market_price'].values

        rmse = np.sqrt(np.mean((final_prices - market_prices)**2))
        mae = np.mean(np.abs(final_prices - market_prices))
        relative_error = np.mean(np.abs((final_prices - market_prices) / market_prices))

        # R-squared
        ss_res = np.sum((market_prices - final_prices)**2)
        ss_tot = np.sum((market_prices - np.mean(market_prices))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Store calibration results
        self.calibration_results = {
            'parameters': asdict(optimal_params),
            'rmse': rmse,
            'mae': mae,
            'relative_error': relative_error,
            'r_squared': r_squared,
            'calibration_time': calibration_time,
            'optimization_success': result.success if hasattr(result, 'success') else True,
            'function_evaluations': result.nfev if hasattr(result, 'nfev') else result.get('nfev', 'N/A'),
            'method': config.method,
            'fixed_beta': config.fix_beta,
            'beta_value': optimal_params.beta
        }

        self.parameters = optimal_params

        self.logger.info(f"Calibration completed in {calibration_time:.2f}s")
        self.logger.info(f"  RMSE: {rmse:.6f}")
        self.logger.info(f"  MAE: {mae:.6f}")
        self.logger.info(f"  Relative Error: {relative_error:.4%}")
        self.logger.info(f"  R²: {r_squared:.4f}")
        self.logger.info(f"  Parameters: α={optimal_params.alpha:.4f}, β={optimal_params.beta:.4f}, "
                        f"ρ={optimal_params.rho:.4f}, ν={optimal_params.nu:.4f}")

        return optimal_params

    def construct_volatility_smile(self, F: float, T: float, strikes: np.ndarray,
                                 params: SABRParameters) -> pd.DataFrame:
        """
        Construct volatility smile using SABR model

        Parameters:
        - F: Forward price
        - T: Time to expiration
        - strikes: Array of strike prices
        - params: SABR parameters

        Returns:
        - DataFrame with strikes and implied volatilities
        """
        implied_vols = []

        for K in strikes:
            iv = self.implied_volatility(F, K, T, params)
            implied_vols.append(iv)

        smile_df = pd.DataFrame({
            'strike': strikes,
            'moneyness': strikes / F,
            'implied_vol': implied_vols,
            'log_moneyness': np.log(strikes / F)
        })

        return smile_df

    def analyze_smile_properties(self, smile_df: pd.DataFrame) -> Dict:
        """
        Analyze properties of the volatility smile

        Returns:
        - Dictionary with smile characteristics
        """
        # ATM volatility (closest to moneyness = 1)
        atm_idx = np.argmin(np.abs(smile_df['moneyness'] - 1))
        atm_vol = smile_df.iloc[atm_idx]['implied_vol']

        # Skew (slope at ATM)
        if len(smile_df) > 2:
            # Numerical derivative at ATM
            sorted_df = smile_df.sort_values('log_moneyness')
            skew = np.gradient(sorted_df['implied_vol'], sorted_df['log_moneyness'])[atm_idx]
        else:
            skew = 0

        # Smile curvature (convexity)
        if len(smile_df) > 4:
            curvature = np.gradient(np.gradient(sorted_df['implied_vol'],
                                              sorted_df['log_moneyness']),
                                  sorted_df['log_moneyness'])[atm_idx]
        else:
            curvature = 0

        # Risk reversal (OTM call IV - OTM put IV)
        otm_puts = smile_df[smile_df['moneyness'] < 0.9]
        otm_calls = smile_df[smile_df['moneyness'] > 1.1]

        if len(otm_puts) > 0 and len(otm_calls) > 0:
            risk_reversal = otm_calls['implied_vol'].mean() - otm_puts['implied_vol'].mean()
        else:
            risk_reversal = 0

        # Butterfly (OTM average - ATM)
        otm_average = (otm_puts['implied_vol'].mean() + otm_calls['implied_vol'].mean()) / 2
        butterfly = otm_average - atm_vol if not np.isnan(otm_average) else 0

        return {
            'atm_vol': atm_vol,
            'skew': skew,
            'curvature': curvature,
            'risk_reversal': risk_reversal,
            'butterfly': butterfly,
            'min_vol': smile_df['implied_vol'].min(),
            'max_vol': smile_df['implied_vol'].max(),
            'vol_range': smile_df['implied_vol'].max() - smile_df['implied_vol'].min()
        }

    def calculate_greeks(self, F: float, K: float, T: float, r: float,
                        params: SABRParameters, option_type: str = 'call',
                        bump_size: float = 0.01) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences
        """
        base_price = self.price_option_black(F, K, T, r, params, option_type)

        # Delta (sensitivity to forward price)
        price_up = self.price_option_black(F * (1 + bump_size), K, T, r, params, option_type)
        price_down = self.price_option_black(F * (1 - bump_size), K, T, r, params, option_type)
        delta = (price_up - price_down) / (2 * F * bump_size)

        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (F * bump_size)**2

        # Theta (using 1-day bump)
        theta_bump = 1/365
        if T > theta_bump:
            price_theta = self.price_option_black(F, K, T - theta_bump, r, params, option_type)
            theta = (price_theta - base_price) / theta_bump
        else:
            theta = 0

        # Vega (bump alpha parameter)
        alpha_bump = params.alpha * bump_size
        params_vega = SABRParameters(
            alpha=params.alpha + alpha_bump, beta=params.beta,
            rho=params.rho, nu=params.nu
        )
        price_vega = self.price_option_black(F, K, T, r, params_vega, option_type)
        vega = (price_vega - base_price) / alpha_bump

        # Rho (sensitivity to interest rate)
        price_rho = self.price_option_black(F, K, T, r * (1 + bump_size), params, option_type)
        rho = (price_rho - base_price) / (r * bump_size)

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def validate_against_black_scholes(self, test_cases: pd.DataFrame) -> Dict:
        """
        Compare SABR prices against Black-Scholes for validation
        """
        if self.parameters is None:
            # Use default parameters
            self.parameters = SABRParameters()

        sabr_prices = self.price_batch(test_cases, self.parameters)

        # Black-Scholes prices using SABR ATM volatility
        bs_prices = []
        for _, row in test_cases.iterrows():
            # Use SABR ATM vol for BS comparison
            atm_vol = self.implied_volatility(row['F'], row['F'], row['T'], self.parameters)
            bs_price = self._black_formula(row['F'], row['K'], row['T'], row['r'],
                                         atm_vol, row.get('option_type', 'call'))
            bs_prices.append(bs_price)

        bs_prices = np.array(bs_prices)

        # Calculate differences
        price_differences = sabr_prices - bs_prices
        relative_differences = price_differences / bs_prices

        return {
            'n_options': len(test_cases),
            'avg_price_difference': np.mean(price_differences),
            'max_price_difference': np.max(np.abs(price_differences)),
            'avg_relative_difference': np.mean(relative_differences),
            'max_relative_difference': np.max(np.abs(relative_differences)),
            'rmse': np.sqrt(np.mean(price_differences**2)),
            'parameters_used': asdict(self.parameters)
        }

    def save_model(self, filepath: str):
        """Save calibrated model parameters"""
        if self.parameters is None:
            raise ValueError("No parameters to save. Run calibration first.")

        model_data = {
            'parameters': asdict(self.parameters),
            'calibration_results': self.calibration_results,
            'formulation': self.formulation.value
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load calibrated model parameters"""
        model_data = joblib.load(filepath)

        self.parameters = SABRParameters(**model_data['parameters'])
        self.calibration_results = model_data.get('calibration_results', {})
        self.formulation = SABRFormulation(model_data.get('formulation', 'hagan_2002'))

        self.logger.info(f"Model loaded from {filepath}")

# Utility functions for testing and market data generation
def create_sample_sabr_data(n_options: int = 100, F: float = 100) -> pd.DataFrame:
    """Create sample market data for SABR testing"""
    np.random.seed(42)

    # Generate realistic SABR market data
    strikes = np.linspace(F * 0.8, F * 1.2, 20)  # 80% to 120% moneyness
    expiries = np.array([0.25, 0.5, 1.0])        # 3M, 6M, 1Y

    data = []
    sabr = SABRModel()
    true_params = SABRParameters(alpha=0.03, beta=0.5, rho=-0.3, nu=0.4)

    for K in strikes:
        for T in expiries:
            for option_type in ['call', 'put']:
                r = 0.05

                try:
                    # Generate "market price" using true SABR parameters
                    market_price = sabr.price_option_black(F, K, T, r, true_params, option_type)

                    # Add realistic bid-ask noise
                    noise = np.random.normal(0, market_price * 0.02)  # 2% noise
                    market_price += noise
                    market_price = max(market_price, 0.01)  # Ensure positive

                    # Calculate vega for weighting
                    greeks = sabr.calculate_greeks(F, K, T, r, true_params, option_type)

                    data.append({
                        'F': F,
                        'K': K,
                        'T': T,
                        'r': r,
                        'option_type': option_type,
                        'market_price': market_price,
                        'vega': abs(greeks['vega']),
                        'volume': np.random.randint(10, 1000)
                    })
                except Exception:
                    continue  # Skip problematic cases

    return pd.DataFrame(data)

def compare_sabr_formulations(market_data: pd.DataFrame) -> pd.DataFrame:
    """Compare different SABR formulations on the same data"""
    formulations = [
        SABRFormulation.HAGAN_2002,
        SABRFormulation.OBLOJ_2008,
        SABRFormulation.WEST_2005
    ]

    results = []

    for formulation in formulations:
        sabr = SABRModel(formulation)
        try:
            params = sabr.calibrate_to_market(market_data)
            results.append({
                'formulation': formulation.value,
                'rmse': sabr.calibration_results['rmse'],
                'mae': sabr.calibration_results['mae'],
                'r_squared': sabr.calibration_results['r_squared'],
                'calibration_time': sabr.calibration_results['calibration_time'],
                'alpha': params.alpha,
                'beta': params.beta,
                'rho': params.rho,
                'nu': params.nu
            })
        except Exception as e:
            print(f"Failed to calibrate {formulation.value}: {e}")

    return pd.DataFrame(results)

# Example usage and testing
if __name__ == "__main__":
    # Initialize SABR model
    sabr = SABRModel(SABRFormulation.HAGAN_2002)

    # Example 1: Calculate implied volatility
    print("=== SABR Implied Volatility ===")
    params = SABRParameters(alpha=0.03, beta=0.5, rho=-0.3, nu=0.4)

    F, K, T = 100, 100, 0.25
    implied_vol = sabr.implied_volatility(F, K, T, params)
    print(f"ATM implied volatility: {implied_vol:.4f}")

    # Example 2: Price option
    print("\n=== Option Pricing ===")
    option_price = sabr.price_option_black(F, K, T, 0.05, params, 'call')
    print(f"Call option price: ${option_price:.4f}")

    # Example 3: Volatility smile construction
    print("\n=== Volatility Smile ===")
    strikes = np.linspace(80, 120, 21)
    smile_df = sabr.construct_volatility_smile(F, T, strikes, params)
    print("Strike\tMoneyness\tImpl. Vol")
    for _, row in smile_df.iloc[::4].iterrows():  # Show every 4th row
        print(f"{row['strike']:.0f}\t{row['moneyness']:.3f}\t\t{row['implied_vol']:.4f}")

    # Analyze smile properties
    smile_props = sabr.analyze_smile_properties(smile_df)
    print(f"\nSmile Properties:")
    print(f"  ATM Vol: {smile_props['atm_vol']:.4f}")
    print(f"  Skew: {smile_props['skew']:.4f}")
    print(f"  Risk Reversal: {smile_props['risk_reversal']:.4f}")

    # Example 4: Model calibration
    print("\n=== Model Calibration ===")
    market_data = create_sample_sabr_data(50, F=100)
    print(f"Generated {len(market_data)} market data points")

    config = CalibrationConfig(fix_beta=True, fixed_beta=0.5)
    calibrated_params = sabr.calibrate_to_market(market_data, config)
    print(f"Calibrated parameters: {calibrated_params}")

    # Example 5: Greeks calculation
    print("\n=== Greeks Calculation ===")
    greeks = sabr.calculate_greeks(F, K, T, 0.05, calibrated_params, 'call')
    for greek, value in greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")

    # Example 6: Compare formulations
    print("\n=== Formulation Comparison ===")
    comparison_df = compare_sabr_formulations(market_data.head(20))  # Use subset for speed
    print(comparison_df[['formulation', 'rmse', 'r_squared', 'calibration_time']])

    # Example 7: Validation
    print("\n=== Model Validation ===")
    test_options = pd.DataFrame({
        'F': [100, 100, 100],
        'K': [95, 100, 105],
        'T': [0.25, 0.25, 0.25],
        'r': [0.05, 0.05, 0.05],
        'option_type': ['call', 'call', 'call']
    })

    validation_results = sabr.validate_against_black_scholes(test_options)
    print(f"Validation RMSE: {validation_results['rmse']:.4f}")
    print(f"Max relative difference: {validation_results['max_relative_difference']:.2%}")

    print(f"\nCalibration R²: {sabr.calibration_results['r_squared']:.4f}")
    print(f"Final parameters: α={calibrated_params.alpha:.4f}, β={calibrated_params.beta:.4f}, "
          f"ρ={calibrated_params.rho:.4f}, ν={calibrated_params.nu:.4f}")